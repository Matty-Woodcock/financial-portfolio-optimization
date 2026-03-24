import time
import numpy as np
from collections import deque
from scipy.optimize import minimize


class DCCGARCHModel:
    """
    DCC-GARCH(1,1) following Engle (2002).

    Reuses the existing per-asset GARCHVolatilityModel instances for σ_i,t,
    then layers the DCC correlation recursion on top:

        e_t  = r_t / σ_t                          (standardised residuals)
        Q_t  = (1-a-b)·Q̄ + a·eₜ₋₁eₜ₋₁ᵀ + b·Qₜ₋₁  (pseudo-correlation)
        R_t  = diag(Q_t)^{-½} Q_t diag(Q_t)^{-½}  (true correlation)
        Σ_t  = D_t R_t D_t                         (full covariance, D=diag(σ))
    """

    def __init__(self, n_assets, window_size=250, refit_garch_every=5,
                 dcc_a=0.01, dcc_b=0.97, estimate_params=True):
        from garch_main import GARCHVolatilityModel

        self.n_assets        = n_assets
        self.window_size     = window_size
        self.dcc_a           = dcc_a
        self.dcc_b           = dcc_b
        self.estimate_params = estimate_params

        self.garch_models = [
            GARCHVolatilityModel(window_size, refit_every=refit_garch_every)
            for _ in range(n_assets)
        ]

        self.std_residuals_buffer = deque(maxlen=window_size)

        self.Q_bar = np.eye(n_assets)
        self.Q_t   = np.eye(n_assets)
        self.R_t   = np.eye(n_assets)

        self.is_initialized    = False
        self.last_volatilities = None
        self.last_cov_matrix   = None

    def add_returns(self, asset_returns):
        for i, model in enumerate(self.garch_models):
            model.add_return(asset_returns[i])

        if all(m.model_fit is not None for m in self.garch_models):
            e = np.zeros(self.n_assets)
            for i, model in enumerate(self.garch_models):
                sigma_pct = model.last_forecast if (model.last_forecast is not None and model.last_forecast > 1e-10) else 1.0
                e[i] = (asset_returns[i] * 100.0) / sigma_pct
            self.std_residuals_buffer.append(e)

    def can_forecast(self):
        return all(m.can_forecast() for m in self.garch_models)

    def initialize_from_history(self):
        """Call once after burn-in to fit Q̄ and optionally (a, b) via MLE."""
        if len(self.std_residuals_buffer) < 10:
            return

        arr = np.array(self.std_residuals_buffer)
        self.Q_bar = np.cov(arr.T) if arr.shape[0] >= 2 else np.eye(self.n_assets)

        if self.estimate_params:
            print("  [DCC] Estimating (a, b) via MLE ...", end=" ", flush=True)
            t0 = time.time()
            self.dcc_a, self.dcc_b = self._mle_dcc_params(arr, self.Q_bar)
            print(f"done in {time.time()-t0:.1f}s  a={self.dcc_a:.5f}, b={self.dcc_b:.5f}")

        self.Q_t = self.Q_bar.copy()
        for e_t in arr:
            self.Q_t = (1 - self.dcc_a - self.dcc_b) * self.Q_bar \
                       + self.dcc_a * np.outer(e_t, e_t) \
                       + self.dcc_b * self.Q_t

        self.is_initialized = True

    def forecast_covariance(self):
        """
        Returns (volatilities, cov_matrix) in decimal units, or (None, None).
        Falls back to a diagonal matrix before DCC is initialised.
        """
        if not self.can_forecast():
            return None, None

        volatilities = np.array([m.forecast_volatility() for m in self.garch_models])

        if not self.is_initialized or len(self.std_residuals_buffer) < 2:
            cov = np.diag(volatilities ** 2)
            self.last_volatilities = volatilities
            self.last_cov_matrix   = cov
            return volatilities, cov

        e_t = np.array(self.std_residuals_buffer[-1])
        self.Q_t = (1 - self.dcc_a - self.dcc_b) * self.Q_bar \
                   + self.dcc_a * np.outer(e_t, e_t) \
                   + self.dcc_b * self.Q_t

        q_diag   = np.maximum(np.diag(self.Q_t), 1e-10)
        inv_sqrt = 1.0 / np.sqrt(q_diag)
        R_t      = np.outer(inv_sqrt, inv_sqrt) * self.Q_t
        np.fill_diagonal(R_t, 1.0)
        off      = ~np.eye(self.n_assets, dtype=bool)
        R_t[off] = np.clip(R_t[off], -0.9999, 0.9999)
        self.R_t = R_t

        D   = np.diag(volatilities)
        cov = D @ R_t @ D

        self.last_volatilities = volatilities
        self.last_cov_matrix   = cov
        return volatilities, cov

    @staticmethod
    def _mle_dcc_params(arr, Q_bar):
        def neg_ll(params):
            a, b = params
            if a <= 0 or b <= 0 or a + b >= 1.0:
                return 1e10
            Q  = Q_bar.copy()
            ll = 0.0
            for t in range(1, len(arr)):
                Q = (1 - a - b) * Q_bar + a * np.outer(arr[t-1], arr[t-1]) + b * Q
                d = np.diag(Q)
                if np.any(d <= 0):
                    return 1e10
                inv_sqrt = 1.0 / np.sqrt(d)
                R  = np.outer(inv_sqrt, inv_sqrt) * Q
                e  = arr[t]
                try:
                    sign, log_det = np.linalg.slogdet(R)
                    if sign <= 0:
                        return 1e10
                    ll += log_det + float(e @ np.linalg.solve(R, e)) - float(e @ e)
                except np.linalg.LinAlgError:
                    return 1e10
            return ll

        res = minimize(neg_ll, x0=[0.01, 0.97], method="L-BFGS-B",
                       bounds=[(1e-6, 0.49), (0.50, 0.9998)],
                       options={"maxiter": 100, "ftol": 1e-7})

        if res.success and res.fun < 1e9:
            a, b = res.x
            if a + b < 1.0:
                return float(a), float(b)
        return 0.01, 0.97
