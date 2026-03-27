# Portfolio Optimisation — RL & Evolutionary Algorithms

**Disclaimer:** I will do a live demonstration during the inspection so it is not required to setup and run this code.

This project compares two adaptive approaches to portfolio management — a reinforcement learning agent and an evolutionary optimiser — tested across a universe of 11 assets including equities, ETFs, bonds, gold, and cash.

**Note:** Running this project requires installing the Python dependencies listed across the source files. Price data is included as a CSV so no API key is needed to get started. A Tiingo API key is only required if you want to refresh or extend the price data.

---

## How to Use

### Running a single experiment

The simplest entry point is `rl_main.py`, which trains a single PPO reinforcement learning agent and evaluates it on the held-out test set. Similarly, `ngsa2_main.py` and `ngsa2_static.py` run the rolling and static NSGA-II variants respectively. Each prints performance metrics to the console when finished.

To compare against an uninformed benchmark, `baselines.py` runs an equal-weight portfolio across the same test period.

### Running the full pipeline

`moo_main.py` trains all strategy variants, applies a Pareto dominance filter across return, maximum drawdown, and portfolio concentration, and produces a visualisation for each non-dominated strategy. This is the recommended starting point for understanding the full set of results.

### Running a large-scale evaluation

`evaluation_framework.py` and `moo_evaluation_framework.py` repeat the full pipeline across many independent runs and aggregate results into a summary table and CSV. The number of runs can be configured with N_RUNS. Results are saved to the `results/` folder, organised by timestamp.

---

## Configurations

The project supports two main configurations, controlled via `config/constants.py`:

- **Large** — trains on data from June 2012 to December 2025 with a 65/10/25 train–eval–test split.
- **Small** — trains on data from June 2017 to December 2024 with a 70/15/15 split.

To switch between them, update `DATA_START`, `DATA_END`, `TRAIN_SPLIT`, and `EVAL_SPLIT` in `constants.py`. The corresponding split ratios are commented out in the file for convenience.

An **extra test set** can also be defined by setting `EXTRA_TEST_END` in `constants.py`. When set, all pipelines will automatically evaluate the non-dominated models on this additional out-of-sample period and report results separately.

If constants.py is changed, run price_data_reader_writer.py to regenerate the price data and GARCH volatility arrays before running any experiments.

It is recommended to only switch between the Large and Small configurations and not change any of the individual values.

---

## Output

Each evaluation produces printed metrics including return, Sharpe ratio, max drawdown, HHI, transaction costs, and bootstrap p-value. Where applicable, portfolio weight visualisations show how allocations evolve across the test period. The additional statistical tests against the equal-weight baseline (bootstrap and Jobson-Korkie) are only computed when running moo_evaluation_framework.py.
