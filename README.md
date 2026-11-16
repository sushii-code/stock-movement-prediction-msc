# Do Deep Learning Gains in Stock Prediction Survive Sound Evaluation?

**A controlled audit of evaluation protocol choices**
MSc Artificial Intelligence dissertation · Brunel University of London · 2025–2026 · **Graded A+**
Author: Gnana Sushmitha Tammisetti · Supervisor: Professor Zidong Wang

---

## In one sentence
Many studies report near-perfect deep learning results for stock prediction. This project tests whether those results are **real**, or whether they come from **how the models were evaluated**, for example data leakage.

## Why this matters
A model can look excellent on paper because of the way it was tested, not because it can really predict. This project measures, in percentage points, how much three common evaluation choices change the reported result:

1. **Target formulation:** predicting the *price level* vs predicting the *direction* (up or down)
2. **Data splitting:** a *random* train/test split vs a *time-ordered* (walk-forward) split
3. **Feature scaling:** scaling on the *whole dataset* (leakage) vs scaling on the *training data only*

## Research questions
- **RQ1:** How big is the gap between leakage-prone and methodologically sound evaluation?
- **RQ2:** Do LSTM, GRU and CNN-LSTM beat simple benchmarks under sound evaluation?
- **RQ3:** How much does each choice (target, split, scaling) contribute on its own?
- **RQ4:** Are any gains statistically and economically meaningful after accounting for random seeds and transaction costs?

## Data
- Daily data downloaded once with **yfinance**, 2005–2025
- 5 assets from different parts of the US market:

| Ticker | Company / fund | Sector |
|---|---|---|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corporation | Technology |
| JPM | JPMorgan Chase & Co. | Financials |
| XOM | Exxon Mobil Corporation | Energy |
| SPY | SPDR S&P 500 ETF | Broad market |

- **2024–2025 is a strictly separated final holdout**, used once at the end and never for tuning.
- Raw data is **not** included in this repository. Notebook 01 downloads it.

## Method
- **Controlled design:** the data, features and model settings stay fixed; only one evaluation choice changes at a time.
- **Two protocols:** a leakage-prone protocol and a methodologically sound protocol.
- **Expanding-window walk-forward validation:** models are always trained on the past and tested on the future.
- **Models:** naive benchmarks (majority class, copy-yesterday), Logistic Regression / OLS, LSTM, GRU, CNN-LSTM.
- **Evaluation:** RMSE, R², directional accuracy, MCC, statistical significance tests with confidence intervals, a power analysis, repeated random seeds, and an economic backtest with transaction costs (5, 10 and 20 basis points).

## Key findings
| Finding | Result |
|---|---|
| Price-level models look almost perfect | R² ≈ **0.998** |
| …but they can't predict direction | Mean directional accuracy **0.4994** (a coin flip) |
| Copy-yesterday benchmark | Lower RMSE than the price models for **all 5 assets** |
| Overall leakage-prone vs sound protocol gap | **+0.998 pp** (95% CI +0.515 to +1.579) |
| Effect of splitting (linear models) | **+1.40 pp** |
| Effect of target formulation (linear models) | **−1.37 pp** (the two nearly cancel out) |
| Effect of scaling (linear models / neural models) | **0.000 pp** / **+0.543 pp** |
| Random seed variation | About **1.2×** the size of the protocol effect |
| Neural models vs benchmarks | **No reliable advantage** (MCC confidence intervals include zero) |
| Economic test | **Buy-and-hold wins** at every transaction cost tested |

**Takeaway:** reported performance depends heavily on evaluation design, and different mistakes can hide each other. Always check *how* a number was measured before trusting it.

## Notebooks (run in order)
Click a number to open the notebook in Google Colab.

| # | Notebook | What it does |
|---|---|---|
| [01](https://colab.research.google.com/github/sushii-code/stock-movement-prediction-msc/blob/main/notebooks/01_baseline_persistence.ipynb) | Data download and baseline benchmarks | Downloads AAPL, MSFT, JPM, XOM, SPY with yfinance; majority-class and persistence benchmarks |
| [02](https://colab.research.google.com/github/sushii-code/stock-movement-prediction-msc/blob/main/notebooks/02_feature_engineering.ipynb) | Feature engineering | 20 backward-looking predictors and 3 targets, plus 3 leakage checks |
| [03](https://colab.research.google.com/github/sushii-code/stock-movement-prediction-msc/blob/main/notebooks/03_walkforward_split.ipynb) | Splitting protocols | Sets aside the 2024–2025 holdout; walk-forward vs random split; scaler leakage |
| [04](https://colab.research.google.com/github/sushii-code/stock-movement-prediction-msc/blob/main/notebooks/04_target_and_ablation_v2.ipynb) | Target formulation and ablation | Price vs return targets; date-matched split × target × scaling ablation (Tables 4.2–4.4) |
| [05](https://colab.research.google.com/github/sushii-code/stock-movement-prediction-msc/blob/main/notebooks/05_neural_protocols.ipynb) | Neural architectures | LSTM, GRU, CNN-LSTM under both protocols, seeds 42–46 (GPU) |
| [06](https://colab.research.google.com/github/sushii-code/stock-movement-prediction-msc/blob/main/notebooks/06_significance_testing.ipynb) | Significance testing | Moving-block bootstrap, power analysis, Benjamini–Hochberg FDR |
| [07](https://colab.research.google.com/github/sushii-code/stock-movement-prediction-msc/blob/main/notebooks/07_economic_evaluation.ipynb) | Economic evaluation | Long-only backtest at 0/5/10/20 bps vs persistence and buy-and-hold; regimes |
| [08](https://colab.research.google.com/github/sushii-code/stock-movement-prediction-msc/blob/main/notebooks/08_final_holdout.ipynb) | Final holdout | One-time evaluation on 2024–2025 (run once only) |
| [09](https://colab.research.google.com/github/sushii-code/stock-movement-prediction-msc/blob/main/notebooks/09_supervisor_requests.ipynb) | Extra analyses | Neural holdout, neural scaling effect, balanced accuracy (GPU) |

## How to run
1. Open the notebooks in **Google Colab** and run them **in order, 01 to 09**.
2. Each notebook saves its outputs to Google Drive (`MyDrive/CS5500_dissertation/` with `data/`, `features/`, `results/` and `figures/` subfolders), and the next notebook reads them.
3. Notebooks **05** and **09** train neural networks: use a **T4 GPU** and set `QUICK_TEST = False` for the full run (about 50 and 70 minutes).
4. Notebook **08** is the final holdout evaluation and was run **once only**.
5. Data: daily yfinance data, 3 Jan 2005 – 30 Dec 2025, downloaded once and saved as CSV. Re-downloading later may give slightly different adjusted prices.
6. Main library versions: pandas 2.2.3, yfinance 0.2.66 (see `requirements.txt`).

## Tools
Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · SciPy · yfinance · Matplotlib · Google Colab

## What I learned
- How to design **controlled experiments** that change one thing at a time
- How **data leakage** quietly inflates results, and how walk-forward validation prevents it
- Why a **strictly separated holdout** set matters
- Why **simple benchmarks**, **repeated runs** and **costs** must be part of any evaluation
- To use **version control from day one**, which is why this project now lives on GitHub

## Licence
Code: MIT License. The dissertation text remains the author's work.
