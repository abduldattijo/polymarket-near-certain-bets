# Is the 99 cent Polymarket bet safe

Code for a test of near certain bets on Polymarket. It counts how many bets priced at 99 percent or more lost in resolved markets, what a bettor would have earned, and whether a model trained on earlier prices can find bets the market gets wrong.

This repository holds code only. It holds no Polymarket data. The scripts download the data from Polymarket's public API when you run them.

## Quick runs

| Script | What it does | Needs | Time |
|---|---|---|---|
| `polymarket_near_certain.py` | Samples 1,500 resolved markets from the 500 biggest events and counts the near certain bets and their losses. | Python 3.9 or later | about 2 minutes |
| `polymarket_model_test.py` | Repeats the model test on a random sample of 5,000 resolved markets. | `numpy`, `scikit-learn` | about 10 minutes |

Both print their results and write no files.

## Full pipeline

These scripts produced the numbers in the article. Run them in this order from one folder.

1. `python3 fetch_events.py` downloads the 2,000 biggest closed events and writes `events_raw.jsonl`.
2. `python3 fetch_full.py` downloads daily price histories and writes `hist_full.jsonl`.
3. `python3 analysis.py end` runs the near certain count and the calibration checks, with prices taken before the scheduled end date. `python3 analysis.py closed` uses the recorded close time instead.
4. `python3 dig1.py` splits buying No on mid priced markets by final trading volume.
5. `python3 model_test.py 7` runs the model test with prices 7 days before the end. Use 1 or 30 for the other timings.
6. `python3 make_figures.py` redraws the two charts (needs `matplotlib`).

Install the extras with `pip install -r requirements.txt`.

Polymarket keeps adding closed markets, so a new download gives different counts from the article.

## Data and terms

The data belongs to Polymarket and is not included. Polymarket's [Terms of Use](https://polymarket.com/tos) limit some uses of its data, including selling or redistributing it. Read them before reusing the data or the results commercially. Do not commit the downloaded files.

The code is under the MIT licence (see `LICENSE`). The licence covers the code only, and not Polymarket's data.

This is a measurement of past markets and is not financial advice.
