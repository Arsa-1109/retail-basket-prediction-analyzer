# Retail Basket Prediction Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/294pmsdbgv/badge.svg)](https://ask.thili.ai/b/294pmsdbgv)

A command-line benchmarking and simulation tool for retail analysts to compare next-item prediction performance between a smoothed $n$-gram model and a small pretrained language model trained on synthetic shopping baskets with planted itemsets.

---

## Key Features

- **Synthetic Basket Generator with Ground-Truth Planted Itemsets:** Uses `thilimarketbasket` to construct a multi-brand, multi-pack grocery catalogue (`160` SKUs across `127` base products) and plant both *natural* bundles (e.g. Tea + Milk + Biscuits + Rusk) and *surprising* cross-category bundles into realistic popularity-weighted background baskets.
- **Smoothed $n$-Gram Next-Item Predictor:** Implements `BasketNGramModel` with configurable order $n$, add-$k$ smoothing, and interpolated backoff ($n \to 2 \to 1$) over shuffled basket item sequences.
- **Pretrained Small Language Model (`SmallBasketLM`):** Implements a pure-NumPy neural language model trained from scratch with mini-batch Adam, weight decay, and a cosine learning-rate schedule on shuffled basket prefixes.
- **Multi-Metric Ranking Evaluation & Scenario Breakdown:** Benchmarks **Popularity Baseline**, **Item-to-Item (Count)**, **N-Gram**, and **Small LM** on Top-1 Accuracy, Recall@3 / @5 / @10, Mean Rank, MRR, and MAP@5 across contiguous, non-adjacent, and cross-category scenarios, plus a capacity ablation.

---

## Project Structure

```text
retail-basket-prediction-analyzer/
├── app/
│   ├── __init__.py
│   ├── cli.py           # CLI entrypoints: hello, stats, predict, compare, eval
│   ├── core.py          # BasketDataset, BasketNGramModel, SmallBasketLM, evaluation & reports
│   ├── llm.py           # Optional OpenAI-compatible chat client (.env)
│   └── storage.py       # Local JSON persistence helper
├── tests/
│   ├── test_core.py     # Milestone 1-5 unit, model, evaluation, and CLI tests
│   └── test_smoke.py    # Starter smoke tests
├── .github/workflows/ci.yml
├── .env.example
├── pytest.ini
└── requirements.txt
```

---

## Quickstart

Requires **Python 3.12+**.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli hello
pytest -q
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.cli hello
.\.venv\Scripts\python.exe -m pytest -q
```

---

## CLI Reference

| Command | Description | Example |
| :--- | :--- | :--- |
| `hello` | Verify environment and CLI wiring | `python -m app.cli hello` |
| `stats` | Inspect catalogue SKUs, basket splits, and planted itemsets | `python -m app.cli stats --baskets 2500` |
| `predict` | Compare top-$k$ next-item predictions for a visible basket | `python -m app.cli predict --items "Chai Chaska Tea Powder 500 g"` |
| `compare` / `eval` | Run held-out evaluation, scenario breakdown, and ablation | `python -m app.cli compare --save outputs/report.txt` |

---

## Sample Output

```text
$ python -m app.cli predict
Visible basket: ['Chai Chaska Tea Powder 500 g']
N-Gram top predictions:
  1. Sunrise Bites Glucose Biscuits 200 g (logP=-3.342)
  2. Chai Chaska Sugar 1 kg (logP=-3.471)
  3. Tomato 1 kg (logP=-3.617)
  4. Amulya Eggs 12 pack (logP=-3.725)
  5. Amulya Milk 500 ml (logP=-3.737)
Small LM top predictions:
  1. Sunrise Bites Glucose Biscuits 200 g (logP=-2.410)
  2. Amulya Milk 500 ml (logP=-2.519)
  3. Crunchy Crust Rusk 200 g (logP=-2.754)
  4. Chai Chaska Sugar 1 kg (logP=-2.845)
  5. Onion 1 kg (logP=-3.828)
```

---

## Baseline Evaluation, Model Comparison & Limitations

Evaluated on **2,500 simulated baskets** (`2,000` train / `500` held-out test, `160` SKUs, `19` planted itemsets, `250` held-out single-item prediction cases) via `python -m app.cli compare`:

### Overall Held-Out Ranking Performance

| Model | Top-1 | Recall@3 | Recall@5 | Recall@10 | Mean Rank | MRR | MAP@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 4.4% | 13.6% | 19.2% | 33.2% | 32.10 | 0.1369 | 0.0950 |
| **Item-to-Item (Count)** | 8.4% | 12.4% | 14.0% | 19.2% | 56.43 | 0.1298 | 0.1051 |
| **N-Gram (`n=3`)** | 7.6% | 14.4% | 19.6% | 28.4% | 35.99 | 0.1524 | 0.1183 |
| **Small LM (`d=48`)** | **17.6%** | **25.6%** | **27.6%** | **38.0%** | **31.37** | **0.2495** | **0.2148** |

### Scenario Breakdown (MRR / Recall@5)

| Scenario | Popularity Baseline | N-Gram (`n=3`) | Small LM (`d=48`) |
| :--- | :---: | :---: | :---: |
| **Contiguous Planted Trigger (`n=80`)** | 0.1763 / 28.7% | 0.6887 / 80.0% | **0.7265 / 92.5%** |
| **Non-Adjacent Planted Trigger (`n=80`)** | 0.1768 / 28.7% | 0.1209 / 16.2% | **0.7202 / 92.5%** |
| **Surprising Cross-Category (`n=30`)** | 0.0362 / 0.0% | 0.1537 / 20.0% | **0.5569 / 80.0%** |

### Capacity & Complexity Ablation

| Model | Capacity | Top-1 | Recall@5 | Mean Rank | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `N-Gram (n=2)` | 10,877 contexts | 8.8% | 22.4% | 35.09 | 0.1730 |
| `N-Gram (n=3)` | 32,612 contexts | 7.6% | 19.6% | 35.99 | 0.1524 |
| `N-Gram (n=4)` | 35,271 contexts | 6.8% | 19.6% | 36.02 | 0.1481 |
| `Small LM (d=16, h=24)` | 9,585 params | 14.4% | 28.0% | 30.80 | 0.2272 |
| `Small LM (d=32, h=48)` | 19,777 params | 16.0% | **30.0%** | **30.58** | 0.2412 |
| `Small LM (d=48, h=64)` | 29,057 params | **16.8%** | 28.4% | 31.78 | **0.2466** |

**Key Takeaways:**
- Because shopping baskets are unordered sets, the $n$-gram model only works well when a planted partner sits immediately at the end of the visible prefix (`0.6887` MRR). When unrelated groceries separate planted items (`Non-Adjacent`), the trigram window loses the signal (`0.1209` MRR, below the popularity baseline), whereas `SmallBasketLM` pools across the full basket and retains **`0.7202` MRR / `92.5%` Recall@5**.
- Increasing $n$-gram order from `2` to `4` triples context states (`10,877` $\to$ `35,271`) and hurts generalisation (`0.1730` $\to$ `0.1481` MRR), while scaling `SmallBasketLM` steadily improves MRR (`0.2272` $\to$ `0.2466`) with under `30k` parameters.

### Known Limitations

1. **Synthetic basket distribution:** Baskets are simulated from hand-weighted catalogue shares and planted bundles (`thilimarketbasket`), so real-world seasonal drift, price elasticity, and customer loyalty segments are not modelled.
2. **Closed SKU vocabulary:** `SmallBasketLM` uses a fixed embedding table over the generated catalogue; unseen SKUs added after training receive fallback log-probability (`-20.0`) until retrained.
3. **Order-invariant pooling in `SmallBasketLM`:** Pooling prefix embeddings over the visible set makes the model robust to shuffled basket order, but intentionally ignores checkout scan order if sequential ordering ever mattered (e.g. recipe step sequences).
4. **CPU-scale capacity vs. Transformer LMs:** `SmallBasketLM` is implemented in pure NumPy (~29k parameters) for fast, zero-GPU execution in CI rather than multi-layer self-attention over subword tokens (`mlx_lm`).

---

## Testing

```bash
pytest -v
```

All 8 unit, model, evaluation, and CLI tests run completely offline on CPU with no API keys required.

## Configuration

Optional LLM endpoint settings live in `.env` (see `.env.example`), which is ignored by git (`.gitignore`). All synthetic data generation, training, evaluation, and CLI workflows operate offline without `.env`.
