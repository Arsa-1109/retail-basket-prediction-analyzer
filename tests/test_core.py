"""Unit and CLI tests for Retail Basket Prediction Analyzer (Milestones 1-5)."""
import json

import pytest

from app import cli
from app.core import (
    BasketNGramModel,
    SmallBasketLM,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    parse_basket,
    support_of,
)


# Milestone 1: Generate a synthetic dataset of shopping baskets with planted itemsets.
def test_milestone_1():
    ds = milestone_1(n_baskets=400, n_items=160, seed=42)
    assert len(ds.catalog) == 160
    assert len(ds.baskets) == 400
    assert len(ds.train_baskets) == 320
    assert len(ds.test_baskets) == 80
    assert len(ds.planted) >= 12
    assert any(p.kind == "natural" for p in ds.planted)
    assert any(p.kind == "surprising" for p in ds.planted)

    # Planted itemsets should appear at least at their target support
    for p in ds.planted:
        assert support_of(ds.baskets, p.items) + 1e-9 >= p.support - (1.0 / len(ds.baskets))

    # Verbalization round-trip preserves distinct items
    texts = ds.verbalize_all(ds.baskets[:5])
    assert len(texts) == 5
    for text, basket in zip(texts, ds.baskets[:5]):
        assert text.startswith("Basket: ")
        assert set(parse_basket(text)) == set(basket)


# Milestone 2: Implement the n-gram model to predict next items in a basket.
def test_milestone_2():
    ds = milestone_1(n_baskets=400, n_items=160, seed=42)
    model = milestone_2(dataset=ds, n=3, smoothing=0.1)
    assert model.n == 3
    assert model.n_sequences > 0

    # Natural planted pair should rank partner high when seeded with the first item
    natural_pair = next(p for p in ds.planted if len(p.items) == 2 and p.kind == "natural")
    seed_item, partner_item = natural_pair.items
    scores = model.score_items([seed_item], ds.item_names)
    assert scores.shape == (len(ds.item_names),)
    preds = model.predict_next([seed_item], ds.item_names, top_k=5)
    assert len(preds) == 5
    assert all(name != seed_item for name, _ in preds)
    assert any(name == partner_item for name, _ in preds) or scores[ds.item_names.index(partner_item)] > float(scores.mean())


# Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
def test_milestone_3():
    ds = milestone_1(n_baskets=400, n_items=160, seed=42)
    lm = milestone_3(dataset=ds, embed_dim=32, hidden_dim=48, epochs=6)
    assert lm.n_params > 0
    assert len(lm.history) == 6
    assert lm.history[-1]["train_loss"] < lm.history[0]["train_loss"]
    assert lm.history[-1]["val_loss"] is not None

    natural_pair = next(p for p in ds.planted if len(p.items) == 2 and p.kind == "natural")
    seed_item, _ = natural_pair.items
    scores = lm.score_items([seed_item], ds.item_names)
    assert scores.shape == (len(ds.item_names),)
    preds = lm.predict_next([seed_item], ds.item_names, top_k=5)
    assert len(preds) == 5
    assert all(name != seed_item for name, _ in preds)


# Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
def test_milestone_4():
    ds = milestone_1(n_baskets=500, n_items=160, seed=42)
    ngram = milestone_2(dataset=ds, n=3, smoothing=0.1)
    lm = milestone_3(dataset=ds, embed_dim=32, hidden_dim=48, epochs=6)
    ev = milestone_4(dataset=ds, ngram_model=ngram, lm_model=lm, n_cases=100)
    assert ev["n_cases"] == 100
    metrics = ev["metrics"]
    assert "Popularity Baseline" in metrics
    assert any(k.startswith("N-Gram") for k in metrics)
    assert any(k.startswith("Small LM") for k in metrics)
    for row in metrics.values():
        assert 0.0 <= row["top1_acc"] <= row["recall_at_5"] <= row["recall_at_10"] <= 1.0
        assert 0.0 < row["mrr"] <= 1.0
        assert 1.0 <= row["mean_rank"] <= len(ds.catalog)


# Milestone 5: Present a comparison of the models' performance.
def test_milestone_5(capsys, tmp_path):
    report = milestone_5(n_baskets=500, n_items=160, n_cases=80, seed=42, include_ablation=True)
    assert "Retail Basket Prediction Analyzer: Model Comparison" in report["formatted_text"]
    assert "Scenario Breakdown" in report["formatted_text"]
    assert "Complexity & Capacity Ablation" in report["formatted_text"]
    assert len(report["ablation"]) == 6

    out_json = tmp_path / "comparison.json"
    cli.main(["compare", "--baskets", "400", "--cases", "60", "--no-ablation", "--save", str(out_json)])
    out = capsys.readouterr().out
    assert "Popularity Baseline" in out
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "evaluation" in payload and "scenarios" in payload

    cli.main(["stats", "--baskets", "300"])
    stats_out = capsys.readouterr().out
    assert "Planted sets:" in stats_out

    cli.main(["predict", "--baskets", "300", "--top-k", "3"])
    pred_out = capsys.readouterr().out
    assert "N-Gram top predictions:" in pred_out
    assert "Small LM top predictions:" in pred_out

    with pytest.raises(SystemExit) as exc:
        cli.main(["predict", "--baskets", "300", "--items", "NonExistent SKU 999"])
    assert exc.value.code == 1

