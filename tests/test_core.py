"""Unit and CLI tests for Retail Basket Prediction Analyzer (Milestones 1-5)."""
import pytest

from app.core import BasketNGramModel, milestone_1, milestone_2, parse_basket, support_of


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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Present a comparison of the models' performance.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

