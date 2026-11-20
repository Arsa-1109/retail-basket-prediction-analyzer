"""Core pipeline for Retail Basket Prediction Analyzer (Milestones 1-5)."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from thilimarketbasket.evaluate import make_cases, popularity_scores, rank_of, summarise
from thilimarketbasket.generate import (
    NATURAL,
    Item,
    Planted,
    background_baskets,
    make_catalog,
    plant_all,
    planted_report,
    popularity,
    support_of,
)
from thilimarketbasket.recommend import ItemToItem, repeat_share, top_k as recommend_top_k
from thilimarketbasket.verbalize import parse_basket, verbalize_basket

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


@dataclass
class BasketDataset:
    """Simulated shopping baskets with planted itemsets and deterministic train/test split."""

    catalog: list[Item]
    shares: np.ndarray
    planted: list[Planted]
    baskets: list[list[str]]
    train_baskets: list[list[str]]
    test_baskets: list[list[str]]
    seed: int

    @property
    def item_names(self) -> list[str]:
        return [item.name for item in self.catalog]

    def verbalize_all(self, baskets: Sequence[Sequence[str]] | None = None, seed: int | None = None) -> list[str]:
        source = self.baskets if baskets is None else baskets
        base_seed = self.seed if seed is None else seed
        return [verbalize_basket(b, seed=base_seed + idx) for idx, b in enumerate(source)]

    def report(self) -> list[dict]:
        return planted_report(self.baskets, self.planted)


def build_planted_itemsets(
    catalog: list[Item],
    shares: np.ndarray,
    seed: int = 42,
    supports: tuple[float, ...] = (0.08, 0.05, 0.035, 0.025),
) -> list[Planted]:
    """Select natural and cross-category surprising itemsets across sizes 2..5 using one SKU per base."""
    rng = np.random.default_rng(seed)
    by_base: dict[str, list[int]] = {}
    category_of: dict[str, str] = {}
    for pos, item in enumerate(catalog):
        by_base.setdefault(item.base, []).append(pos)
        category_of[item.base] = item.category

    chosen_sku: dict[str, str] = {}

    def sku_for(base: str) -> str:
        if base not in chosen_sku:
            positions = np.array(by_base[base], dtype=int)
            probs = shares[positions] / shares[positions].sum()
            chosen_sku[base] = catalog[int(rng.choice(positions, p=probs))].name
        return chosen_sku[base]

    planted: list[Planted] = []
    used_sets: list[set[str]] = []

    def can_add(items: tuple[str, ...]) -> bool:
        s = set(items)
        return not any(s <= other or other <= s for other in used_sets)

    # Natural itemsets from thilimarketbasket.generate.NATURAL
    for size in sorted(NATURAL, reverse=True):
        for sup, bases in zip(supports, NATURAL[size]):
            if all(b in by_base for b in bases):
                items = tuple(sku_for(b) for b in bases)
                if can_add(items):
                    planted.append(Planted(items=items, support=sup, kind="natural"))
                    used_sets.append(set(items))

    # Surprising cross-category itemsets (sizes 2..4) to test non-obvious co-occurrence learning
    categories = sorted(set(category_of.values()))
    bases_by_cat: dict[str, list[str]] = {}
    for base, cat in category_of.items():
        bases_by_cat.setdefault(cat, []).append(base)

    for size, sup in [(4, 0.03), (3, 0.04), (2, 0.06)]:
        for _ in range(250):
            picked_cats = list(rng.choice(categories, size=size, replace=False))
            picked_bases = [str(rng.choice(bases_by_cat[c])) for c in picked_cats]
            items = tuple(sku_for(b) for b in picked_bases)
            if can_add(items):
                planted.append(Planted(items=items, support=sup, kind="surprising"))
                used_sets.append(set(items))
                break

    return planted


def generate_basket_dataset(
    n_baskets: int = 2500,
    n_items: int = 160,
    brands_per_product: int = 3,
    seed: int = 42,
    test_ratio: float = 0.2,
    mean_size: float = 6.5,
) -> BasketDataset:
    """Generate simulated retail baskets with planted itemsets and an 80/20 train/test split."""
    if n_baskets < 50:
        raise ValueError("n_baskets must be at least 50")
    if not 0.05 <= test_ratio <= 0.5:
        raise ValueError("test_ratio must be between 0.05 and 0.5")

    catalog = make_catalog(n_items=n_items, brands_per_product=brands_per_product, seed=seed)
    shares = popularity(catalog)
    background = background_baskets(n_baskets, catalog, shares, seed=seed, mean_size=mean_size)
    planted = build_planted_itemsets(catalog, shares, seed=seed)
    baskets = plant_all(background, planted, catalog, seed=seed)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(baskets))
    split_idx = int(len(baskets) * (1.0 - test_ratio))
    train_baskets = [baskets[int(i)] for i in perm[:split_idx]]
    test_baskets = [baskets[int(i)] for i in perm[split_idx:]]

    return BasketDataset(
        catalog=catalog,
        shares=shares,
        planted=planted,
        baskets=baskets,
        train_baskets=train_baskets,
        test_baskets=test_baskets,
        seed=seed,
    )


# --- Milestone 1: Generate a synthetic dataset of shopping baskets with planted itemsets.
def milestone_1(
    n_baskets: int = 2500,
    n_items: int = 160,
    seed: int = 42,
) -> BasketDataset:
    """Milestone 1 entrypoint: generate synthetic shopping baskets with planted itemsets."""
    return generate_basket_dataset(n_baskets=n_baskets, n_items=n_items, seed=seed)


# --- Milestone 2: Implement the n-gram model to predict next items in a basket.
class BasketNGramModel:
    """Smoothed n-gram language model over ordered basket item sequences with backoff."""

    def __init__(self, n: int = 3, smoothing: float = 0.1):
        if n < 1:
            raise ValueError("n must be at least 1")
        if smoothing <= 0:
            raise ValueError("smoothing (k) must be > 0")
        self.n = n
        self.smoothing = float(smoothing)
        self.vocab: list[str] = []
        self._vocab_index: dict[str, int] = {}
        self._ngram_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {}
        self._context_totals: dict[int, dict[tuple[str, ...], int]] = {}
        self.n_sequences = 0

    def fit(
        self,
        baskets: Sequence[Sequence[str]],
        catalog_names: Sequence[str],
        shuffle_epochs: int = 2,
        seed: int = 42,
    ) -> "BasketNGramModel":
        """Train n-gram counts across orders 1..n on basket item sequences."""
        self.vocab = list(catalog_names) + [EOS_TOKEN]
        self._vocab_index = {tok: idx for idx, tok in enumerate(self.vocab)}
        self._ngram_counts = {order: defaultdict(Counter) for order in range(1, self.n + 1)}
        self._context_totals = {order: defaultdict(int) for order in range(1, self.n + 1)}

        rng = np.random.default_rng(seed)
        self.n_sequences = 0

        for _ in range(max(1, shuffle_epochs)):
            for basket in baskets:
                distinct = sorted(set(basket))
                if not distinct:
                    continue
                perm = rng.permutation(len(distinct))
                seq = [distinct[int(i)] for i in perm]
                self.n_sequences += 1
                for order in range(1, self.n + 1):
                    pad = order - 1
                    padded = ([BOS_TOKEN] * pad) + seq + [EOS_TOKEN]
                    for idx in range(pad, len(padded)):
                        ctx = tuple(padded[idx - pad : idx])
                        target = padded[idx]
                        self._ngram_counts[order][ctx][target] += 1
                        self._context_totals[order][ctx] += 1
        return self

    def _normalize_context(self, visible_items: Sequence[str], order: int) -> tuple[str, ...]:
        pad = order - 1
        if pad <= 0:
            return ()
        items = list(visible_items)
        if len(items) < pad:
            items = ([BOS_TOKEN] * (pad - len(items))) + items
        else:
            items = items[-pad:]
        return tuple(items)

    def score_items(self, visible_items: Sequence[str], candidate_names: Sequence[str]) -> np.ndarray:
        """Compute interpolated smoothed log-probabilities for every candidate item."""
        if not self.vocab:
            raise RuntimeError("BasketNGramModel must be fitted before scoring.")
        vocab_size = len(self.vocab)
        weights_by_order = {1: 0.15, 2: 0.30, 3: 0.55}

        probs = np.zeros(len(candidate_names), dtype=np.float64)
        active_weight = 0.0

        for order in range(1, self.n + 1):
            ctx = self._normalize_context(visible_items, order=order)
            total = self._context_totals[order].get(ctx, 0)
            if order > 1 and total == 0:
                continue
            w = weights_by_order.get(order, 0.55 if order == self.n else 0.25)
            counts = self._ngram_counts[order].get(ctx)
            denom = total + self.smoothing * vocab_size
            order_probs = np.empty(len(candidate_names), dtype=np.float64)
            for idx, name in enumerate(candidate_names):
                cnt = counts[name] if counts else 0
                order_probs[idx] = (cnt + self.smoothing) / denom
            probs += w * order_probs
            active_weight += w

        if active_weight > 0:
            probs /= active_weight
        return np.log(np.maximum(probs, 1e-12))

    def predict_next(
        self,
        visible_items: Sequence[str],
        candidate_names: Sequence[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return top-k unseen candidate items ranked by log-probability."""
        scores = self.score_items(visible_items, candidate_names)
        seen = set(visible_items)
        exclude = [idx for idx, name in enumerate(candidate_names) if name in seen]
        top_indices = recommend_top_k(scores, k=top_k, exclude=exclude)
        return [(candidate_names[idx], float(scores[idx])) for idx in top_indices]


def milestone_2(
    dataset: BasketDataset | None = None,
    n: int = 3,
    smoothing: float = 0.1,
) -> BasketNGramModel:
    """Milestone 2 entrypoint: fit the smoothed n-gram next-item predictor."""
    ds = milestone_1() if dataset is None else dataset
    return BasketNGramModel(n=n, smoothing=smoothing).fit(ds.train_baskets, ds.item_names, seed=ds.seed)


# --- Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
# Hint: Adapt the small language model training from 'Teach a Language Model What Sells Together'.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
# Hint: Define and calculate appropriate next-item prediction metrics like accuracy or rank.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Present a comparison of the models' performance.
# Hint: Summarize and visualize the evaluation results for a clear comparison.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
