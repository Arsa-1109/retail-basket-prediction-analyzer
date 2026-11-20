"""Core pipeline for Retail Basket Prediction Analyzer (Milestones 1-5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
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
from thilimarketbasket.verbalize import parse_basket, verbalize_basket


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
# Hint: Apply the n-gram model technique from 'Build a Next-Word Predictor from Scratch'.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


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
