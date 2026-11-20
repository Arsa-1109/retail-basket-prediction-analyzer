"""Core pipeline for Retail Basket Prediction Analyzer (Milestones 1-5)."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import json
from pathlib import Path
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


# --- Milestone 3: Pretrain a small language model on the basket sequences.
def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class SmallBasketLM:
    """Small neural causal/context language model trained from scratch on shuffled basket sequences."""

    def __init__(
        self,
        embed_dim: int = 48,
        hidden_dim: int = 64,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 0.015,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ):
        if embed_dim < 4 or hidden_dim < 4:
            raise ValueError("embed_dim and hidden_dim must be at least 4")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed

        self.vocab: list[str] = []
        self._item_to_id: dict[str, int] = {}
        self.history: list[dict] = []
        self._params: dict[str, np.ndarray] = {}

    @property
    def n_params(self) -> int:
        return int(sum(arr.size for arr in self._params.values()))

    def _encode_prefixes(self, prefixes: Sequence[Sequence[str]]) -> np.ndarray:
        """Encode each visible basket prefix as a normalized multi-hot context vector."""
        x = np.zeros((len(prefixes), len(self.vocab)), dtype=np.float64)
        bos_idx = self._item_to_id[BOS_TOKEN]
        for row_idx, prefix in enumerate(prefixes):
            ids = [self._item_to_id[item] for item in prefix if item in self._item_to_id]
            if not ids:
                x[row_idx, bos_idx] = 1.0
            else:
                inv = 1.0 / math.sqrt(len(ids))
                for tok_id in ids:
                    x[row_idx, tok_id] += inv
        return x

    def _build_examples(
        self,
        baskets: Sequence[Sequence[str]],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        prefixes: list[list[str]] = []
        targets: list[int] = []
        for basket in baskets:
            distinct = sorted(set(basket))
            if len(distinct) < 2:
                continue
            perm = rng.permutation(len(distinct))
            seq = [distinct[int(i)] for i in perm]
            # Sample prefix -> next-item transitions (every item predicted from preceding items)
            for pos in range(1, len(seq)):
                target_name = seq[pos]
                if target_name in self._item_to_id:
                    prefixes.append(seq[:pos])
                    targets.append(self._item_to_id[target_name])
        x = self._encode_prefixes(prefixes)
        y = np.asarray(targets, dtype=np.int64)
        return x, y

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        p = self._params
        h0 = x @ p["E"]
        h1 = np.tanh(h0 @ p["W1"] + p["b1"])
        logits = h1 @ p["W2"] + h0 @ p["Wres"] + p["b2"]
        return logits, (h0, h1)

    def cross_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        logits, _ = self._forward(x)
        log_probs = _log_softmax(logits)
        return float(-log_probs[np.arange(len(y)), y].mean())

    def fit(
        self,
        train_baskets: Sequence[Sequence[str]],
        catalog_names: Sequence[str],
        val_baskets: Sequence[Sequence[str]] | None = None,
    ) -> "SmallBasketLM":
        """Pretrain on shuffled basket sequences with Adam and cosine learning-rate schedule."""
        self.vocab = [BOS_TOKEN] + list(catalog_names)
        self._item_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        v_size = len(self.vocab)

        rng = np.random.default_rng(self.seed)
        self._params = {
            "E": rng.normal(0.0, 0.08, size=(v_size, self.embed_dim)),
            "W1": rng.normal(0.0, 1.0 / math.sqrt(self.embed_dim), size=(self.embed_dim, self.hidden_dim)),
            "b1": np.zeros(self.hidden_dim, dtype=np.float64),
            "W2": rng.normal(0.0, 1.0 / math.sqrt(self.hidden_dim), size=(self.hidden_dim, v_size)),
            "Wres": rng.normal(0.0, 1.0 / math.sqrt(self.embed_dim), size=(self.embed_dim, v_size)),
            "b2": np.zeros(v_size, dtype=np.float64),
        }
        adam_m = {k: np.zeros_like(val) for k, val in self._params.items()}
        adam_v = {k: np.zeros_like(val) for k, val in self._params.items()}

        val_xy = None
        if val_baskets:
            val_xy = self._build_examples(val_baskets, np.random.default_rng(self.seed + 999))

        self.history = []
        step_idx = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, self.epochs + 1):
            x_all, y_all = self._build_examples(train_baskets, rng)
            order = rng.permutation(len(y_all))
            x_all, y_all = x_all[order], y_all[order]

            progress = (epoch - 1) / max(1, self.epochs - 1) if self.epochs > 1 else 1.0
            lr = self.learning_rate * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress)))

            epoch_losses: list[float] = []
            for start in range(0, len(y_all), self.batch_size):
                xb = x_all[start : start + self.batch_size]
                yb = y_all[start : start + self.batch_size]
                bsz = len(yb)

                logits, (h0, h1) = self._forward(xb)
                log_probs = _log_softmax(logits)
                loss = float(-log_probs[np.arange(bsz), yb].mean())
                epoch_losses.append(loss)

                probs = np.exp(log_probs)
                d_logits = probs
                d_logits[np.arange(bsz), yb] -= 1.0
                d_logits /= bsz

                p = self._params
                grads = {
                    "W2": h1.T @ d_logits + self.weight_decay * p["W2"],
                    "Wres": h0.T @ d_logits + self.weight_decay * p["Wres"],
                    "b2": d_logits.sum(axis=0),
                }
                d_h1 = d_logits @ p["W2"].T
                d_pre = d_h1 * (1.0 - h1 * h1)
                grads["W1"] = h0.T @ d_pre + self.weight_decay * p["W1"]
                grads["b1"] = d_pre.sum(axis=0)
                d_h0 = d_pre @ p["W1"].T + d_logits @ p["Wres"].T
                grads["E"] = xb.T @ d_h0 + self.weight_decay * p["E"]

                step_idx += 1
                bc1 = 1.0 - beta1**step_idx
                bc2 = 1.0 - beta2**step_idx
                for name, grad in grads.items():
                    adam_m[name] = beta1 * adam_m[name] + (1.0 - beta1) * grad
                    adam_v[name] = beta2 * adam_v[name] + (1.0 - beta2) * (grad * grad)
                    m_hat = adam_m[name] / bc1
                    v_hat = adam_v[name] / bc2
                    p[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            val_loss = self.cross_entropy(*val_xy) if val_xy is not None else None
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(float(np.mean(epoch_losses)), 4),
                    "val_loss": round(val_loss, 4) if val_loss is not None else None,
                    "lr": round(lr, 6),
                }
            )
        return self

    def score_items(self, visible_items: Sequence[str], candidate_names: Sequence[str]) -> np.ndarray:
        """Compute log P(candidate | visible_items) for every catalogue candidate."""
        if not self._params:
            raise RuntimeError("SmallBasketLM must be fitted before scoring.")
        x = self._encode_prefixes([visible_items])
        logits, _ = self._forward(x)
        log_probs = _log_softmax(logits)[0]
        return np.array(
            [log_probs[self._item_to_id[name]] if name in self._item_to_id else -20.0 for name in candidate_names],
            dtype=np.float64,
        )

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


def milestone_3(
    dataset: BasketDataset | None = None,
    embed_dim: int = 48,
    hidden_dim: int = 64,
    epochs: int = 10,
) -> SmallBasketLM:
    """Milestone 3 entrypoint: pretrain SmallBasketLM on shuffled basket sequences."""
    ds = milestone_1() if dataset is None else dataset
    return SmallBasketLM(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        seed=ds.seed,
    ).fit(ds.train_baskets, ds.item_names, val_baskets=ds.test_baskets)


# --- Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
def build_planted_probe_cases(
    dataset: BasketDataset,
    max_cases: int = 180,
    seed: int = 42,
) -> list[dict]:
    """Build evaluation cases from held-out baskets that contain a planted itemset."""
    rng = np.random.default_rng(seed)
    planted_sets = [
        (p, set(p.items))
        for p in dataset.planted
    ]
    cases: list[dict] = []
    for basket in dataset.test_baskets:
        distinct = sorted(set(basket))
        basket_set = set(distinct)
        for p, p_set in planted_sets:
            if p_set <= basket_set:
                target = str(rng.choice(list(p.items)))
                visible = [item for item in distinct if item != target]
                # Shuffle visible so planted co-items sit at arbitrary (often non-adjacent) positions
                perm = rng.permutation(len(visible))
                shuffled_visible = [visible[int(i)] for i in perm]
                cases.append(
                    {
                        "visible": shuffled_visible,
                        "target": target,
                        "kind": p.kind,
                        "set_size": len(p.items),
                    }
                )
                break
        if len(cases) >= max_cases:
            break
    return cases


def score_cases_with_fn(
    score_fn,
    cases: Sequence[dict],
    dataset: BasketDataset,
) -> dict:
    """Compute Top-1, Recall@K, Mean/Median Rank, MRR, MAP@5, and Repeat Share@5."""
    pos_map = {name: idx for idx, name in enumerate(dataset.item_names)}
    ranks: list[int] = []
    repeat_shares: list[float] = []

    for case in cases:
        visible = case["visible"]
        target = case["target"]
        scores = np.asarray(score_fn(visible), dtype=np.float64)
        vis_idx = [pos_map[item] for item in visible if item in pos_map]
        target_idx = pos_map[target]
        ranks.append(rank_of(scores, target_idx, vis_idx))
        top5 = recommend_top_k(scores, k=5, exclude=vis_idx)
        repeat_shares.append(repeat_share(top5, vis_idx, dataset.catalog))

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    summary = summarise(ranks, ks=(1, 3, 5, 10))
    map_at_5 = float(np.mean(np.where(ranks_arr <= 5, 1.0 / ranks_arr, 0.0))) if len(ranks) else 0.0
    return {
        "n_cases": len(cases),
        "top1_acc": round(summary["hits@1"], 4),
        "recall_at_3": round(summary["hits@3"], 4),
        "recall_at_5": round(summary["hits@5"], 4),
        "recall_at_10": round(summary["hits@10"], 4),
        "mean_rank": round(float(np.mean(ranks_arr)), 2) if len(ranks) else 0.0,
        "median_rank": round(summary["median_rank"], 2) if len(ranks) else 0.0,
        "mrr": round(summary["mrr"], 4),
        "map_at_5": round(map_at_5, 4),
        "repeat_share_at_5": round(float(np.mean(repeat_shares)), 4) if repeat_shares else 0.0,
        "ranks": ranks,
    }


def evaluate_models(
    dataset: BasketDataset | None = None,
    ngram_model: BasketNGramModel | None = None,
    lm_model: SmallBasketLM | None = None,
    n_cases: int = 250,
    seed: int = 42,
) -> dict:
    """Evaluate Popularity baseline, N-Gram model, and Small LM on identical held-out cases."""
    ds = milestone_1() if dataset is None else dataset
    ngram = milestone_2(ds) if ngram_model is None else ngram_model
    lm = milestone_3(ds) if lm_model is None else lm_model

    cases = make_cases(ds.test_baskets, ds.item_names, n_cases=min(n_cases, len(ds.test_baskets)), seed=seed)
    pop_vec = popularity_scores(ds.train_baskets, ds.item_names)
    item_to_item = ItemToItem(ds.train_baskets, ds.item_names)

    predictors = {
        "Popularity Baseline": lambda _vis: pop_vec,
        "Item-to-Item (Count)": lambda vis: item_to_item.score(vis, lift=True),
        f"N-Gram (n={ngram.n})": lambda vis: ngram.score_items(vis, ds.item_names),
        f"Small LM (d={lm.embed_dim})": lambda vis: lm.score_items(vis, ds.item_names),
    }

    metrics = {
        name: score_cases_with_fn(fn, cases, ds)
        for name, fn in predictors.items()
    }
    return {
        "n_cases": len(cases),
        "n_catalog": len(ds.catalog),
        "n_train": len(ds.train_baskets),
        "n_test": len(ds.test_baskets),
        "metrics": metrics,
        "cases": cases,
    }


def milestone_4(
    dataset: BasketDataset | None = None,
    ngram_model: BasketNGramModel | None = None,
    lm_model: SmallBasketLM | None = None,
    n_cases: int = 250,
) -> dict:
    """Milestone 4 entrypoint: evaluate next-item predictors on held-out baskets."""
    return evaluate_models(
        dataset=dataset,
        ngram_model=ngram_model,
        lm_model=lm_model,
        n_cases=n_cases,
    )


# --- Milestone 5: Present a comparison of the models' performance.
def build_scenario_cases(dataset: BasketDataset, max_per_scenario: int = 80, seed: int = 42) -> dict[str, list[dict]]:
    """Build targeted evaluation slices: contiguous vs non-adjacent vs cross-category triggers."""
    rng = np.random.default_rng(seed)
    contiguous: list[dict] = []
    non_adjacent: list[dict] = []
    surprising: list[dict] = []

    for basket in dataset.test_baskets:
        distinct = sorted(set(basket))
        bset = set(distinct)
        for p in dataset.planted:
            pset = set(p.items)
            if not (pset <= bset):
                continue
            target = str(rng.choice(list(p.items)))
            planted_vis = [x for x in p.items if x != target]
            fillers = [x for x in distinct if x not in pset]

            # Contiguous: planted partners sit right at the tail of visible
            if len(contiguous) < max_per_scenario:
                contiguous.append({"visible": fillers + planted_vis, "target": target})

            # Non-adjacent: at least 2 filler items sit between planted partners and tail
            if len(fillers) >= 2 and len(non_adjacent) < max_per_scenario:
                non_adjacent.append({"visible": planted_vis + fillers, "target": target})

            # Surprising cross-category bundle
            if p.kind == "surprising" and len(surprising) < max_per_scenario:
                mixed = planted_vis + fillers
                rng.shuffle(mixed)
                surprising.append({"visible": mixed, "target": target})
            break

    return {
        "Contiguous Planted Trigger": contiguous,
        "Non-Adjacent Planted Trigger": non_adjacent,
        "Surprising Cross-Category": surprising,
    }


def scenario_analysis(
    dataset: BasketDataset,
    ngram_model: BasketNGramModel,
    lm_model: SmallBasketLM,
    max_per_scenario: int = 80,
    seed: int = 42,
) -> dict[str, dict]:
    """Compare Popularity, N-Gram, and Small LM across structural basket scenarios."""
    scenarios = build_scenario_cases(dataset, max_per_scenario=max_per_scenario, seed=seed)
    pop_vec = popularity_scores(dataset.train_baskets, dataset.item_names)
    predictors = {
        "Popularity Baseline": lambda _vis: pop_vec,
        f"N-Gram (n={ngram_model.n})": lambda vis: ngram_model.score_items(vis, dataset.item_names),
        f"Small LM (d={lm_model.embed_dim})": lambda vis: lm_model.score_items(vis, dataset.item_names),
    }
    out: dict[str, dict] = {}
    for scenario_name, cases in scenarios.items():
        if not cases:
            continue
        out[scenario_name] = {
            model_name: score_cases_with_fn(fn, cases, dataset)
            for model_name, fn in predictors.items()
        }
    return out


def complexity_ablation(
    dataset: BasketDataset,
    cases: Sequence[dict],
) -> list[dict]:
    """Benchmark n-gram orders (2, 3, 4) vs SmallBasketLM capacities on identical held-out cases."""
    rows: list[dict] = []
    for order in (2, 3, 4):
        ng = BasketNGramModel(n=order, smoothing=0.1).fit(dataset.train_baskets, dataset.item_names, seed=dataset.seed)
        n_states = sum(len(v) for v in ng._ngram_counts[order].values())
        metrics = score_cases_with_fn(lambda vis, m=ng: m.score_items(vis, dataset.item_names), cases, dataset)
        rows.append(
            {
                "model": f"N-Gram (n={order})",
                "capacity": f"{n_states} contexts",
                "top1_acc": metrics["top1_acc"],
                "recall_at_5": metrics["recall_at_5"],
                "mrr": metrics["mrr"],
                "mean_rank": metrics["mean_rank"],
            }
        )

    for emb, hid in [(16, 24), (32, 48), (48, 64)]:
        lm = SmallBasketLM(embed_dim=emb, hidden_dim=hid, epochs=8, seed=dataset.seed).fit(
            dataset.train_baskets, dataset.item_names
        )
        metrics = score_cases_with_fn(lambda vis, m=lm: m.score_items(vis, dataset.item_names), cases, dataset)
        rows.append(
            {
                "model": f"Small LM (d={emb}, h={hid})",
                "capacity": f"{lm.n_params:,} params",
                "top1_acc": metrics["top1_acc"],
                "recall_at_5": metrics["recall_at_5"],
                "mrr": metrics["mrr"],
                "mean_rank": metrics["mean_rank"],
            }
        )
    return rows


def format_comparison_report(report: dict) -> str:
    """Format model metrics, ASCII MRR chart, scenario breakdown, and ablation into readable text."""
    ev = report["evaluation"]
    lines = [
        "=== Retail Basket Prediction Analyzer: Model Comparison ===",
        (
            f"Catalog: {ev['n_catalog']} SKUs | Train baskets: {ev['n_train']} | "
            f"Held-out baskets: {ev['n_test']} | Eval cases: {ev['n_cases']}"
        ),
        "",
        "Model                    | Top-1   | Recall@3 | Recall@5 | Recall@10 | Mean Rank | MRR    | MAP@5",
        "-------------------------+---------+----------+----------+-----------+-----------+--------+-------",
    ]
    max_mrr = max((row["mrr"] for row in ev["metrics"].values()), default=1.0) or 1.0
    for name, row in ev["metrics"].items():
        lines.append(
            f"{name:<24} | {row['top1_acc']:>6.1%} | {row['recall_at_3']:>7.1%} | "
            f"{row['recall_at_5']:>7.1%} | {row['recall_at_10']:>8.1%} | "
            f"{row['mean_rank']:>9.2f} | {row['mrr']:.4f} | {row['map_at_5']:.4f}"
        )

    lines.append("")
    lines.append("MRR Comparison (higher is better):")
    for name, row in ev["metrics"].items():
        bar_len = max(1, int(round((row["mrr"] / max_mrr) * 28)))
        lines.append(f"  {name:<24} | {'#' * bar_len:<28} {row['mrr']:.4f}")

    scenarios = report.get("scenarios", {})
    if scenarios:
        lines.append("")
        lines.append("=== Scenario Breakdown (MRR / Recall@5) ===")
        for sc_name, sc_metrics in scenarios.items():
            parts = [
                f"{m_name}: MRR={vals['mrr']:.4f}, R@5={vals['recall_at_5']:.1%}"
                for m_name, vals in sc_metrics.items()
            ]
            lines.append(f"  {sc_name} (n={next(iter(sc_metrics.values()))['n_cases']}):")
            for part in parts:
                lines.append(f"    - {part}")

    ablation = report.get("ablation", [])
    if ablation:
        lines.append("")
        lines.append("=== Complexity & Capacity Ablation ===")
        lines.append("Model                   | Capacity         | Top-1   | Recall@5 | Mean Rank | MRR")
        lines.append("------------------------+------------------+---------+----------+-----------+-------")
        for row in ablation:
            lines.append(
                f"{row['model']:<23} | {row['capacity']:<16} | {row['top1_acc']:>6.1%} | "
                f"{row['recall_at_5']:>7.1%} | {row['mean_rank']:>9.2f} | {row['mrr']:.4f}"
            )
    return "\n".join(lines)


def save_comparison_report(output_path: str | Path, report: dict) -> Path:
    """Save structured evaluation/comparison results to .json or formatted .txt."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        serializable = {
            "evaluation": {
                k: v for k, v in report["evaluation"].items() if k != "cases"
            },
            "scenarios": report.get("scenarios", {}),
            "ablation": report.get("ablation", []),
        }
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return path
    path.write_text(format_comparison_report(report) + "\n", encoding="utf-8")
    return path


def milestone_5(
    n_baskets: int = 2500,
    n_items: int = 160,
    n_cases: int = 250,
    seed: int = 42,
    include_ablation: bool = True,
) -> dict:
    """Milestone 5 entrypoint: end-to-end training, evaluation, scenario breakdown, and report."""
    ds = milestone_1(n_baskets=n_baskets, n_items=n_items, seed=seed)
    ngram = milestone_2(dataset=ds, n=3, smoothing=0.1)
    lm = milestone_3(dataset=ds, embed_dim=48, hidden_dim=64, epochs=10)
    ev = milestone_4(dataset=ds, ngram_model=ngram, lm_model=lm, n_cases=n_cases)
    scenarios = scenario_analysis(ds, ngram, lm, max_per_scenario=min(80, n_cases), seed=seed)
    ablation = complexity_ablation(ds, ev["cases"]) if include_ablation else []
    report = {
        "evaluation": ev,
        "scenarios": scenarios,
        "ablation": ablation,
        "lm_history": lm.history,
    }
    report["formatted_text"] = format_comparison_report(report)
    return report
