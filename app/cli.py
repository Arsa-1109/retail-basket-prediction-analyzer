"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import sys

from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_5,
    save_comparison_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retail-basket-prediction-analyzer",
        description=(
            "Retail analysts use this tool to compare next-item prediction accuracy "
            "between an n-gram model and a small pretrained LM on simulated shopping baskets."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="check that everything is wired up")

    stats = sub.add_parser("stats", help="inspect synthetic catalogue, basket splits, and planted itemsets")
    stats.add_argument("--baskets", type=int, default=2500, help="number of simulated baskets (default: 2500)")
    stats.add_argument("--items", type=int, default=160, help="number of SKUs in catalogue (default: 160)")
    stats.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")

    pred = sub.add_parser("predict", help="predict next basket items using both n-gram and small LM")
    pred.add_argument(
        "--items",
        default=None,
        help="semicolon-separated starting items (defaults to a planted natural pair seed)",
    )
    pred.add_argument("-k", "--top-k", type=int, default=5, help="number of predictions per model (default: 5)")
    pred.add_argument("--baskets", type=int, default=2500, help="training basket count (default: 2500)")
    pred.add_argument("--catalog-size", type=int, default=160, help="catalogue SKU count (default: 160)")
    pred.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")

    for cmd_name in ("compare", "eval"):
        ev = sub.add_parser(cmd_name, help="train and compare Popularity, N-Gram, and Small LM on held-out cases")
        ev.add_argument("--baskets", type=int, default=2500, help="number of simulated baskets (default: 2500)")
        ev.add_argument("--items", type=int, default=160, help="catalogue SKU count (default: 160)")
        ev.add_argument("--cases", type=int, default=250, help="held-out evaluation cases (default: 250)")
        ev.add_argument("--no-ablation", action="store_true", help="skip the complexity/size ablation table")
        ev.add_argument("--save", default=None, help="save comparison report to .txt or .json")
        ev.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "hello":
            print("Retail Basket Prediction Analyzer is ready. Open README.md for your milestones.")
            return

        if args.command == "stats":
            ds = milestone_1(n_baskets=args.baskets, n_items=args.items, seed=args.seed)
            print(
                f"Catalog: {len(ds.catalog)} SKUs | Baskets: {len(ds.baskets)} "
                f"(train={len(ds.train_baskets)}, test={len(ds.test_baskets)}) | Planted sets: {len(ds.planted)}"
            )
            for row in ds.report()[:6]:
                print(
                    f"  [{row['kind']:<10}] target={row['target']:.3f} realised={row['realised']:.3f} "
                    f"| {' + '.join(row['items'])}"
                )
            return

        if args.command == "predict":
            ds = milestone_1(n_baskets=args.baskets, n_items=args.catalog_size, seed=args.seed)
            ngram = milestone_2(dataset=ds)
            lm = milestone_3(dataset=ds)

            if args.items:
                raw_items = [part.strip() for part in args.items.split(";") if part.strip()]
                known = set(ds.item_names)
                unknown = [item for item in raw_items if item not in known]
                if unknown:
                    raise ValueError(f"Unknown SKU(s): {unknown}. Run `stats` or omit --items for a sample basket.")
                visible = raw_items
            else:
                natural_pair = next(p for p in ds.planted if len(p.items) == 2 and p.kind == "natural")
                visible = [natural_pair.items[0]]

            print(f"Visible basket: {visible}")
            print("N-Gram top predictions:")
            for rank, (name, score) in enumerate(ngram.predict_next(visible, ds.item_names, top_k=args.top_k), start=1):
                print(f"  {rank}. {name} (logP={score:.3f})")
            print("Small LM top predictions:")
            for rank, (name, score) in enumerate(lm.predict_next(visible, ds.item_names, top_k=args.top_k), start=1):
                print(f"  {rank}. {name} (logP={score:.3f})")
            return

        if args.command in {"compare", "eval"}:
            report = milestone_5(
                n_baskets=args.baskets,
                n_items=args.items,
                n_cases=args.cases,
                seed=args.seed,
                include_ablation=not args.no_ablation,
            )
            print(report["formatted_text"])
            if args.save:
                saved = save_comparison_report(args.save, report)
                print(f"\nSaved report to: {saved}")
            return
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
