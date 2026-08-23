"""Command-line entry point for the reproducible ToxLens workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightning as L
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

from src import deep_tox


ROOT = Path(__file__).resolve().parent
PRIMARY_TASKS = (
    "Ames",
    "LD50_Zhu",
    "hERG_Karim",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "SR-ARE",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments in O(k) time and space for k arguments."""
    parser = argparse.ArgumentParser(description="Run the ToxLens classification pipeline.")
    parser.add_argument(
        "stage",
        choices=("featurise", "train", "evaluate", "baselines", "all"),
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT / "data" / "splits",
        help="Directory containing the frozen train.csv, validation.csv, and test.csv files.",
    )
    parser.add_argument(
        "--graph-cache",
        type=Path,
        default=ROOT / "artifacts" / "pyg_graphs_class.pkl",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force-refeaturise", action="store_true")
    return parser.parse_args()


def load_frozen_split_frame(split_dir: Path) -> pd.DataFrame:
    """Load and audit the three frozen folds in O(n) time and space."""
    split_parts: list[pd.DataFrame] = []
    seen_smiles: set[str] = set()
    for filename, split_name in (
        ("train.csv", "train"),
        ("validation.csv", "validation"),
        ("test.csv", "test"),
    ):
        split_path = split_dir / filename
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing frozen split file: {split_path}")
        part = pd.read_csv(split_path)
        if "smiles" not in part.columns or part["smiles"].isna().any():
            raise RuntimeError(f"Invalid SMILES column in {split_path}")
        current_smiles = set(part["smiles"].astype(str))
        overlap = seen_smiles.intersection(current_smiles)
        if overlap:
            raise RuntimeError(
                f"Frozen folds overlap by {len(overlap)} exact SMILES before featurisation."
            )
        seen_smiles.update(current_smiles)
        part = part.copy()
        part["benchmark_split"] = split_name
        split_parts.append(part)

    return pd.concat(split_parts, ignore_index=True)


def load_or_build_graphs(
    split_dir: Path,
    cache_path: Path,
    *,
    force: bool,
    require_frozen_cache: bool,
) -> tuple:
    """Load cached graphs or featurise the frozen folds in O(n) time and space."""
    frame = load_frozen_split_frame(split_dir)
    if cache_path.exists() and not force:
        bundle = joblib.load(cache_path)
        if require_frozen_cache:
            expected_parts = {
                name: part["smiles"].astype(str).tolist()
                for name, part in frame.groupby("benchmark_split", sort=False)
            }
            cached_parts = {
                "train": bundle[3]["smiles"].astype(str).tolist(),
                "validation": bundle[4]["smiles"].astype(str).tolist(),
                "test": bundle[5]["smiles"].astype(str).tolist(),
            }
            for name in ("train", "validation", "test"):
                if cached_parts[name] != expected_parts[name]:
                    raise RuntimeError(
                        f"Graph cache {cache_path} does not match frozen {name} assignments. "
                        "Rebuild it with --force-refeaturise."
                    )
        return bundle

    task_names = deep_tox.get_primary_classification_tasks(frame.columns, require_all=True)
    if tuple(task_names) != PRIMARY_TASKS:
        raise RuntimeError(f"Unexpected task order: {task_names}")
    frame = deep_tox.coerce_binary_classification_targets(frame, task_names)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = deep_tox.molecular_graphs_representation(
        frame,
        task_names,
        split_column="benchmark_split",
        make_split_figures=False,
        artifact_dir=str(cache_path.parent),
    )
    joblib.dump(bundle, cache_path)
    return bundle


def main() -> int:
    """Execute one requested stage in O(n) model-workflow time and space."""
    args = parse_args()
    bundle = load_or_build_graphs(
        args.split_dir,
        args.graph_cache,
        force=args.force_refeaturise or args.stage == "featurise",
        require_frozen_cache=args.stage != "evaluate",
    )
    if args.stage == "featurise":
        print(f"Saved graph bundle to {args.graph_cache}")
        return 0

    train_list, val_list, test_list, train_df, val_df, test_df = bundle
    task_names = deep_tox.get_primary_classification_tasks(train_df.columns, require_all=True)

    if args.stage == "baselines":
        results = deep_tox.run_multitask_baselines(
            (train_df, val_df, test_df),
            task_names,
            out_dir=str(ROOT / "results" / "baselines"),
        )
        deep_tox.plot_baseline_comparison(
            results,
            out_dir=str(ROOT / "results" / "baselines"),
        )
        return 0

    if args.stage == "evaluate":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for evaluation")
        accelerator = "gpu" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        model = deep_tox.GAT_class.load_from_checkpoint(
            str(args.checkpoint.resolve()),
            map_location=args.device,
        )
        model.output_dir_cls = str(ROOT / "results")
        batch_size = int(deep_tox.MODEL_DEFAULTS["batch_size"])
        val_loader = GeoDataLoader(val_list, batch_size=batch_size, shuffle=False)
        test_loader = GeoDataLoader(test_list, batch_size=batch_size, shuffle=False)
        trainer = L.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            inference_mode=True,
        )
        trainer.validate(model=model, dataloaders=val_loader)
        trainer.test(model=model, dataloaders=test_loader)
        return 0

    deep_tox.geometry_gnn_classification(
        train_list,
        val_list,
        test_list,
        task_names,
        val_df,
        test_df,
        out_dir_cls=str(ROOT / "results"),
        run_external_audit=False,
    )
    if args.stage == "all":
        results = deep_tox.run_multitask_baselines(
            (train_df, val_df, test_df),
            task_names,
            out_dir=str(ROOT / "results" / "baselines"),
        )
        deep_tox.plot_baseline_comparison(
            results,
            out_dir=str(ROOT / "results" / "baselines"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
