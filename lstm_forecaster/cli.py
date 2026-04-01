"""Command-line interface for the LSTM power-management forecaster."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lstm-forecaster",
        description="LSTM 4-scenario power-management forecaster",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ #
    # preprocess
    # ------------------------------------------------------------------ #
    pp = sub.add_parser(
        "preprocess",
        help="Load raw telemetry, derive labels if missing, save preprocessed CSV.",
    )
    pp.add_argument("input", type=str, help="Path to raw CSV or JSONL file.")
    pp.add_argument("output", type=str, help="Destination path for preprocessed CSV.")

    # ------------------------------------------------------------------ #
    # train
    # ------------------------------------------------------------------ #
    tr = sub.add_parser("train", help="Train the LSTM forecaster.")
    tr.add_argument("--data-path", type=str, required=True, help="Path to training data.")
    tr.add_argument("--window-size", type=int, default=120)
    tr.add_argument("--horizon", type=int, default=60)
    tr.add_argument("--stride", type=int, default=1)
    tr.add_argument("--hidden-size", type=int, default=128)
    tr.add_argument("--num-layers", type=int, default=2)
    tr.add_argument("--dropout", type=float, default=0.3)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--max-epochs", type=int, default=50)
    tr.add_argument("--patience", type=int, default=5)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    tr.add_argument("--no-class-weights", action="store_true")
    tr.add_argument("--val-ratio", type=float, default=0.15)
    tr.add_argument("--test-ratio", type=float, default=0.15)

    # ------------------------------------------------------------------ #
    # evaluate
    # ------------------------------------------------------------------ #
    ev = sub.add_parser("evaluate", help="Evaluate a trained checkpoint.")
    ev.add_argument("checkpoint", type=str, help="Path to best_model.pt checkpoint.")
    ev.add_argument("--data-path", type=str, required=True)
    ev.add_argument("--output-dir", type=str, default="eval_output")
    ev.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
    )

    # ------------------------------------------------------------------ #
    # export
    # ------------------------------------------------------------------ #
    ex = sub.add_parser("export", help="Export a trained checkpoint.")
    ex.add_argument("checkpoint", type=str, help="Path to best_model.pt checkpoint.")
    ex.add_argument("--output-dir", type=str, default="export_output")
    ex.add_argument(
        "--format",
        type=str,
        default="torchscript",
        choices=["torchscript", "onnx"],
        help="Export format (default: torchscript).",
    )

    return parser


def main(argv=None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess":
        _cmd_preprocess(args)
    elif args.command == "train":
        _cmd_train(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "export":
        _cmd_export(args)
    else:
        parser.print_help()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _cmd_preprocess(args) -> None:
    """Load raw data, derive labels, and save preprocessed CSV."""
    from lstm_forecaster.dataset import LABEL_COL, label_from_telemetry, load_data

    df = load_data(args.input)
    if LABEL_COL not in df.columns:
        print("Deriving scenario labels from telemetry thresholds...")
        df[LABEL_COL] = df.apply(label_from_telemetry, axis=1)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}  ({len(df)} rows)")


def _cmd_train(args) -> None:
    """Run the training pipeline."""
    from lstm_forecaster.train import train

    config = {
        "data_path": args.data_path,
        "window_size": args.window_size,
        "horizon": args.horizon,
        "stride": args.stride,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "checkpoint_dir": args.checkpoint_dir,
        "use_class_weights": not args.no_class_weights,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }
    ckpt = train(config)
    print(f"\nTraining complete. Checkpoint: {ckpt}")


def _cmd_evaluate(args) -> None:
    """Evaluate a saved checkpoint."""
    from lstm_forecaster.evaluate import evaluate

    evaluate(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        split=args.split,
    )


def _cmd_export(args) -> None:
    """Export a saved checkpoint."""
    if args.format == "torchscript":
        from lstm_forecaster.export import export_torchscript

        export_torchscript(args.checkpoint, args.output_dir)
    else:
        from lstm_forecaster.export import export_onnx

        export_onnx(args.checkpoint, args.output_dir)
