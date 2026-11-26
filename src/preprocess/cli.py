import argparse
import logging
from pathlib import Path
from typing import Any

from rich.console import Console

from src.common.cli import (
    CLISpec,
    build_cli_parser,
    print_cli_help,
    print_summary_table,
    run_cli_app,
    run_with_progress,
)
from src.common.core.errors import EXIT_SUCCESS, InvalidArgumentError

_DEFAULT_OUT = Path("src/preprocess")
_DEFAULT_ASSETS = Path("src/preprocess/assets")
_DEFAULT_COMBINED_REPORT = Path("src/data/README.md")
_CLI_SPEC = CLISpec(
    prog="python -m src.preprocess",
    description="AgriPak preprocessing, feature engineering, selection, and CV evaluation CLI.",
    default_input="data/raw",
    default_output=str(_DEFAULT_OUT),
)


def _configure_extra(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=[
            "run",
            "transform",
            "encode",
            "scale",
            "select",
            "evaluate",
            "plot",
            "report",
            "help",
        ],
    )
    parser.add_argument("--assets", type=Path, default=_DEFAULT_ASSETS)
    parser.add_argument("--scale-method", default="standard")
    parser.add_argument("--encode-method", default="onehot")
    parser.add_argument("--combined-report", type=Path, default=None)


def _resolve_combined_report(args: argparse.Namespace) -> Path | None:
    if args.combined_report is not None:
        return Path(args.combined_report)
    out_dir = Path(args.output)
    if out_dir == _DEFAULT_OUT or out_dir == Path("data/clean"):
        return _DEFAULT_COMBINED_REPORT
    return None


def _cmd_run(args: argparse.Namespace, console: Console) -> int:
    out_dir = Path(args.output)
    assets_dir = Path(args.assets)
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules", 0.0)
        from src.common.core.io import load_dataset
        from src.common.report.assembler import relative_asset_prefix
        from src.preprocess.modeling.workflow import run_preprocess_pipeline
        from src.preprocess.outputs.feature_charts import generate_preprocess_plots
        from src.preprocess.outputs.reporting import generate_preprocess_reports

        tick("Loading dataset", 10.0)
        df = load_dataset(Path(args.input))
        tick("Running preprocessing pipeline", 10.0)
        bundle = run_preprocess_pipeline(
            df, scale_method=args.scale_method, encode_method=args.encode_method
        )
        tick("Generating plots", 45.0)
        plots = generate_preprocess_plots(
            bundle, assets_dir, on_plot=lambda p: tick(f"Plot {p}", 6.25)
        )
        tick("Writing reports", 5.0)
        prefix = relative_asset_prefix(assets_dir, out_dir)
        generate_preprocess_reports(
            bundle, plots, out_dir, prefix, _resolve_combined_report(args)
        )
        state["rows"] = len(df)
        state["bundle"] = bundle

    run_with_progress(console, 100.0, "Preprocess Pipeline", _run)
    bundle = state["bundle"]
    print_summary_table(
        console,
        "Preprocess Pipeline",
        [
            ("Rows", str(state["rows"])),
            ("Selected", str(len(bundle.selection.selected_features))),
            ("Ablation Delta", f"{bundle.evaluation.ablation_delta:+.4f}"),
        ],
    )
    return EXIT_SUCCESS


def _cmd_transform(args: argparse.Namespace, console: Console) -> int:
    in_path = Path(args.input)
    out_file = Path(args.output) / "engineered.csv"
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset, save_dataframe
        from src.preprocess.transforms.transformers import engineer_features

        tick("Engineering features")
        df = load_dataset(in_path)
        res = engineer_features(df)
        tick("Saving CSV")
        save_dataframe(res.df, out_file, source_input=in_path)
        state["res"] = res

    run_with_progress(console, 3, "Transform Features", _run)
    res = state["res"]
    print_summary_table(
        console,
        "Transform",
        [("Engineered", ", ".join(res.engineered_cols)), ("Output", str(out_file))],
    )
    return EXIT_SUCCESS


def _cmd_encode(args: argparse.Namespace, console: Console) -> int:
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset
        from src.common.dataset.columns import infer_schema
        from src.preprocess.transforms.encoders import encode_categorical_columns

        tick("Encoding categoricals")
        df = load_dataset(Path(args.input))
        schema = infer_schema(df)
        state["res"] = encode_categorical_columns(
            df, list(schema.categorical), method=args.encode_method
        )

    run_with_progress(console, 2, "Encode Categoricals", _run)
    res = state["res"]
    print_summary_table(
        console,
        "Encode",
        [
            ("Method", res.method),
            ("Source", str(len(res.source_columns))),
            ("Encoded", str(len(res.encoded_columns))),
        ],
    )
    return EXIT_SUCCESS


def _cmd_scale(args: argparse.Namespace, console: Console) -> int:
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset
        from src.common.dataset.columns import infer_schema
        from src.preprocess.transforms.scalers import scale_numeric_columns

        tick("Scaling numerics")
        df = load_dataset(Path(args.input))
        schema = infer_schema(df)
        state["res"] = scale_numeric_columns(
            df, list(schema.numeric_columns), method=args.scale_method
        )

    run_with_progress(console, 2, "Scale Numerics", _run)
    res = state["res"]
    print_summary_table(
        console,
        "Scale",
        [
            ("Method", res.method),
            ("Scaled", str(len(res.scaled_columns))),
            ("Zero-Var", str(len(res.zero_variance_columns))),
        ],
    )
    return EXIT_SUCCESS


def _cmd_select(args: argparse.Namespace, console: Console) -> int:
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset
        from src.common.dataset.columns import infer_schema
        from src.preprocess.transforms.selection import select_features
        from src.preprocess.transforms.transformers import engineer_features

        tick("Selecting features")
        df = load_dataset(Path(args.input))
        eng = engineer_features(df)
        state["sel"] = select_features(eng.df, infer_schema(df), eng.engineered_cols)

    run_with_progress(console, 2, "Select Features", _run)
    sel = state["sel"]
    print_summary_table(
        console,
        "Selection",
        [
            ("Selected", str(len(sel.selected_features))),
            ("Dropped", str(len(sel.dropped_features))),
        ],
    )
    return EXIT_SUCCESS


def _cmd_evaluate(args: argparse.Namespace, console: Console) -> int:
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset
        from src.common.dataset.columns import infer_schema
        from src.preprocess.modeling.evaluation import evaluate_feature_sets
        from src.preprocess.transforms.selection import select_features
        from src.preprocess.transforms.transformers import engineer_features

        tick("Running CV evaluation")
        df = load_dataset(Path(args.input))
        schema = infer_schema(df)
        eng = engineer_features(df, schema)
        sel = select_features(eng.df, schema, eng.engineered_cols)
        state["ev"] = evaluate_feature_sets(eng.df, schema, sel)

    run_with_progress(console, 2, "CV Evaluation", _run)
    ev = state["ev"]
    print_summary_table(
        console,
        "CV Evaluation",
        [
            ("Ablation Delta", f"{ev.ablation_delta:+.4f}"),
            ("Noise Lift", f"{ev.noise_lift:+.4f}"),
        ],
    )
    return EXIT_SUCCESS


def _cmd_plot(args: argparse.Namespace, console: Console) -> int:
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset
        from src.preprocess.modeling.workflow import run_preprocess_pipeline
        from src.preprocess.outputs.feature_charts import generate_preprocess_plots

        tick("Loading dataset")
        df = load_dataset(Path(args.input))
        tick("Running pipeline")
        bundle = run_preprocess_pipeline(df, run_cv=True)
        tick("Generating plots")
        state["plots"] = generate_preprocess_plots(bundle, Path(args.assets))

    run_with_progress(console, 4, "Preprocess Plots", _run)
    plots = state["plots"]
    created = sum(1 for p in plots if p.path is not None)
    print_summary_table(
        console,
        "Plots",
        [("Generated", str(created)), ("Skipped", str(len(plots) - created))],
    )
    return EXIT_SUCCESS


def _cmd_report(args: argparse.Namespace, console: Console) -> int:
    out_dir = Path(args.output)
    assets_dir = Path(args.assets)
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules", 0.0)
        from src.common.core.io import load_dataset
        from src.common.report.assembler import relative_asset_prefix
        from src.preprocess.modeling.workflow import run_preprocess_pipeline
        from src.preprocess.outputs.feature_charts import generate_preprocess_plots
        from src.preprocess.outputs.reporting import generate_preprocess_reports

        tick("Loading dataset", 10.0)
        df = load_dataset(Path(args.input))
        tick("Running pipeline", 10.0)
        bundle = run_preprocess_pipeline(df, run_cv=True)
        tick("Generating plots", 45.0)
        plots = generate_preprocess_plots(
            bundle, assets_dir, on_plot=lambda p: tick(f"Plot {p}", 6.25)
        )
        tick("Writing reports", 5.0)
        prefix = relative_asset_prefix(assets_dir, out_dir)
        state["paths"] = generate_preprocess_reports(
            bundle, plots, out_dir, prefix, _resolve_combined_report(args)
        )

    run_with_progress(console, 100.0, "Preprocess Report", _run)
    readme_p, frag_p = state["paths"]
    print_summary_table(
        console, "Reports", [("README", str(readme_p)), ("Fragment", str(frag_p))]
    )
    return EXIT_SUCCESS


def _execute(
    args: argparse.Namespace, console: Console, _logger: logging.Logger
) -> int:
    handlers = {
        "run": _cmd_run,
        "transform": _cmd_transform,
        "encode": _cmd_encode,
        "scale": _cmd_scale,
        "select": _cmd_select,
        "evaluate": _cmd_evaluate,
        "plot": _cmd_plot,
        "report": _cmd_report,
    }
    if args.command == "help":
        print_cli_help(build_cli_parser(_CLI_SPEC, _configure_extra), console)
        return EXIT_SUCCESS
    if args.command not in handlers:
        raise InvalidArgumentError(f"Unknown command: {args.command}")
    return handlers[args.command](args, console)


def main(argv: list[str] | None = None) -> int:
    return run_cli_app(_CLI_SPEC, argv, _execute, configure_extra=_configure_extra)
