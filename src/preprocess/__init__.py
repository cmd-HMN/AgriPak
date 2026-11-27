from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.common.core.errors import TransformError
    from src.preprocess.cli import main
    from src.preprocess.modeling.builders import (
        build_column_transformer,
        build_hgb_pipeline,
        build_logistic_pipeline,
    )
    from src.preprocess.modeling.evaluation import (
        EvaluationResult,
        evaluate_feature_sets,
    )
    from src.preprocess.modeling.splitting import (
        SplitResult,
        save_train_test_split,
        split_train_test,
    )
    from src.preprocess.modeling.workflow import (
        PreprocessBundle,
        run_preprocess_pipeline,
    )
    from src.preprocess.outputs.feature_charts import generate_preprocess_plots
    from src.preprocess.outputs.reporting import (
        build_preprocess_fragment,
        build_preprocess_readme,
        generate_preprocess_reports,
        render_preprocess_sections,
    )
    from src.preprocess.transforms.encoders import (
        EncodingResult,
        build_categorical_encoder,
        encode_categorical_columns,
    )
    from src.preprocess.transforms.scalers import (
        ScalingResult,
        build_numeric_scaler,
        scale_numeric_columns,
    )
    from src.preprocess.transforms.selection import (
        FeatureSelectionResult,
        select_features,
    )
    from src.preprocess.transforms.transformers import (
        EngineeredFeaturesResult,
        add_binary_signal_sum,
        add_deficit_x_zero_signal,
        add_driver_deficit,
        add_driver_x_signal_sum,
        engineer_features,
        infer_risk_floor_threshold,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "TransformError": ("src.common.core.errors", "TransformError"),
    "EngineeredFeaturesResult": (
        "src.preprocess.transforms.transformers",
        "EngineeredFeaturesResult",
    ),
    "infer_risk_floor_threshold": (
        "src.preprocess.transforms.transformers",
        "infer_risk_floor_threshold",
    ),
    "add_binary_signal_sum": (
        "src.preprocess.transforms.transformers",
        "add_binary_signal_sum",
    ),
    "add_driver_deficit": (
        "src.preprocess.transforms.transformers",
        "add_driver_deficit",
    ),
    "add_deficit_x_zero_signal": (
        "src.preprocess.transforms.transformers",
        "add_deficit_x_zero_signal",
    ),
    "add_driver_x_signal_sum": (
        "src.preprocess.transforms.transformers",
        "add_driver_x_signal_sum",
    ),
    "engineer_features": (
        "src.preprocess.transforms.transformers",
        "engineer_features",
    ),
    "EncodingResult": ("src.preprocess.transforms.encoders", "EncodingResult"),
    "build_categorical_encoder": (
        "src.preprocess.transforms.encoders",
        "build_categorical_encoder",
    ),
    "encode_categorical_columns": (
        "src.preprocess.transforms.encoders",
        "encode_categorical_columns",
    ),
    "ScalingResult": ("src.preprocess.transforms.scalers", "ScalingResult"),
    "build_numeric_scaler": (
        "src.preprocess.transforms.scalers",
        "build_numeric_scaler",
    ),
    "scale_numeric_columns": (
        "src.preprocess.transforms.scalers",
        "scale_numeric_columns",
    ),
    "FeatureSelectionResult": (
        "src.preprocess.transforms.selection",
        "FeatureSelectionResult",
    ),
    "select_features": ("src.preprocess.transforms.selection", "select_features"),
    "build_column_transformer": (
        "src.preprocess.modeling.builders",
        "build_column_transformer",
    ),
    "build_logistic_pipeline": (
        "src.preprocess.modeling.builders",
        "build_logistic_pipeline",
    ),
    "build_hgb_pipeline": ("src.preprocess.modeling.builders", "build_hgb_pipeline"),
    "EvaluationResult": ("src.preprocess.modeling.evaluation", "EvaluationResult"),
    "evaluate_feature_sets": (
        "src.preprocess.modeling.evaluation",
        "evaluate_feature_sets",
    ),
    "SplitResult": ("src.preprocess.modeling.splitting", "SplitResult"),
    "save_train_test_split": (
        "src.preprocess.modeling.splitting",
        "save_train_test_split",
    ),
    "split_train_test": ("src.preprocess.modeling.splitting", "split_train_test"),
    "PreprocessBundle": ("src.preprocess.modeling.workflow", "PreprocessBundle"),
    "run_preprocess_pipeline": (
        "src.preprocess.modeling.workflow",
        "run_preprocess_pipeline",
    ),
    "generate_preprocess_plots": (
        "src.preprocess.outputs.feature_charts",
        "generate_preprocess_plots",
    ),
    "render_preprocess_sections": (
        "src.preprocess.outputs.reporting",
        "render_preprocess_sections",
    ),
    "build_preprocess_fragment": (
        "src.preprocess.outputs.reporting",
        "build_preprocess_fragment",
    ),
    "build_preprocess_readme": (
        "src.preprocess.outputs.reporting",
        "build_preprocess_readme",
    ),
    "generate_preprocess_reports": (
        "src.preprocess.outputs.reporting",
        "generate_preprocess_reports",
    ),
    "main": ("src.preprocess.cli", "main"),
}

__all__ = [
    "EncodingResult",
    "EngineeredFeaturesResult",
    "EvaluationResult",
    "FeatureSelectionResult",
    "PreprocessBundle",
    "ScalingResult",
    "SplitResult",
    "TransformError",
    "add_binary_signal_sum",
    "add_deficit_x_zero_signal",
    "add_driver_deficit",
    "add_driver_x_signal_sum",
    "build_categorical_encoder",
    "build_column_transformer",
    "build_hgb_pipeline",
    "build_logistic_pipeline",
    "build_numeric_scaler",
    "build_preprocess_fragment",
    "build_preprocess_readme",
    "encode_categorical_columns",
    "engineer_features",
    "evaluate_feature_sets",
    "generate_preprocess_plots",
    "generate_preprocess_reports",
    "infer_risk_floor_threshold",
    "main",
    "render_preprocess_sections",
    "run_preprocess_pipeline",
    "save_train_test_split",
    "scale_numeric_columns",
    "select_features",
    "split_train_test",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.preprocess' has no attribute {name!r}")
    mod_path, attr = _EXPORTS[name]
    return getattr(import_module(mod_path), attr)
