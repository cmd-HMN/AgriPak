from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocess.transforms.encoders import build_categorical_encoder
from src.preprocess.transforms.scalers import build_numeric_scaler


def build_column_transformer(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_method: str = "standard",
    encode_method: str = "onehot",
) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_cols:
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", build_numeric_scaler(scale_method)),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="Unidentified"),
                ),
                ("encoder", build_categorical_encoder(encode_method)),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_logistic_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_method: str = "standard",
    encode_method: str = "onehot",
    max_iter: int = 1000,
) -> Pipeline:
    preprocessor = build_column_transformer(
        numeric_cols, categorical_cols, scale_method, encode_method
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=max_iter, random_state=42)),
        ]
    )


def build_hgb_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    encode_method: str = "ordinal",
) -> Pipeline:
    preprocessor = build_column_transformer(
        numeric_cols, categorical_cols, "standard", encode_method
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", HistGradientBoostingClassifier(random_state=42, max_iter=100)),
        ]
    )
