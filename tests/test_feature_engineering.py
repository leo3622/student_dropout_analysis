import numpy as np
import pandas as pd

from scripts.model_pipeline import (
    OutlierClipper,
    add_derived_features,
    build_preprocessor,
    infer_feature_lists,
    normalize_categories,
    rename_columns,
)


def test_rename_and_derived_features():
    df = pd.DataFrame(
        {
            "Nacionality": [1, 2],
            "Curricular units 1st sem (grade)": [10, 14],
            "Curricular units 2nd sem (grade)": [12, 16],
            "Output": ["Dropout", "Graduate"],
        }
    )

    cleaned = add_derived_features(rename_columns(df))
    assert "Nationality" in cleaned.columns
    assert "performance_score" in cleaned.columns
    assert cleaned["performance_score"].iloc[0] == 11


def test_preprocessor_handles_missing_and_categoricals():
    df = pd.DataFrame(
        {
            "Gender": ["Male", "Female", None],
            "evening attendance": ["Day time", "Evening time", "Day time"],
            "performance_score": [10.0, None, 15.5],
            "Output": ["Dropout", "Graduate", "Enrolled"],
        }
    )
    cat_cols, num_cols = infer_feature_lists(df)
    preprocessor = build_preprocessor(cat_cols, num_cols)
    transformed = preprocessor.fit_transform(df.drop(columns=["Output"]))
    assert transformed.shape[0] == len(df)
    # Expect at least one encoded gender column and one numeric column
    assert transformed.shape[1] >= 3


def test_outlier_clipper_caps_extremes():
    data = np.array([[0], [1], [2], [100]])
    clipper = OutlierClipper(lower_quantile=0.25, upper_quantile=0.75)
    clipper.fit(data)
    transformed = clipper.transform(data)

    assert transformed.max() <= clipper.upper_bounds_[0]
    assert transformed.min() >= clipper.lower_bounds_[0]
