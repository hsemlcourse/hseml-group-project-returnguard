"""
test.py — тесты пайплайна предобработки и моделирования
Запуск: pytest tests/test.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.modeling import (
    classification_metrics,
    find_association_rules,
    regression_metrics,
    train_gbm,
    train_logistic_regression,
    train_polynomial_regression,
    train_random_forest,
)
from src.preprocessing import (
    FEATURES,
    RAW_DATA_PATH,
    REGION_CODE,
    TARGET_CLF,
    TARGET_REG,
    binarize_features,
    create_binary_target,
    filter_region,
    get_Xy,
    load_raw_data,
    scale_features,
    split_data,
)


@pytest.fixture(scope="module")
def raw_data():
    return load_raw_data(RAW_DATA_PATH)


@pytest.fixture(scope="module")
def region_data(raw_data):
    return filter_region(raw_data, REGION_CODE)


@pytest.fixture(scope="module")
def processed_data(region_data):
    df, q3 = create_binary_target(region_data)
    return df, q3


@pytest.fixture(scope="module")
def Xy_reg(processed_data):
    df, _ = processed_data
    return get_Xy(df, FEATURES, TARGET_REG)


@pytest.fixture(scope="module")
def Xy_clf(processed_data):
    df, _ = processed_data
    return get_Xy(df, FEATURES, TARGET_CLF)


@pytest.fixture(scope="module")
def split_reg(Xy_reg):
    X, y = Xy_reg
    return split_data(X, y)


@pytest.fixture(scope="module")
def split_clf(Xy_clf):
    X, y = Xy_clf
    return split_data(X, y, stratify=y)


class TestPreprocessing:

    def test_load_data_shape(self, raw_data):
        assert raw_data.shape[0] > 0
        assert raw_data.shape[1] >= 42

    def test_filter_region(self, region_data, raw_data):
        assert len(region_data) < len(raw_data)
        assert (region_data["ter"] == REGION_CODE).all()

    def test_filter_region_not_empty(self, region_data):
        assert len(region_data) > 0

    def test_binary_target_created(self, processed_data):
        df, q3 = processed_data
        assert TARGET_CLF in df.columns
        assert set(df[TARGET_CLF].unique()).issubset({0, 1})

    def test_binary_target_q3_ratio(self, processed_data):
        df, q3 = processed_data
        ratio = df[TARGET_CLF].mean()
        # Должно быть около 25% (3-й квартиль)
        assert 0.20 <= ratio <= 0.30, f"Ожидалось ~25%, получено {ratio:.1%}"

    def test_binary_target_q3_value(self, processed_data):
        df, q3 = processed_data
        assert q3 > 0
        assert isinstance(q3, float)

    def test_get_Xy_regression(self, Xy_reg):
        X, y = Xy_reg
        assert set(FEATURES).issubset(set(X.columns))
        assert len(X) == len(y)
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_get_Xy_classification(self, Xy_clf):
        X, y = Xy_clf
        assert set(y.unique()).issubset({0, 1})

    def test_split_data_sizes(self, split_reg, Xy_reg):
        X_tr, X_te, y_tr, y_te = split_reg
        X, y = Xy_reg
        assert len(X_tr) + len(X_te) == len(X)
        assert abs(len(X_tr) / len(X) - 0.7) < 0.01

    def test_scale_features(self, split_reg):
        X_tr, X_te, y_tr, y_te = split_reg
        X_tr_s, X_te_s, scaler = scale_features(X_tr.values, X_te.values)
        # Среднее на train ≈ 0
        assert abs(X_tr_s.mean()) < 1e-8
        # Std на train ≈ 1
        assert abs(X_tr_s.std() - 1.0) < 0.1

    def test_binarize_features(self, Xy_reg):
        X, y = Xy_reg
        df_bin = binarize_features(X, FEATURES)
        assert df_bin.shape == (len(X), len(FEATURES))
        assert set(df_bin.values.flatten()).issubset({0, 1})
        # Для каждой колонки должно быть примерно 50/50
        for col in df_bin.columns:
            ratio = df_bin[col].mean()
            assert 0.3 <= ratio <= 0.7, f"{col}: ratio={ratio:.2f}"


class TestMetrics:

    def test_regression_metrics_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = regression_metrics(y, y, y, y)
        assert "train_r2" in metrics
        assert "test_r2" in metrics
        assert "train_rmse" in metrics
        assert "test_rmse" in metrics
        assert "train_mae" in metrics
        assert "test_mae" in metrics

    def test_regression_metrics_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = regression_metrics(y, y, y, y)
        assert metrics["train_r2"] == pytest.approx(1.0)
        assert metrics["train_rmse"] == pytest.approx(0.0)

    def test_classification_metrics_keys(self):
        y = np.array([0, 1, 0, 1])
        prob = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = classification_metrics(y, y, prob, y, y, prob)
        assert "train_accuracy" in metrics
        assert "test_roc_auc" in metrics
        assert "train_cm" in metrics
        assert "test_report" in metrics

    def test_classification_metrics_perfect(self):
        y = np.array([0, 1, 0, 1])
        prob = np.array([0.0, 1.0, 0.0, 1.0])
        metrics = classification_metrics(y, y, prob, y, y, prob)
        assert metrics["train_accuracy"] == pytest.approx(1.0)
        assert metrics["test_roc_auc"] == pytest.approx(1.0)


class TestModels:

    def test_polynomial_regression(self, split_reg):
        X_tr, X_te, y_tr, y_te = split_reg
        results = train_polynomial_regression(
            X_tr.values, X_te.values, y_tr.values, y_te.values, degrees=range(1, 3)
        )
        assert len(results) == 2
        assert "train_r2" in results.columns
        assert "test_r2" in results.columns
        # Хотя бы степень 1 должна давать положительный R² на тесте
        assert results.loc[results["degree"] == 1, "test_r2"].values[0] > 0

    def test_logistic_regression(self, split_clf):
        X_tr, X_te, y_tr, y_te = split_clf
        result = train_logistic_regression(X_tr, X_te, y_tr.values, y_te.values)
        assert "model" in result
        assert "selected_features" in result
        assert len(result["selected_features"]) == 3
        assert result["test_roc_auc"] > 0.5

    def test_random_forest(self, split_clf):
        X_tr, X_te, y_tr, y_te = split_clf
        result = train_random_forest(X_tr, X_te, y_tr.values, y_te.values)
        assert "model" in result
        assert result["test_accuracy"] > 0.7
        assert result["test_roc_auc"] > 0.7
        assert "feature_importance_mdi" in result

    def test_gbm(self, split_clf):
        X_tr, X_te, y_tr, y_te = split_clf
        result = train_gbm(X_tr, X_te, y_tr.values, y_te.values)
        assert "model" in result
        assert result["test_accuracy"] > 0.7
        assert result["test_roc_auc"] > 0.7

    def test_association_rules_fpgrowth(self, Xy_reg):
        X, y = Xy_reg
        df_bin = binarize_features(X, FEATURES).astype(bool)
        rules = find_association_rules(df_bin, method="fpgrowth", min_support=0.1)
        assert len(rules) > 0
        assert "lift" in rules.columns
        assert "confidence" in rules.columns
        assert "support" in rules.columns
        # Правила отсортированы по lift убыванию
        assert rules["lift"].is_monotonic_decreasing

    def test_association_rules_apriori(self, Xy_reg):
        X, y = Xy_reg
        df_bin = binarize_features(X, FEATURES).astype(bool)
        rules = find_association_rules(df_bin, method="apriori", min_support=0.1)
        assert len(rules) > 0
