from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import PolynomialFeatures

from src.preprocessing import RANDOM_STATE

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def regression_metrics(y_tr, y_tr_pred, y_te, y_te_pred):
    return {
        "train_r2":   r2_score(y_tr, y_tr_pred),
        "test_r2":    r2_score(y_te, y_te_pred),
        "train_rmse": mean_squared_error(y_tr, y_tr_pred) ** 0.5,
        "test_rmse":  mean_squared_error(y_te, y_te_pred) ** 0.5,
        "train_mae":  mean_absolute_error(y_tr, y_tr_pred),
        "test_mae":   mean_absolute_error(y_te, y_te_pred),
    }


def classification_metrics(y_tr, y_tr_pred, y_tr_prob, y_te, y_te_pred, y_te_prob):
    return {
        "train_accuracy": accuracy_score(y_tr, y_tr_pred),
        "test_accuracy":  accuracy_score(y_te, y_te_pred),
        "train_roc_auc":  roc_auc_score(y_tr, y_tr_prob),
        "test_roc_auc":   roc_auc_score(y_te, y_te_prob),
        "train_cm":       confusion_matrix(y_tr, y_tr_pred),
        "test_cm":        confusion_matrix(y_te, y_te_pred),
        "train_report":   classification_report(y_tr, y_tr_pred),
        "test_report":    classification_report(y_te, y_te_pred),
    }


def train_polynomial_regression(X_train, X_test, y_train, y_test, degrees=range(1, 9)):
    records = []
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        Xtr_p = poly.fit_transform(X_train)
        Xte_p = poly.transform(X_test)
        model = LinearRegression().fit(Xtr_p, y_train)
        y_tr_pred = model.predict(Xtr_p)
        y_te_pred = model.predict(Xte_p)
        records.append({
            "degree":    degree,
            "n_features": Xtr_p.shape[1],
            "train_r2":  r2_score(y_train, y_tr_pred),
            "test_r2":   r2_score(y_test, y_te_pred),
            "train_std": float(np.std(y_train - y_tr_pred)),
            "test_std":  float(np.std(y_test - y_te_pred)),
        })
    return pd.DataFrame(records)


def pca_regression(X_scaled, y, variance_threshold=0.70, train_size=0.6, random_state=RANDOM_STATE):
    from sklearn.model_selection import train_test_split

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.argmax(explained >= variance_threshold) + 1)

    X_gk = X_pca[:, :n_components]
    X_tr, X_te, y_tr, y_te = train_test_split(X_gk, y, train_size=train_size, random_state=random_state)

    reg = LinearRegression().fit(X_tr, y_tr)
    metrics = regression_metrics(y_tr, reg.predict(X_tr), y_te, reg.predict(X_te))
    metrics.update({"n_components": n_components, "explained_variance": explained, "pca": pca, "model": reg})
    return metrics


def train_logistic_regression(X_train, X_test, y_train, y_test, n_features_to_select=3):
    logit = LogisticRegression(max_iter=1000, solver="lbfgs")
    rfe = RFE(estimator=logit, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)

    selected = list(X_train.columns[rfe.support_])
    Xtr_rfe, Xte_rfe = X_train[selected], X_test[selected]

    model = LogisticRegression(max_iter=1000, solver="lbfgs").fit(Xtr_rfe, y_train)
    y_tr_pred, y_te_pred = model.predict(Xtr_rfe), model.predict(Xte_rfe)
    y_tr_prob = model.predict_proba(Xtr_rfe)[:, 1]
    y_te_prob = model.predict_proba(Xte_rfe)[:, 1]

    metrics = classification_metrics(y_train, y_tr_pred, y_tr_prob, y_test, y_te_pred, y_te_prob)
    metrics.update({
        "model": model,
        "selected_features": selected,
        "coef": pd.Series(model.coef_[0], index=selected),
    })
    return metrics


def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=2, random_state=RANDOM_STATE):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, y_train)

    y_tr_pred, y_te_pred = rf.predict(X_train), rf.predict(X_test)
    y_tr_prob = rf.predict_proba(X_train)[:, 1]
    y_te_prob = rf.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_train, y_tr_pred, y_tr_prob, y_test, y_te_pred, y_te_prob)
    metrics.update({
        "model": rf,
        "feature_importance_mdi": pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False),
    })
    return metrics


def train_gbm(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=2, learning_rate=0.1, random_state=RANDOM_STATE):
    gbm = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=random_state,
    )
    gbm.fit(X_train, y_train)

    y_tr_pred, y_te_pred = gbm.predict(X_train), gbm.predict(X_test)
    y_tr_prob = gbm.predict_proba(X_train)[:, 1]
    y_te_prob = gbm.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_train, y_tr_pred, y_tr_prob, y_test, y_te_pred, y_te_prob)
    metrics.update({
        "model": gbm,
        "feature_importance_mdi": pd.Series(gbm.feature_importances_, index=X_train.columns).sort_values(ascending=False),
    })
    return metrics


def train_xgboost_regressor(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=7, learning_rate=0.1, random_state=RANDOM_STATE):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=random_state,
    )
    model.fit(X_train, y_train)

    metrics = regression_metrics(y_train, model.predict(X_train), y_test, model.predict(X_test))
    metrics.update({
        "model": model,
        "feature_importance_gain":       pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False),
        "feature_importance_total_gain": pd.Series(model.get_booster().get_score(importance_type="total_gain")).sort_values(ascending=False),
    })
    return metrics


def find_association_rules(df_bin, method="fpgrowth", min_support=0.05, min_confidence=0.5):
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

    df_bool = df_bin.astype(bool)
    frequent = apriori(df_bool, min_support=min_support, use_colnames=True) \
        if method == "apriori" \
        else fpgrowth(df_bool, min_support=min_support, use_colnames=True)

    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def save_model(model, name, models_dir=MODELS_DIR):
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    path = models_dir / f"{name}.pkl"
    joblib.dump(model, path)
    return path


def load_model(name, models_dir=MODELS_DIR):
    return joblib.load(Path(models_dir) / f"{name}.pkl")
