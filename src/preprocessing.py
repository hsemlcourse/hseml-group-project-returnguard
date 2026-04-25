import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = "data/raw/Доходы домохозяйств_2023.xlsx"
REGION_CODE = 71
RANDOM_STATE = 42

FEATURES = ["chlico", "chdet", "rasress", "rasq", "potras"]
TARGET_REG = "doxodn"
TARGET_CLF = "doxodn_high"


def load_raw_data(path=RAW_DATA_PATH):
    xl = pd.ExcelFile(path)
    return pd.read_excel(path, sheet_name=xl.sheet_names[0])


def filter_region(data, region_code=REGION_CODE):
    return data[data["ter"] == region_code].copy()


def create_binary_target(data, target=TARGET_REG, quantile=0.75):
    q_val = float(np.percentile(data[target].dropna(), quantile * 100))
    data = data.copy()
    data[TARGET_CLF] = (data[target] > q_val).astype(int)
    return data, q_val


def get_Xy(data, features=FEATURES, target=TARGET_REG):
    df = data[features + [target]].dropna()
    return df[features], df[target]


def split_data(X, y, train_size=0.7, random_state=RANDOM_STATE, stratify=None):
    return train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=stratify)


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def binarize_features(data, features):
    df_bin = pd.DataFrame(index=data.index)
    for col in features:
        df_bin[f"{col}_high"] = (data[col] > data[col].median()).astype(int)
    return df_bin


def load_processed_data(path=RAW_DATA_PATH):
    raw = load_raw_data(path)
    df = filter_region(raw)
    df, q3 = create_binary_target(df)

    X, y_reg = get_Xy(df, FEATURES, TARGET_REG)
    _, y_clf = get_Xy(df, FEATURES, TARGET_CLF)

    X_tr, X_te, y_tr_reg, y_te_reg = split_data(X, y_reg)
    _, _, y_tr_clf, y_te_clf = split_data(X, y_clf, stratify=y_clf)

    return {
        "df": df, "q3": q3,
        "X_train": X_tr, "X_test": X_te,
        "y_train_reg": y_tr_reg, "y_test_reg": y_te_reg,
        "y_train_clf": y_tr_clf, "y_test_clf": y_te_clf,
        "features": FEATURES,
    }
