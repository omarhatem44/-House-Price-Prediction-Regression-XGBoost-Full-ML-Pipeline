import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from xgboost import XGBRegressor  

DATA_PATH = r".\Data\train.csv"
TARGET_COL = "SalePrice"
MODEL_PATH = r"D:\projects for my CV\House price predection ( Regression )\XGBoost\model_xgb_pipeline.pkl"


def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Raw shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def prepare_features_and_target(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data.")

    drop_cols = [TARGET_COL]
    if "Id" in df.columns:
        drop_cols.append("Id")

    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"[INFO] Numeric features: {len(numeric_features)}")
    print(f"[INFO] Categorical features: {len(categorical_features)}")

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])   # Numeric: NaN → median
     # Categorical: NaN → mode + OneHot
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_and_evaluate_model(X, y, preprocessor):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[INFO] Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

    # XGBoost Regressor
    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("[INFO] Training XGBoost pipeline...")
    clf.fit(X_train, y_train)

    print("[INFO] Predicting on validation set...")
    y_pred = clf.predict(X_valid)

    mse = mean_squared_error(y_valid, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_valid, y_pred)

    print("\n===== XGBoost Evaluation Results =====")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.4f}")

    return clf


def save_model(model, path: str):
    print(f"[INFO] Saving trained XGBoost pipeline model to: {path}")
    joblib.dump(model, path)
    print("[INFO] Model saved.")


def main():
    df = load_data(DATA_PATH)
    X, y, numeric_features, categorical_features = prepare_features_and_target(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = train_and_evaluate_model(X, y, preprocessor)
    save_model(model, MODEL_PATH)
    print("\n[INFO] All done (XGBoost training with full pipeline).")


if __name__ == "__main__":
    main()

