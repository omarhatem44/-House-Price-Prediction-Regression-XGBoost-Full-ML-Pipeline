import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


MODEL_PATH = r".\XGBoost\model_xgb_pipeline.pkl"


def load_model(path: str):
    print(f"[INFO] Loading model from: {path}")
    model = joblib.load(path)
    print("[INFO] Model loaded.")
    return model


def get_feature_names_from_pipeline(model: Pipeline):
    preprocessor = model.named_steps["preprocessor"]
    transformers = preprocessor.transformers_

    feature_names = []

    for name, transformer, cols in transformers:
        if name == "num":
            feature_names.extend(cols)    # numeric columns 
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]     # categorical columns → OneHotEncoder
            ohe_feature_names = ohe.get_feature_names_out(cols)
            feature_names.extend(ohe_feature_names)

    return feature_names


def plot_feature_importance(model: Pipeline, feature_names):
    xgb: XGBRegressor = model.named_steps["model"]

    importances = xgb.feature_importances_

    indices = np.argsort(importances)[::-1]

    top_n = 20 
    top_indices = indices[:top_n]

    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[top_indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices][::-1])
    plt.xlabel("Feature Importance Score")
    plt.title("Top 20 Most Important Features (XGBoost)")
    plt.tight_layout()

    plt.savefig("xgb_feature_importance.png", dpi=300)
    plt.show()

    print("[INFO] Saved plot as xgb_feature_importance.png")


def main():
    model = load_model(MODEL_PATH)
    feature_names = get_feature_names_from_pipeline(model)
    plot_feature_importance(model, feature_names)


if __name__ == "__main__":
    main()

