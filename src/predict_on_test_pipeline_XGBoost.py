import pandas as pd
import joblib
import os

MODEL_PATH = r"D:\projects for my CV\House price predection ( Regression )\XGBoost\model_xgb_pipeline.pkl"
TEST_DATA_PATH = r".\Data\test.csv"
OUTPUT_PATH = r"D:\projects for my CV\House price predection ( Regression )\XGBoost\\submission_pipeline_XGBoost.csv"


def load_model(path: str):
    print(f"[INFO] Loading model from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    print("[INFO] Model loaded.")
    return model


def load_test_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading test data from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Test shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def predict_and_save(model, df_test: pd.DataFrame, out_path: str):
    id_col = None
    for cand in ["Id", "id", "ID"]:
        if cand in df_test.columns:
            id_col = cand
            break

    if id_col is None:
        raise ValueError("No Id column found in test data.")

    test_ids = df_test[id_col].copy()

    X_test = df_test.drop(columns=[id_col])

    print("[INFO] Making predictions on test data (using full pipeline)...")
    preds = model.predict(X_test)

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": preds
    })

    print(f"[INFO] Saving submission file to: {out_path}")
    submission.to_csv(out_path, index=False)
    print("[INFO] Submission saved.")


def main():
    model = load_model(MODEL_PATH)
    df_test = load_test_data(TEST_DATA_PATH)
    predict_and_save(model, df_test, OUTPUT_PATH)
    print("\n[INFO] All done (prediction with pipeline).")


if __name__ == "__main__":
    main()

