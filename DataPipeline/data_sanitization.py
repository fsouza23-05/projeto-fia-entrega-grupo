import pandas as pd
import yaml
import joblib
from pathlib import Path


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def select_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    return df[features].copy()


def fill_occupation_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("UNKNOWN")
    return df


def fill_bureau_cols(df: pd.DataFrame, bureau_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[bureau_cols] = df[bureau_cols].fillna(0)
    return df


def create_missing_flags(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    # Flags criadas ANTES da imputação para registrar ausência original
    df = df.copy()
    for col in cols:
        df[f"{col}_MISSING"] = df[col].isnull().astype(int)
    return df


def impute_medians(df: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    medianas = {}
    for col in cols:
        median_val = float(df[col].median())
        medianas[col] = median_val
        df[col] = df[col].fillna(median_val)
    return df, medianas


def run_sanitization(
    config_path: str = "DataPipeline/config.yaml",
    use_minio: bool = False,
) -> pd.DataFrame:
    cfg = _load_config(config_path)

    if use_minio:
        from utils.storage import load_csv, save_csv, save_joblib
        mc = cfg["minio"]
        df = load_csv(mc["buckets"]["raw_data"], mc["objects"]["raw_data"])
    else:
        df = load_raw_data(cfg["paths"]["raw_data"])

    df = select_features(df, cfg["abt_features"])
    df = fill_occupation_type(df)
    df = fill_bureau_cols(df, cfg["bureau_cols"])
    df = create_missing_flags(df, cfg["missing_flag_cols"])
    df, medianas = impute_medians(df, cfg["numeric_impute_cols"])

    expected_cols = len(cfg["abt_features"]) + len(cfg["missing_flag_cols"])
    assert df.isnull().sum().sum() == 0, "ABT ainda contém valores nulos após sanitização."
    assert df.shape[1] == expected_cols, f"Colunas esperadas: {expected_cols}, obtidas: {df.shape[1]}"

    if use_minio:
        save_csv(df, mc["buckets"]["processed_data"], mc["objects"]["clean_data"])
        save_joblib(medianas, mc["buckets"]["artifacts"], mc["objects"]["medianas"])
        print(f"clean_data.csv → MinIO ({mc['buckets']['processed_data']}/{mc['objects']['clean_data']}): {df.shape}")
        print(f"medianas.joblib → MinIO ({mc['buckets']['artifacts']}/{mc['objects']['medianas']})")
    else:
        Path(cfg["paths"]["clean_data"]).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg["paths"]["clean_data"], index=False)
        Path(cfg["paths"]["medianas"]).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(medianas, cfg["paths"]["medianas"])
        print(f"clean_data.csv salvo: {df.shape}")
        print(f"Valores nulos: {df.isnull().sum().sum()}")
        print(f"medianas.joblib salvo em: {cfg['paths']['medianas']}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--minio", action="store_true", help="Lê/escreve via MinIO em vez do sistema de arquivos local")
    args = parser.parse_args()
    run_sanitization(use_minio=args.minio)
