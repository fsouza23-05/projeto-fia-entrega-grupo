import pandas as pd
import yaml
from pathlib import Path


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_clean_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # LightGBM rejeita nomes com caracteres especiais JSON (ex: ":" em
    # "ORGANIZATION_TYPE_Industry: type 1" gerado pelo get_dummies).
    import re
    df.columns = [re.sub(r'[:\[\]{}".,\s]+', '_', c).strip('_') for c in df.columns]
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    # Reproduz exatamente pd.get_dummies(X, drop_first=True) do notebook original.
    encoded = pd.get_dummies(df, drop_first=True)
    return sanitize_column_names(encoded)


def run_transform(
    config_path: str = "DataPipeline/config.yaml",
    use_minio: bool = False,
) -> pd.DataFrame:
    cfg = _load_config(config_path)

    if use_minio:
        from utils.storage import load_csv, save_csv
        mc = cfg["minio"]
        df = load_csv(mc["buckets"]["processed_data"], mc["objects"]["clean_data"])
    else:
        df = load_clean_data(cfg["paths"]["clean_data"])

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    X_encoded = encode_categorical(X)

    assert X_encoded.shape[1] > 0, "Encoding não gerou nenhuma coluna. Verifique clean_data.csv."
    assert X_encoded.shape[0] == df.shape[0], "Encoding alterou o número de linhas inesperadamente."

    # Reconstitui o ABT com TARGET na primeira coluna
    abt = X_encoded.copy()
    abt.insert(0, "TARGET", y.values)

    if use_minio:
        save_csv(abt, mc["buckets"]["processed_data"], mc["objects"]["abt"])
        print(f"abt.csv → MinIO ({mc['buckets']['processed_data']}/{mc['objects']['abt']}): {abt.shape}")
    else:
        Path(cfg["paths"]["abt"]).parent.mkdir(parents=True, exist_ok=True)
        abt.to_csv(cfg["paths"]["abt"], index=False)
        print(f"abt.csv salvo: {abt.shape}  (119 features + TARGET)")

    return abt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--minio", action="store_true", help="Lê/escreve via MinIO em vez do sistema de arquivos local")
    args = parser.parse_args()
    run_transform(use_minio=args.minio)
