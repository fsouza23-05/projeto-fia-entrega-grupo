import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from lightgbm import LGBMClassifier


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    import re
    df.columns = [re.sub(r'[:\[\]{}".,\s]+', '_', c).strip('_') for c in df.columns]
    return df


def load_abt(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return _sanitize_columns(df)


def split_features_target(
    df: pd.DataFrame, target_col: str = "TARGET"
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    # negativos / positivos — mesma lógica do XGBoost (~11.387)
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    return negatives / positives


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: dict
) -> LGBMClassifier:
    scale_pos_weight = compute_scale_pos_weight(y_train)
    m = cfg["model"]
    model = LGBMClassifier(
        n_estimators=m["n_estimators"],
        max_depth=m["max_depth"],
        num_leaves=m["num_leaves"],
        learning_rate=m["learning_rate"],
        scale_pos_weight=scale_pos_weight,
        random_state=m["random_state"],
        metric=m["metric"],
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall":    float(recall_score(y_test, y_pred)),
        "f1":        float(f1_score(y_test, y_pred)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
    }


def save_artifacts(
    model: LGBMClassifier, feature_names: list, cfg: dict, use_minio: bool = False
) -> None:
    if use_minio:
        from utils.storage import save_joblib
        mc = cfg["minio"]
        bucket = mc["bucket"]
        version = mc["version"]
        save_joblib(model, bucket, f"{version}/{mc['objects']['model']}")
        save_joblib(feature_names, bucket, f"{version}/{mc['objects']['features']}")
        print(f"Modelo salvo → MinIO ({bucket}/{version}/{mc['objects']['model']})")
        print(f"Features salvas → MinIO ({bucket}/{version}/{mc['objects']['features']})  ({len(feature_names)} colunas)")
    else:
        artifacts = cfg["artifacts"]
        Path(artifacts["model"]).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifacts["model"])
        joblib.dump(feature_names, artifacts["features"])
        print(f"Modelo salvo:   {artifacts['model']}")
        print(f"Features salvas: {artifacts['features']}  ({len(feature_names)} colunas)")


def run_training(
    pipeline_config_path: str = "DataPipeline/config.yaml",
    model_config_path: str = "Model/config.yaml",
    use_minio: bool = False,
    use_db: bool = False,
) -> dict:
    pipeline_cfg = _load_config(pipeline_config_path)
    model_cfg = _load_config(model_config_path)

    if use_minio:
        from utils.storage import load_csv
        mc = pipeline_cfg["minio"]
        df = load_csv(mc["buckets"]["processed_data"], mc["objects"]["abt"])
    else:
        df = load_abt(pipeline_cfg["paths"]["abt"])

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=model_cfg["split"]["test_size"],
        random_state=model_cfg["split"]["random_state"],
    )

    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

    model = train_model(X_train, y_train, model_cfg)
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, X_train.columns.tolist(), model_cfg, use_minio=use_minio)

    print("\nMétricas — LightGBM Balanced:")
    for k, v in metrics.items():
        print(f"  {k:12}: {v:.4f}")

    if use_db:
        from utils.db import register_model
        mc = model_cfg.get("minio", {})
        version = mc.get("version", "v2")
        register_model(
            version=version,
            metrics=metrics,
            description="LightGBM Balanced",
            artifact_bucket=mc.get("bucket", "model-artifacts"),
            artifact_path=f"{version}/{mc.get('objects', {}).get('model', 'lgbm_balanced_model.joblib')}",
        )
        print(f"Modelo registrado no PostgreSQL: versão {version}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--minio", action="store_true", help="Lê/escreve via MinIO em vez do sistema de arquivos local")
    parser.add_argument("--db", action="store_true", help="Registra o modelo no PostgreSQL após o treino")
    args = parser.parse_args()
    run_training(use_minio=args.minio, use_db=args.db)
