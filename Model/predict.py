import io
import os
import sys
import joblib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = "Model/artifacts/lgbm_balanced_model.joblib"
FEATURES_PATH = "Model/artifacts/features.joblib"

_model = None
_features = None


def _log(msg):
    sys.stderr.write(f"[PREDICT] {msg}\n")
    sys.stderr.flush()


def _load_from_minio(object_name: str):
    try:
        _log(f"minio: tentando carregar {object_name}")
        from utils.storage import get_client
        version = os.environ.get("MODEL_VERSION", "v2")
        bucket = "model-artifacts"
        _log(f"minio: get_object {bucket}/{version}/{object_name}")
        response = get_client().get_object(bucket, f"{version}/{object_name}")
        _log(f"minio: response obtido, lendo bytes...")
        data = response.read()
        _log(f"minio: {len(data)} bytes lidos, carregando com joblib...")
        obj = joblib.load(io.BytesIO(data))
        _log(f"minio: carregado com sucesso")
        return obj
    except Exception as e:
        _log(f"minio: falhou ({type(e).__name__}: {e}) — usando fallback local")
        return None


def load_model():
    global _model
    if _model is None:
        _log("load_model: _model is None, carregando...")
        obj = _load_from_minio("lgbm_balanced_model.joblib")
        if obj is not None:
            _model = obj
            _log("load_model: carregado do MinIO")
        else:
            _log(f"load_model: joblib.load({MODEL_PATH})")
            _model = joblib.load(MODEL_PATH)
            _log(f"load_model: carregado localmente — tipo: {type(_model)}")
    else:
        _log("load_model: usando cache (_model ja carregado)")
    return _model


def load_features():
    global _features
    if _features is None:
        _log("load_features: _features is None, carregando...")
        obj = _load_from_minio("features.joblib")
        if obj is not None:
            _features = obj
            _log("load_features: carregado do MinIO")
        else:
            _log(f"load_features: joblib.load({FEATURES_PATH})")
            _features = joblib.load(FEATURES_PATH)
            _log(f"load_features: {len(_features)} features carregadas")
    else:
        _log(f"load_features: usando cache ({len(_features)} features)")
    return _features


def predict(data):
    _log("predict: inicio")

    _log("predict: chamando load_model()")
    model = load_model()
    _log(f"predict: modelo ok — {type(model)}")

    _log("predict: chamando load_features()")
    features = load_features()
    feature_list = list(features)
    _log(f"predict: features ok — {len(feature_list)} colunas")

    # Usar numpy array em vez de DataFrame para evitar conflito com PyArrow:
    # pandas 3.0+ ativa infer_string=True por padrão, convertendo índices de colunas
    # para ArrowStringArray. Extensões C++ do LightGBM acessam esses índices em
    # threads internas que conflitam com o WebSocket do Streamlit (também PyArrow)
    # → potencial SIGSEGV. Numpy bypassa completamente o PyArrow.
    _log(f"predict: criando array numpy (1, {len(feature_list)})")
    arr = np.zeros((1, len(feature_list)), dtype=np.float64)

    filled = 0
    for i, feat in enumerate(feature_list):
        if feat in data:
            arr[0, i] = data[feat]
            filled += 1
    _log(f"predict: {filled} valores preenchidos")

    _log("predict: chamando model.predict(arr)  ← PONTO CRITICO 1")
    prediction = model.predict(arr)[0]
    _log(f"predict: model.predict OK — resultado: {prediction}")

    _log("predict: chamando model.predict_proba(arr)  ← PONTO CRITICO 2")
    probability = model.predict_proba(arr)[0][1]
    _log(f"predict: model.predict_proba OK — probabilidade: {probability:.4f}")

    _log("predict: retornando resultado")
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
