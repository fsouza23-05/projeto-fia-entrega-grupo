from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from Model.predict import predict


REQUEST_COUNT = Counter(
    "creditguard_prediction_requests_total",
    "Total de requisicoes de predicao",
)

ERROR_COUNT = Counter(
    "creditguard_prediction_errors_total",
    "Total de erros na predicao",
)

PREDICTION_LATENCY = Histogram(
    "creditguard_prediction_latency_seconds",
    "Latencia da predicao em segundos",
)


class PredictionInput(BaseModel):
    amt_income_total: float = Field(..., ge=0)
    amt_credit: float = Field(..., ge=0)
    amt_annuity: float = Field(..., ge=0)
    down_payment: float = Field(0.0, ge=0)
    age_years: int = Field(..., ge=18, le=100)
    years_employed: int = Field(0, ge=0, le=60)
    cnt_children: int = Field(0, ge=0, le=20)
    ext_source_1: float = Field(0.5, ge=0.0, le=1.0)
    ext_source_2: float = Field(0.5, ge=0.0, le=1.0)
    ext_source_3: float = Field(0.5, ge=0.0, le=1.0)


app = FastAPI(
    title="CreditGuard AI - Prediction API",
    description="API de predicao de inadimplencia",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "creditguard-api",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict_endpoint(payload: PredictionInput) -> dict:
    REQUEST_COUNT.inc()

    with PREDICTION_LATENCY.time():
        try:
            feature_data = {
                "AMT_INCOME_TOTAL": payload.amt_income_total,
                "AMT_CREDIT": payload.amt_credit,
                "AMT_ANNUITY": payload.amt_annuity,
                "AMT_GOODS_PRICE": payload.amt_credit + payload.down_payment,
                "DAYS_BIRTH": -(payload.age_years * 365),
                "DAYS_EMPLOYED": -(payload.years_employed * 365),
                "CNT_CHILDREN": payload.cnt_children,
                "EXT_SOURCE_1": payload.ext_source_1,
                "EXT_SOURCE_2": payload.ext_source_2,
                "EXT_SOURCE_3": payload.ext_source_3,
            }

            output = predict(feature_data)
            probability = output["probability"]

            if probability < 0.30:
                risk_level = "BAIXO RISCO"
            elif probability < 0.70:
                risk_level = "MEDIO RISCO"
            else:
                risk_level = "ALTO RISCO"

            return {
                "prediction": output["prediction"],
                "probability": round(probability, 6),
                "risk_level": risk_level,
                "model_version": "v2",
            }

        except Exception as exc:
            ERROR_COUNT.inc()
            return {
                "error": "prediction_failed",
                "detail": str(exc),
            }
