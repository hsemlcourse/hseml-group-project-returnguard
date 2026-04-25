from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.modeling import load_model

app = FastAPI(
    title="ReturnGuard — Income Prediction API",
    description="Предсказание среднедушевого дохода домохозяйств Тульской области",
    version="1.0.0",
)

Q3_THRESHOLD = 151_400.0


class HouseholdFeatures(BaseModel):
    chlico: float   # число лиц в домохозяйстве
    chdet: float    # число детей
    rasress: float  # располагаемые ресурсы
    rasq: float     # среднедушевые располагаемые ресурсы
    potras: float   # расход на конечное потребление


class PredictionResponse(BaseModel):
    income_predicted: float
    above_q3: int
    above_q3_probability: float
    q3_threshold: float


def _load_models():
    try:
        reg = load_model("xgboost_regressor")
        clf = load_model("gbm")
        return reg, clf
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Модели не найдены. Сначала запустите ноутбуки для обучения.",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseholdFeatures):
    reg_model, clf_model = _load_models()

    X = [[features.chlico, features.chdet, features.rasress, features.rasq, features.potras]]

    income = float(reg_model.predict(X)[0])
    above_q3 = int(clf_model.predict(X)[0])
    probability = float(clf_model.predict_proba(X)[0][1])

    return PredictionResponse(
        income_predicted=round(income, 2),
        above_q3=above_q3,
        above_q3_probability=round(probability, 4),
        q3_threshold=Q3_THRESHOLD,
    )
