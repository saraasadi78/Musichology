"""FastAPI service for mental-health score predictions."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.preprocessing import MENTAL_HEALTH_TARGETS

from app.prediction import predict_scores

app = FastAPI(
    title="Musichology Mental Health Predictor",
    description=(
        "Predicts self-reported mental-health scores from music habits. "
        "This is not a diagnostic tool."
    ),
)


class PredictionInput(BaseModel):
    """Validated listening-habit input accepted by the prediction endpoint."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    age: float | None = Field(default=None, ge=0, le=120, alias="Age")
    hours_per_day: float | None = Field(
        default=None,
        ge=0,
        le=24,
        alias="Hours per day",
    )
    primary_streaming_service: str | None = Field(
        default=None,
        alias="Primary streaming service",
    )
    while_working: str | None = Field(default=None, alias="While working")
    instrumentalist: str | None = Field(default=None, alias="Instrumentalist")
    composer: str | None = Field(default=None, alias="Composer")
    fav_genre: str | None = Field(default=None, alias="Fav genre")
    exploratory: str | None = Field(default=None, alias="Exploratory")
    foreign_languages: str | None = Field(
        default=None,
        alias="Foreign languages",
    )
    bpm: float | None = Field(default=None, ge=0, le=400, alias="BPM")
    music_effects: str | None = Field(default=None, alias="Music effects")

    frequency_classical: str | None = Field(
        default=None,
        alias="Frequency [Classical]",
    )
    frequency_country: str | None = Field(default=None, alias="Frequency [Country]")
    frequency_edm: str | None = Field(default=None, alias="Frequency [EDM]")
    frequency_folk: str | None = Field(default=None, alias="Frequency [Folk]")
    frequency_gospel: str | None = Field(default=None, alias="Frequency [Gospel]")
    frequency_hip_hop: str | None = Field(default=None, alias="Frequency [Hip hop]")
    frequency_jazz: str | None = Field(default=None, alias="Frequency [Jazz]")
    frequency_k_pop: str | None = Field(default=None, alias="Frequency [K pop]")
    frequency_latin: str | None = Field(default=None, alias="Frequency [Latin]")
    frequency_lofi: str | None = Field(default=None, alias="Frequency [Lofi]")
    frequency_metal: str | None = Field(default=None, alias="Frequency [Metal]")
    frequency_pop: str | None = Field(default=None, alias="Frequency [Pop]")
    frequency_r_and_b: str | None = Field(default=None, alias="Frequency [R&B]")
    frequency_rap: str | None = Field(default=None, alias="Frequency [Rap]")
    frequency_rock: str | None = Field(default=None, alias="Frequency [Rock]")
    frequency_video_game_music: str | None = Field(
        default=None,
        alias="Frequency [Video game music]",
    )


class PredictionResponse(BaseModel):
    """Prediction response returned by the API."""

    anxiety: float
    depression: float
    insomnia: float
    ocd: float
    disclaimer: str = "This is not a diagnostic tool."


@app.get("/health")
def health() -> dict[str, str]:
    """Return a liveness response without loading the model."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput) -> PredictionResponse:
    """Validate input and return predictions for all four targets."""
    try:
        scores, _ = predict_scores(payload.model_dump(by_alias=True, exclude_none=True))
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return PredictionResponse(
        **{target.lower(): scores[target] for target in MENTAL_HEALTH_TARGETS}
    )

