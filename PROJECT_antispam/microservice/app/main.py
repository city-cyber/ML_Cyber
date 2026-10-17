"""API для классификации текста и RFC 822/MIME-писем."""

import os
from contextlib import asynccontextmanager
from email.message import EmailMessage
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.artifacts import DEFAULT_MODEL_DIR, load_artifact
from src.inference import spam_scores
from src.preprocessing import normalize, parse_email

MAX_EML_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARS = 200_000


class EmailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"subject": "Team meeting", "body": "Please review the project report before Monday."},
        {"text": "Congratulations! Claim your free prize now!"},
    ]})
    text: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)
    subject: str = Field(default="", max_length=2000)
    body: str = Field(default="", max_length=MAX_TEXT_CHARS)
    body_format: Literal["plain", "html"] = "plain"

    @field_validator("subject")
    @classmethod
    def subject_has_no_newlines(cls, value):
        if "\r" in value or "\n" in value:
            raise ValueError("Тема должна быть одной строкой")
        return value

    @model_validator(mode="after")
    def check_content(self):
        if self.text is not None:
            if self.subject or self.body or self.body_format != "plain":
                raise ValueError("Передайте text либо subject/body")
            if not normalize(self.text):
                raise ValueError("Текст письма пуст")
        elif not normalize(self.subject + " " + self.body):
            raise ValueError("Укажите text либо непустые subject/body")
        return self

    def normalized_text(self):
        if self.text is not None:
            return normalize(self.text)
        message = EmailMessage()
        message["Subject"] = self.subject
        message.set_content(self.body, subtype=self.body_format, charset="utf-8")
        return parse_email(message.as_bytes())["text"]


class Prediction(BaseModel):
    label: Literal["ham", "spam"]
    is_spam: bool
    score: float
    threshold: float
    score_kind: str
    model_name: str


def create_app(model_dir: Path | None = None):
    directory = Path(model_dir or os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR)))

    @asynccontextmanager
    async def lifespan(application):
        # Один раз при старте; без артефакта или с несовместимыми версиями старт прерывается.
        bundle, metadata = load_artifact(directory)
        application.state.bundle = bundle
        application.state.metadata = metadata
        yield

    api = FastAPI(
        title="Антиспам API", version="1.0.0", lifespan=lifespan,
        description=("Классификация писем моделью TF-IDF + ComplementNB. "
                     "Порог выбран с приоритетом снижения ложных блокировок. "
                     "score — оценка модели, а не отдельно откалиброванная вероятность."),
    )

    def predict_text(text, request):
        if not text.strip():
            raise HTTPException(status_code=422, detail="После обработки текст письма пуст")
        bundle = request.app.state.bundle
        score = float(spam_scores(bundle["pipeline"], [text])[0])
        is_spam = score >= bundle["threshold"]
        return Prediction(
            label="spam" if is_spam else "ham", is_spam=is_spam,
            score=score, threshold=bundle["threshold"],
            score_kind=bundle["score_kind"], model_name=bundle["model_name"],
        )

    @api.get("/", include_in_schema=False)
    def home():
        return RedirectResponse(url="/docs")

    @api.get("/health", tags=["Состояние"])
    def health(request: Request):
        return {"status": "ok", "model_loaded": True,
                "model_name": request.app.state.bundle["model_name"]}

    @api.get("/model", tags=["Состояние"])
    def model_info(request: Request):
        metadata = request.app.state.metadata
        fields = ["model_name", "threshold", "score_kind", "preprocessing_version",
                  "model_sha256", "training_rows", "created_at_utc", "packages"]
        return {key: metadata[key] for key in fields}

    @api.post("/predict", response_model=Prediction, tags=["Классификация"])
    def predict(payload: EmailRequest, request: Request):
        try:
            text = payload.normalized_text()
        except (ValueError, RecursionError) as exc:
            raise HTTPException(status_code=422, detail="Не удалось разобрать письмо") from exc
        return predict_text(text, request)

    @api.post("/predict/eml", response_model=Prediction, tags=["Классификация"])
    def predict_eml(request: Request, file: Annotated[UploadFile, File(description="RFC 822/MIME .eml, до 2 МиБ")]):
        raw = file.file.read(MAX_EML_BYTES + 1)
        if len(raw) > MAX_EML_BYTES:
            raise HTTPException(status_code=413, detail="Письмо больше 2 МиБ")
        if not raw.strip():
            raise HTTPException(status_code=422, detail="Файл письма пуст")
        try:
            text = parse_email(raw)["text"]
        except (ValueError, RecursionError) as exc:
            raise HTTPException(status_code=422, detail="Не удалось разобрать MIME-письмо") from exc
        return predict_text(text, request)

    return api


app = create_app()
