"""FastAPI service.

Endpoints:
  POST /ocr              upload an image, get text back through the agent
  GET  /review/queue     low confidence lines waiting for a human
  POST /review/{id}      submit a correction
  GET  /stats            counters for the dashboard
  GET  /healthz          liveness
  GET  /metrics          prometheus, added by the instrumentator

The static review UI is served from /, so one container gives you the API
and the frontend together.
"""

import time
import uuid

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from nastaliqops import review_store
from nastaliqops.config import Paths, load_params
from nastaliqops.llm.graph import build_graph
from nastaliqops.models.predictor import OcrPredictor, default_adapter_path


app = FastAPI(title="NastaliqOps", version="1.0.0")

UPLOAD_DIR = Paths.root / "data" / "uploads"
FRONTEND_DIR = Paths.root / "frontend"

STATE = {}


class CorrectionIn(BaseModel):
    corrected_text: str
    note: str = ""


@app.on_event("startup")
def startup():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    params = load_params()
    STATE["params"] = params

    adapter = default_adapter_path()
    predictor = OcrPredictor(adapter_path=adapter)
    STATE["predictor"] = predictor
    STATE["graph"] = build_graph(predictor)

    print("model loaded, variant " + predictor.variant)


@app.get("/healthz")
def healthz():
    status = {}
    status["ok"] = "predictor" in STATE
    if "predictor" in STATE:
        status["variant"] = STATE["predictor"].variant
    return status


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), save: str = Form("true")):
    if "graph" not in STATE:
        raise HTTPException(status_code=503, detail="model is still loading")

    raw = await file.read()
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode that image")

    file_name = str(uuid.uuid4()) + ".png"
    saved_path = UPLOAD_DIR / file_name
    cv2.imwrite(str(saved_path), image)

    started = time.time()
    final_state = STATE["graph"].invoke({"image": image})
    elapsed = time.time() - started

    record = {}
    record["image_path"] = file_name
    record["model_variant"] = final_state.get("variant", "unknown")
    record["raw_text"] = final_state.get("text", "")
    record["final_text"] = final_state.get("text", "")
    record["confidence"] = final_state.get("confidence", 0.0)
    record["route"] = final_state.get("route", "accept")
    record["llm_calls"] = final_state.get("llm_calls", 0)
    record["needs_review"] = final_state.get("needs_review", False)

    prediction_id = review_store.insert_prediction(record)

    response = {}
    response["id"] = prediction_id
    response["text"] = record["final_text"]
    response["confidence"] = record["confidence"]
    response["route"] = record["route"]
    response["reason"] = final_state.get("reason", "")
    response["llm_calls"] = record["llm_calls"]
    response["needs_review"] = record["needs_review"]
    response["model_variant"] = record["model_variant"]
    response["latency_seconds"] = elapsed
    return response


@app.get("/review/queue")
def review_queue(limit: int = 25):
    return {"items": review_store.pending_reviews(limit)}


@app.post("/review/{prediction_id}")
def review_submit(prediction_id: int, payload: CorrectionIn):
    ok = review_store.submit_correction(prediction_id, payload.corrected_text, payload.note)
    if not ok:
        raise HTTPException(status_code=404, detail="no such prediction")
    return {"ok": True}


@app.get("/stats")
def stats():
    return review_store.stats()


@app.get("/uploads/{file_name}")
def uploaded_image(file_name: str):
    path = UPLOAD_DIR / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(path))


Instrumentator().instrument(app).expose(app)

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
