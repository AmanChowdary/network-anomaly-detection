"""
Network Anomaly Detection — FastAPI REST Service
Serves real-time anomaly predictions from the trained XGBoost model.
"""
import os, json, sys
from typing import List
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features.extract import prepare_features

app = FastAPI(
    title="Network Anomaly Detection API",
    description="Real-time network traffic anomaly detection powered by XGBoost + SMOTE",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH     = os.getenv("MODEL_PATH", "model/artifacts/anomaly_model.joblib")
FEATURES_PATH  = os.getenv("FEATURES_PATH", "model/artifacts/feature_names.joblib")
_model         = None
_feature_names = None

def load_model():
    global _model, _feature_names
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Run model/train.py first.")
        _model = joblib.load(MODEL_PATH)
        _feature_names = joblib.load(FEATURES_PATH)
    return _model, _feature_names

class TrafficEvent(BaseModel):
    event_id:          str   = Field(..., example="EVT_000001")
    duration_sec:      float = Field(..., ge=0, example=2.4)
    src_bytes:         float = Field(..., ge=0, example=3200.0)
    dst_bytes:         float = Field(..., ge=0, example=1024.0)
    src_packets:       int   = Field(..., ge=0, example=20)
    dst_packets:       int   = Field(..., ge=0, example=18)
    packet_size_avg:   float = Field(..., ge=0, example=500.0)
    packet_size_std:   float = Field(..., ge=0, example=80.0)
    inter_arrival_ms:  float = Field(..., ge=0, example=50.0)
    protocol:          int   = Field(..., example=6, description="6=TCP 17=UDP 1=ICMP")
    src_port:          int   = Field(..., ge=0, le=65535, example=443)
    dst_port:          int   = Field(..., ge=0, le=65535, example=54321)
    flags_syn:         int   = Field(..., ge=0, le=1, example=1)
    flags_ack:         int   = Field(..., ge=0, le=1, example=1)
    flags_fin:         int   = Field(..., ge=0, le=1, example=0)
    flags_rst:         int   = Field(..., ge=0, le=1, example=0)
    land:              int   = Field(..., ge=0, le=1, example=0)
    wrong_fragment:    int   = Field(..., ge=0, example=0)
    urgent:            int   = Field(..., ge=0, example=0)
    protocol_entropy:  float = Field(..., ge=0, le=1, example=0.6)

class AnomalyResponse(BaseModel):
    event_id:            str
    is_anomaly:          bool
    anomaly_probability: float
    severity:            str
    signals:             List[str]

def severity(prob: float) -> str:
    if prob >= 0.8: return "CRITICAL"
    if prob >= 0.5: return "HIGH"
    if prob >= 0.3: return "MEDIUM"
    return "LOW"

def anomaly_signals(event: dict) -> List[str]:
    signals = []
    if event.get("flags_syn", 0) == 1 and event.get("flags_rst", 0) == 1:
        signals.append("SYN+RST flag combination — possible brute force")
    if event.get("src_bytes", 0) > 1e7:
        signals.append("Extremely high outbound bytes — possible DoS/exfiltration")
    if event.get("duration_sec", 1) < 0.1 and event.get("src_packets", 10) < 5:
        signals.append("Very short connection — possible port scan")
    if event.get("wrong_fragment", 0) > 0:
        signals.append("Fragmented packets detected")
    if event.get("land", 0) == 1:
        signals.append("Source and destination IP identical — land attack")
    return signals[:3] if signals else ["No anomalous signals detected"]

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict", response_model=AnomalyResponse)
async def predict(event: TrafficEvent):
    model, feature_names = load_model()
    data = event.dict()
    eid = data.pop("event_id")
    data["label"] = 0
    df = pd.DataFrame([data])
    try:
        X, _ = prepare_features(df)
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]
        prob = float(model.predict_proba(X)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return AnomalyResponse(event_id=eid, is_anomaly=prob >= 0.5,
                           anomaly_probability=round(prob, 4),
                           severity=severity(prob),
                           signals=anomaly_signals(event.dict()))

@app.post("/predict/batch")
async def predict_batch(events: List[TrafficEvent]):
    model, feature_names = load_model()
    records, eids = [], []
    for event in events:
        data = event.dict(); eids.append(data.pop("event_id")); data["label"] = 0; records.append(data)
    df = pd.DataFrame(records)
    X, _ = prepare_features(df)
    for col in feature_names:
        if col not in X.columns: X[col] = 0
    X = X[feature_names]
    probs = model.predict_proba(X)[:, 1]
    return {"predictions": [{"event_id": e, "is_anomaly": bool(p >= 0.5),
            "anomaly_probability": round(float(p), 4), "severity": severity(p)}
            for e, p in zip(eids, probs)], "total": len(eids)}

@app.get("/model/metrics")
async def model_metrics():
    metrics_path = "model/artifacts/metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(404, "Metrics not found.")
    with open(metrics_path) as f:
        return json.load(f)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
