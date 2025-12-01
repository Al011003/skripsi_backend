from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict
import os

# ===========================
# FASTAPI INITIALIZATION
# ===========================
app = FastAPI(
    title="NPM Prediction API",
    description="API untuk prediksi Net Profit Margin perusahaan berdasarkan volatilitas pasar",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
async def startup_event():
    print("🚀 Application startup complete, ready to accept requests")
# ===========================
# LOAD MODELS & ENCODER
# ===========================
print("Loading models...")

model_files = {
    "model_revneg": "ml/model_revneg.pkl",
    "model_netprofneg": "ml/model_netprofneg.pkl",
    "model_npm": "ml/model_npm_v2.pkl",
    "label_encoder": "ml/label_encoder.pkl",
    "scaler": "ml/scaler.pkl",
    "npm_metadata": "ml/model_npm_metadata.pkl"
}

models = {}
all_loaded = True

for name, path in model_files.items():
    if not os.path.exists(path):
        print(f"❌ Missing model file: {path}")
        all_loaded = False
    else:
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"✅ Loaded {name} from {path}")

if all_loaded:
    print("✅ All models loaded successfully!")
    print(f"NPM Model Version: v2 (with volatility features)")
    print(f"NPM Model features: {models['model_npm'].feature_names_in_}")
    print(f"Scaler features: {models['scaler'].feature_names_in_}")
    print(f"NPM prediction will be clipped to: [{models['npm_metadata']['npm_min_original']:.2f}, {models['npm_metadata']['npm_max_original']:.2f}]")
else:
    print("⚠️ Some model files are missing. API may not work properly.")

# Assign loaded models to variables for code compatibility
model_revneg = models.get("model_revneg")
model_netprofneg = models.get("model_netprofneg")
model_npm = models.get("model_npm")
label_encoder = models.get("label_encoder")
scaler = models.get("scaler")
npm_metadata = models.get("npm_metadata")

# ===========================
# PYDANTIC MODELS
# ===========================
class PredictionInput(BaseModel):
    tahun: int = Field(..., ge=2025, description="Tahun prediksi (minimal 2025)")
    kuartal: int = Field(..., ge=1, le=4, description="Kuartal (1-4)")
    ihsg: float = Field(..., description="Volatilitas IHSG (contoh: 0.0234)")
    lq45: float = Field(..., description="Volatilitas LQ45 (contoh: 0.0156)")
    
    @validator('kuartal')
    def validate_kuartal(cls, v, values):
        if 'tahun' in values and values['tahun'] == 2025 and v <= 2:
            raise ValueError("Untuk tahun 2025, kuartal harus lebih dari 2")
        return v

class CompanyPrediction(BaseModel):
    rank: int
    kode_label: int
    kode_perusahaan: str
    revneg: int
    revneg_confidence: float
    netprofneg: int
    netprofneg_confidence: float
    npm_prediction: float
    composite_score: float

class PredictionResponse(BaseModel):
    input: Dict
    top_predictions: List[CompanyPrediction]
    total_analyzed: int
    total_qualified: int

# ===========================
# HELPER FUNCTIONS
# ===========================
def get_company_name(kode_label: int, encoder) -> str:
    try:
        return encoder.inverse_transform([kode_label])[0]
    except:
        return f"COMPANY_{kode_label}"

def calculate_composite_score(npm: float, revneg_conf: float, netprofneg_conf: float) -> float:
    confidence_bonus = (revneg_conf + netprofneg_conf) / 2 * 0.1
    return npm + confidence_bonus

def predict_for_all_companies(tahun: int, kuartal: int, ihsg: float, lq45: float):
    results = []
    for kode_label in range(71):
        X_class = pd.DataFrame({
            'tahun': [tahun],
            'kuartal': [kuartal],
            'kode_label': [kode_label],
            'lq45': [lq45],
            'ihsg': [ihsg]
        })
        revneg_pred = model_revneg.predict(X_class)[0]
        revneg_conf = 1 - model_revneg.predict_proba(X_class)[0][1]
        netprofneg_pred = model_netprofneg.predict(X_class)[0]
        netprofneg_conf = 1 - model_netprofneg.predict_proba(X_class)[0][1]
        if revneg_pred != 0 or netprofneg_pred != 0: continue
        if revneg_conf < 0.5 or netprofneg_conf < 0.5: continue
        X_reg = pd.DataFrame({
            'tahun': [tahun],
            'kuartal': [kuartal],
            'revneg': [int(revneg_pred)],
            'netprofneg': [int(netprofneg_pred)],
            'ihsg': [ihsg],
            'lq45': [lq45],
            'kode_label': [kode_label]
        })
        npm_pred = model_npm.predict(X_reg)[0]
        composite_score = calculate_composite_score(npm_pred, revneg_conf, netprofneg_conf)
        company_name = get_company_name(kode_label, label_encoder)
        results.append({
            'kode_label': kode_label,
            'kode_perusahaan': company_name,
            'revneg': int(revneg_pred),
            'revneg_confidence': float(revneg_conf),
            'netprofneg': int(netprofneg_pred),
            'netprofneg_confidence': float(netprofneg_conf),
            'npm_prediction': float(npm_pred),
            'composite_score': float(composite_score)
        })
    results_sorted = sorted(results, key=lambda x: x['npm_prediction'], reverse=True)
    top_10 = results_sorted[:10]
    for idx, result in enumerate(top_10, 1):
        result['rank'] = idx
    return top_10, 71, len(results)

# ===========================
# API ENDPOINTS
# ===========================
@app.get("/")
def read_root():
    return {
        "message": "NPM Prediction API v2.0",
        "status": "running",
        "model_version": "v2 (with volatility features)",
        "endpoints": {"predict": "/predict (POST)", "health": "/health (GET)"}
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": all_loaded,
        "model_version": "v2" if all_loaded else "unknown",
        "features": list(model_npm.feature_names_in_) if all_loaded else []
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_npm(input_data: PredictionInput):
    try:
        top_predictions, total_analyzed, total_qualified = predict_for_all_companies(
            tahun=input_data.tahun,
            kuartal=input_data.kuartal,
            ihsg=input_data.ihsg,
            lq45=input_data.lq45
        )
        return {
            "input": input_data.dict(),
            "top_predictions": top_predictions,
            "total_analyzed": total_analyzed,
            "total_qualified": total_qualified
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ===========================
# RUN SERVER
# ===========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
