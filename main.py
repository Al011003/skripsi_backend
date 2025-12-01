from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict

# ===========================
# FASTAPI INITIALIZATION
# ===========================
app = FastAPI(
    title="NPM Prediction API",
    description="API untuk prediksi Net Profit Margin perusahaan berdasarkan volatilitas pasar",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# LOAD MODELS & ENCODER
# ===========================
print("Loading models...")

try:
    with open("ml/model_revneg.pkl", "rb") as f:
        model_revneg = pickle.load(f)
    
    with open("ml/model_netprofneg.pkl", "rb") as f:
        model_netprofneg = pickle.load(f)
    
    # ✅ UPDATED: Load model NPM v2 (with volatility features)
    with open("ml/model_npm_v2.pkl", "rb") as f:
        model_npm = pickle.load(f)
    
    with open("ml/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    with open("ml/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open("ml/model_npm_metadata.pkl", "rb") as f:
        npm_metadata = pickle.load(f)
    
    print("✅ All models loaded successfully!")
    print(f"NPM Model Version: v2 (with volatility features)")
    print(f"NPM Model features: {model_npm.feature_names_in_}")
    print(f"Scaler features: {scaler.feature_names_in_}")
    print(f"NPM prediction will be clipped to: [{npm_metadata['npm_min_original']:.2f}, {npm_metadata['npm_max_original']:.2f}]")
    
except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}")
    print("Make sure all model files are in the 'ml/' directory")
    print("Required files:")
    print("  - model_revneg.pkl")
    print("  - model_netprofneg.pkl")
    print("  - model_npm_v2.pkl  ← NEW!")
    print("  - label_encoder.pkl")
    print("  - scaler.pkl")
    print("  - model_npm_metadata.pkl")

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
    """Convert kode_label ke kode_perusahaan menggunakan encoder"""
    try:
        return encoder.inverse_transform([kode_label])[0]
    except:
        return f"COMPANY_{kode_label}"

def calculate_composite_score(npm: float, revneg_conf: float, netprofneg_conf: float) -> float:
    """
    Hitung composite score untuk ranking
    
    Formula:
    - NPM: main factor (raw value)
    - Confidence: small bonus (max 0.1)
    """
    confidence_bonus = (revneg_conf + netprofneg_conf) / 2 * 0.1
    score = npm + confidence_bonus
    return score

def predict_for_all_companies(tahun: int, kuartal: int, ihsg: float, lq45: float):
    """
    Prediksi RevNeg, NetProfNeg, dan NPM untuk semua perusahaan (kode_label 0-70)
    Filter: hanya yang revneg=0 dan netprofneg=0
    Sort by: NPM prediction tertinggi
    """
    results = []
    
    # Loop semua kode_label (0-70)
    for kode_label in range(71):
        # 1. Prepare input untuk klasifikasi
        X_class = pd.DataFrame({
            'tahun': [tahun],
            'kuartal': [kuartal],
            'kode_label': [kode_label],
            'lq45': [lq45],
            'ihsg': [ihsg]
        })
        
        # 2. Prediksi RevNeg
        revneg_pred = model_revneg.predict(X_class)[0]
        revneg_proba_neg = model_revneg.predict_proba(X_class)[0][1]
        revneg_confidence = 1 - revneg_proba_neg
        
        # 3. Prediksi NetProfNeg
        netprofneg_pred = model_netprofneg.predict(X_class)[0]
        netprofneg_proba_neg = model_netprofneg.predict_proba(X_class)[0][1]
        netprofneg_confidence = 1 - netprofneg_proba_neg
        
        # 4. FILTER: Skip kalau revenue atau profit negatif
        if revneg_pred != 0 or netprofneg_pred != 0:
            continue
        
        # 5. FILTER: Skip kalau confidence terlalu rendah (< 50%)
        if revneg_confidence < 0.5 or netprofneg_confidence < 0.5:
            continue
        
        # ✅ UPDATED: Prepare input untuk regresi NPM v2 (7 features)
        X_reg = pd.DataFrame({
            'tahun': [tahun],
            'kuartal': [kuartal],          # ← ADDED
            'revneg': [int(revneg_pred)],
            'netprofneg': [int(netprofneg_pred)],
            'ihsg': [ihsg],                # ← ADDED
            'lq45': [lq45],                # ← ADDED
            'kode_label': [kode_label]
        })
        
        # 6. Prediksi NPM dengan model v2
        npm_pred = model_npm.predict(X_reg)[0]
        
        # 7. Calculate composite score
        composite_score = calculate_composite_score(npm_pred, revneg_confidence, netprofneg_confidence)
        
        # 8. Get company name
        company_name = get_company_name(kode_label, label_encoder)
        
        # 9. Store result
        results.append({
            'kode_label': int(kode_label),
            'kode_perusahaan': company_name,
            'revneg': int(revneg_pred),
            'revneg_confidence': float(revneg_confidence),
            'netprofneg': int(netprofneg_pred),
            'netprofneg_confidence': float(netprofneg_confidence),
            'npm_prediction': float(npm_pred),
            'composite_score': float(composite_score)
        })
    
    # Sort by NPM prediction (tertinggi)
    results_sorted = sorted(results, key=lambda x: x['npm_prediction'], reverse=True)
    
    # Ambil TOP 10
    top_10 = results_sorted[:10]
    
    # Add ranking
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
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "model_version": "v2",
        "features": list(model_npm.feature_names_in_)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_npm(input_data: PredictionInput):
    """
    Prediksi NPM untuk semua perusahaan dan return TOP 10 terbaik
    
    **v2 Updates:**
    - Model NPM sekarang menggunakan volatilitas (IHSG & LQ45) sebagai feature
    - Prediksi NPM akan berubah berdasarkan kondisi pasar
    
    **Filtering:**
    - Hanya perusahaan dengan Revenue POSITIF (revneg=0)
    - Hanya perusahaan dengan Net Profit POSITIF (netprofneg=0)
    - Confidence minimal 50% untuk revneg dan netprofneg
    
    **Sorting:**
    - Sorted by NPM prediction tertinggi
    
    **Input:**
    - tahun: Tahun prediksi (>=2025)
    - kuartal: Kuartal (1-4, jika 2025 maka >2)
    - ihsg: Volatilitas IHSG (decimal, contoh: 0.0234 = 2.34%)
    - lq45: Volatilitas LQ45 (decimal, contoh: 0.0156 = 1.56%)
    
    **Output:**
    - TOP 10 perusahaan dengan NPM tertinggi
    """
    try:
        # Jalankan prediksi untuk semua perusahaan
        top_predictions, total_analyzed, total_qualified = predict_for_all_companies(
            tahun=input_data.tahun,
            kuartal=input_data.kuartal,
            ihsg=input_data.ihsg,
            lq45=input_data.lq45
        )
        
        # Format response
        response = {
            "input": {
                "tahun": input_data.tahun,
                "kuartal": input_data.kuartal,
                "ihsg": input_data.ihsg,
                "lq45": input_data.lq45
            },
            "top_predictions": top_predictions,
            "total_analyzed": total_analyzed,
            "total_qualified": total_qualified
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ===========================
# RUN SERVER
# ===========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)