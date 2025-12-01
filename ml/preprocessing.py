import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_dataframe(df):
    """
    Melakukan preprocessing data agar siap dipakai untuk training ML.
    """
    print(">>> Preprocessing started...")

    # ===================================================
    # 1. Drop missing values
    # ===================================================
    print("Step 1: Dropping missing values...")
    df = df.dropna()

    # ===================================================
    # 2. Winsorizing kolom NPM
    # ===================================================
    print("Step 2: Winsorizing NPM...")
    lower = df["NPM"].quantile(0.05)
    upper = df["NPM"].quantile(0.95)
    df["NPM_winsor"] = np.clip(df["NPM"], lower, upper)

    # ===================================================
    # 3. Label encoding kode perusahaan
    # ===================================================
    print("Step 3: Label Encoding kode_perusahaan...")
    le = LabelEncoder()
    df["kode_label"] = le.fit_transform(df["kode_perusahaan"])

    # ===================================================
    # 4. Normalisasi IHSG, LQ45, dan NPM_winsor
    # ===================================================
    print("Step 4: Scaling IHSG, LQ45, dan NPM_winsor...")
    scaler = StandardScaler()
    cols_to_scale = ["ihsg", "lq45", "NPM_winsor"]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # ===================================================
    # 5. Drop kolom tidak penting untuk model
    # ===================================================

    print(">>> Preprocessing completed!\n")
    return df, le, scaler
