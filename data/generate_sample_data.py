"""
Utility script to generate a synthetic soil_measures.csv dataset.

The real dataset can be downloaded from:
  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Run this script ONLY if you do not have the original soil_measures.csv:
    python data/generate_sample_data.py
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Crop profiles: (N_mean, P_mean, K_mean, ph_mean) with realistic ranges
CROP_PROFILES = {
    "rice":        (80,  40,  40, 6.5),
    "maize":       (78,  48,  20, 6.0),
    "chickpea":    (40,  68,  80, 7.2),
    "kidneybeans": (20,  67,  20, 5.8),
    "pigeonpeas":  (20,  68,  20, 5.5),
    "mothbeans":   (21,  48,  20, 6.5),
    "mungbean":    (20,  47,  20, 6.8),
    "blackgram":   (40,  68,  20, 7.0),
    "lentil":      (18,  68,  19, 6.5),
    "pomegranate": (18,  18,  40, 6.0),
    "banana":      (100, 82, 50, 6.0),
    "mango":       (20,  27,  30, 6.2),
    "grapes":      (23,  132,200, 6.0),
    "watermelon":  (99,  35,  50, 6.5),
    "muskmelon":   (100, 17,  50, 6.5),
    "apple":       (21,  134,200, 6.0),
    "orange":      (20,  16,  10, 7.2),
    "papaya":      (49,  59,  50, 6.8),
    "coconut":     (22,  16,  30, 5.8),
    "cotton":      (118, 46,  20, 7.0),
    "jute":        (78,  46,  40, 6.5),
    "coffee":      (101, 28,  30, 6.5),
}

records = []
for crop, (n_mu, p_mu, k_mu, ph_mu) in CROP_PROFILES.items():
    n_samples = 100
    N  = rng.normal(n_mu,  10, n_samples).clip(0, 200)
    P  = rng.normal(p_mu,  10, n_samples).clip(0, 200)
    K  = rng.normal(k_mu,  10, n_samples).clip(0, 250)
    ph = rng.normal(ph_mu, 0.5, n_samples).clip(3.5, 9.5)
    for i in range(n_samples):
        records.append({"N": round(N[i], 2), "P": round(P[i], 2),
                        "K": round(K[i], 2), "ph": round(ph[i], 2),
                        "crop": crop})

df = pd.DataFrame(records).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
df.to_csv("data/soil_measures.csv", index=False)
print(f"[OK] Generated soil_measures.csv  -- {len(df)} rows, {df['crop'].nunique()} crops")
