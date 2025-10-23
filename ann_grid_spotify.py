import os, time, warnings, json, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------
# 1) Import & Load dataset
# ------------------------------------------------------------
CSV = "spotify_churn_dataset.csv"  
if not os.path.exists(CSV):
    raise FileNotFoundError(
        f"Tidak menemukan {CSV}. Letakkan file CSV di folder kerja ini: {os.getcwd()}"
    )

df = pd.read_csv(CSV)

print("="*70)
print("📥 DATASET LOADED")
print(f"Baris, Kolom: {df.shape}")
print("5 data teratas:")
print(df.head())

# ------------------------------------------------------------
# 2) Cek kolom target
# ------------------------------------------------------------
TARGET_COL = "is_churned"
if TARGET_COL not in df.columns:
    raise KeyError(
        f"Kolom target '{TARGET_COL}' tidak ditemukan di dataset. "
        f"Kolom tersedia: {list(df.columns)}"
    )

drop_cols = [c for c in ["user_id"] if c in df.columns]
X = df.drop(columns=[TARGET_COL] + drop_cols)
y = df[TARGET_COL].astype(int) 

X = pd.get_dummies(X, drop_first=True)

feature_cols = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)
print("="*70)
print("✂️ Split data selesai.")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

hidden_configs = [
    [100, 50],
    [100, 50, 50],
]
activations = ["relu", "linear", "tanh", "sigmoid"]

# EarlyStopping supaya training tidak lama
early_stop = EarlyStopping(
    monitor="val_loss", mode="min", patience=5, restore_best_weights=True
)

def build_model(input_dim: int, hidden_layers: list[int], activation: str):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(Dense(units, activation=activation))
    # Output biner
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ------------------------------------------------------------
# 6) Latihan & evaluasi untuk setiap kombinasi
# ------------------------------------------------------------
results = []
best = {"acc": -1, "cfg": None, "activation": None, "model": None}

EPOCHS = 100
BATCH_SIZE = 32

for hl in hidden_configs:
    for act in activations:
        print("="*70)
        print(f"🚀 Training model: HIDDEN={hl} | AKTIVASI={act}")
        model = build_model(input_dim=X_train.shape[1], hidden_layers=hl, activation=act)

        t0 = time.time()
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[early_stop]
        )
        train_time = time.time() - t0

        # Evaluasi di test set
        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)

        # Simpan hasil dan ringkasan
        results.append({
            "HIDDEN_LAYER": str(hl),
            "AKTIVASI": act.upper(),
            "AKURASI": round(acc, 4),
            "EPOCHS_RILL": len(history.history["loss"]),
            "WAKTU_LATIH_DETIK": round(train_time, 2),
            "PARAMS": int(model.count_params()),
        })

        print(f"✅ Selesai: ACC={acc:.4f} | epochs={len(history.history['loss'])} | waktu={train_time:.2f}s")

        if acc > best["acc"]:
            best.update({"acc": acc, "cfg": hl, "activation": act, "model": model, "y_pred": y_pred})

# ------------------------------------------------------------
# 7) Tampilkan tabel hasil seperti di PDF dan simpan CSV
# ------------------------------------------------------------
df_res = pd.DataFrame(results).sort_values(by=["HIDDEN_LAYER", "AKTIVASI"]).reset_index(drop=True)

print("="*70)
print("📊 HASIL EKSPERIMEN")
print(df_res[["HIDDEN_LAYER", "AKTIVASI", "AKURASI"]])

OUT_CSV = "hasil_experimen_ann.csv"
df_res.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"\n💾 Hasil disimpan ke: {OUT_CSV}")

# ------------------------------------------------------------
# 8) Classification Report utk model terbaik
# ------------------------------------------------------------
print("="*70)
print("🏆 MODEL TERBAIK")
print(f"HIDDEN={best['cfg']} | AKTIVASI={best['activation'].upper()} | ACC={best['acc']:.4f}")

cls_rep = classification_report(y_test, best["y_pred"], digits=4)
print("\nClassification Report (Test):")
print(cls_rep)

with open("classification_report_best.txt", "w", encoding="utf-8") as f:
    f.write(f"MODEL TERBAIK\nHidden: {best['cfg']}\nAktivasi: {best['activation']}\nAkurasi: {best['acc']:.4f}\n\n")
    f.write(cls_rep)

print("\n💾 classification_report_best.txt tersimpan.")
print("✅ Selesai.")
