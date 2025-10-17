# ======================================================
# 1️⃣ 필요한 라이브러리
# ======================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import os

# ======================================================
# 2️⃣ 데이터 불러오기
# ======================================================
drive_path = "/content/drive/MyDrive"
file_path = os.path.join(drive_path, "2014_2020_시계열_지하수_기상_train.csv")
model_dir = os.path.join(drive_path, "models")
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(file_path, encoding="cp949")
df["ymd"] = pd.to_datetime(df["ymd"], errors="coerce")
df = df.sort_values(["code_new", "ymd"]).reset_index(drop=True)
df = df.interpolate(limit_direction="both")

# ======================================================
# 3️⃣ NSE, KGE 계산 함수 (평가 전용)
# ======================================================
def calc_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - numerator / (denominator + 1e-9)

def calc_kge(y_true, y_pred):
    r = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-9)
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-9)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

# ======================================================
# 4️⃣ 시퀀스 생성 함수
# ======================================================
def create_sequences(data, seq_length=365, target_col="elev"):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        y.append(data.iloc[i + seq_length][target_col])
    return np.array(X), np.array(y)

# ======================================================
# 5️⃣ 모델 정의 함수
# ======================================================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")  # 🔹 MSE 손실로 학습
    return model

# ======================================================
# 6️⃣ 지역별 학습 함수
# ======================================================
def train_region_model(region_id, seq_length=365, epochs=50):
    region_df = df[df["code_new"] == region_id].copy()
    feature_cols = ['wtemp', 'ec', '기온(°C)', '강수량(mm)', '풍속(m/s)',
                    '습도(%)', '현지기압(hPa)', '지면온도(°C)']
    target_col = "elev"

    # 표준화
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    region_df[feature_cols] = scaler_x.fit_transform(region_df[feature_cols])
    region_df[target_col] = scaler_y.fit_transform(region_df[[target_col]])

    X, y = create_sequences(region_df[feature_cols + [target_col]], seq_length, target_col)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = build_lstm_model((X.shape[1], X.shape[2]))

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ckpt_path = os.path.join(model_dir, f"region_{region_id}.h5")
    ckpt = ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)

    print(f"\n🚀 Training Region {region_id}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[es, ckpt],
        verbose=1
    )

    print(f"✅ Region {region_id} model saved to: {ckpt_path}")

    # 학습/검증 손실 시각화
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Region {region_id} Loss Curve (MSE 기반)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, scaler_x, scaler_y, X_val, y_val

# ======================================================
# 7️⃣ 모델 평가 (NSE/KGE)
# ======================================================
def evaluate_model(model, scaler_y, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_val_inv = scaler_y.inverse_transform(y_val.reshape(-1, 1))
    nse = calc_nse(y_val_inv, y_pred_inv)
    kge = calc_kge(y_val_inv, y_pred_inv)
    print(f"📊 NSE = {nse:.4f}, KGE = {kge:.4f}")

    # 실제 vs 예측 시각화
    plt.figure(figsize=(10,5))
    plt.plot(y_val_inv[:500], label='Actual')
    plt.plot(y_pred_inv[:500], label='Predicted')
    plt.title("Actual vs Predicted Groundwater Level")
    plt.xlabel("Time Step")
    plt.ylabel("Elevation (m)")
    plt.legend()
    plt.grid(True)
    plt.show()


