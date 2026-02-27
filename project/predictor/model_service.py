import numpy as np
import pandas as pd
import joblib

# ===============================
# 載入模型（只載入一次）
# ===============================

MODEL_PATH = "model.pkl"
AE_PATH = "ae.pkl"
SCALER_PATH = "scaler.pkl"
HISTORY_PATH = "114_dailykwh_cleaned1.csv"
THRESHOLD_PATH = "ae_threshold.pkl"


ae_threshold = joblib.load(THRESHOLD_PATH)
model = joblib.load(MODEL_PATH)
autoencoder = joblib.load(AE_PATH)
scaler = joblib.load(SCALER_PATH)

# 歷史資料
history_df = pd.read_csv(HISTORY_PATH)
history_df["date"] = pd.to_datetime(history_df["date"])


# ===============================
# 預測核心函數
# ===============================

def predict_energy(target_date,
                   input_kwh_lag1,
                   input_temp,
                   is_holiday_input):

    global model, autoencoder, scaler, history_df

    # ---------- 日期特徵 ----------
    month = target_date.month
    weekday = target_date.weekday() + 1

    # 假日欄位
    is_holiday = 1 if is_holiday_input else 0
    is_vacation = 0

    # ---------- lag7 查歷史 ----------
    lag7_date = target_date - pd.Timedelta(days=7)

    lag7_row = history_df[history_df["date"] == lag7_date]

    if len(lag7_row) > 0:
        kwh_lag7 = lag7_row["kwh"].values[0]
    else:
        kwh_lag7 = history_df["kwh"].mean()

    # ---------- rolling7 ----------
    rolling7 = history_df[
        (history_df["date"] < target_date) &
        (history_df["date"] >= target_date - pd.Timedelta(days=7))
    ]["kwh"].mean()

    if np.isnan(rolling7):
        rolling7 = history_df["kwh"].mean()

    # ---------- 特徵構造 ----------
    avg_temp = input_temp
    avg_temp_sq = avg_temp ** 2

    feature_vector = np.array([[
        month,
        weekday,
        is_holiday,
        is_vacation,
        avg_temp,
        avg_temp_sq,
        input_kwh_lag1,
        kwh_lag7,
        rolling7
    ]])

    # ---------- 預測 ----------
    pred_kwh = model.predict(feature_vector)[0]

    # ---------- 殘差 ----------
    residual = input_kwh_lag1 - pred_kwh

    # ---------- Autoencoder 判斷 ----------
    residual_scaled = scaler.transform([[residual]])

    reconstructed = autoencoder.predict(residual_scaled)

    error = np.mean((residual_scaled - reconstructed) ** 2)


    is_anomaly = error > ae_threshold

    return pred_kwh, is_anomaly


def predict_next_7_days(start_date,
                        input_kwh_lag1,
                        input_temp,
                        is_holiday_input):

    predictions = []

    current_date = start_date
    current_lag1 = input_kwh_lag1

    for i in range(7):

        pred_kwh, _ = predict_energy(
            current_date,
            current_lag1,
            input_temp,
            is_holiday_input
        )

        predictions.append({
            "date": current_date,
            "prediction": round(pred_kwh, 2)
        })

        # 下一天準備
        current_lag1 = pred_kwh
        current_date = current_date + pd.Timedelta(days=1)

    return predictions