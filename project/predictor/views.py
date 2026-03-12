from django.shortcuts import render
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from .model_service import predict_energy
from .model_service import autoencoder, scaler, ae_threshold
from .weather_service import get_temp_for_date


# ==========================
# 讀取歷史資料
# ==========================
HISTORY_PATH = "predictor/history.csv"
history_df = pd.read_csv(HISTORY_PATH)
history_df["date"] = pd.to_datetime(history_df["date"])
history_df = history_df.sort_values("date")

if history_df.empty:
    print("[DEBUG] 歷史資料讀取失敗")
else:
    print(f"[DEBUG] 歷史資料讀取完成，共 {len(history_df)} 筆")

# ==========================
# 遞迴預測未來
# ==========================
def recursive_prediction(lag1, start_date, end_date, is_holiday, is_vacation):

    results = []
    current_date = start_date

    while current_date <= end_date:

        avg_temp = get_temp_for_date(current_date)

        pred_kwh, _ = predict_energy(
            current_date,
            lag1,
            avg_temp,
            is_holiday,
            is_vacation
        )

        results.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "pred_kwh": round(pred_kwh, 2)
        })

        lag1 = pred_kwh
        current_date += timedelta(days=1)

    return results


# ==========================
# 歷史七天異常檢測（使用 Autoencoder）
# ==========================
def check_last_7_days_anomaly():
    last_7_days = history_df.sort_values("date", ascending=False).head(7)
    results = []

    for _, row in last_7_days.iterrows():
        date = row["date"]
        actual_kwh = row["kwh"]

        # 直接用實際用電量做 Autoencoder
        scaled_input = scaler.transform([[actual_kwh]])
        reconstructed = autoencoder.predict(scaled_input)
        error = np.mean((scaled_input - reconstructed) ** 2)
        is_anomaly = error > ae_threshold

        results.append({
            "date": date.strftime("%Y-%m-%d"),
            "kwh": actual_kwh,
            "is_anomaly": bool(is_anomaly)
        })

    return results


# ==========================
# Django View
# ==========================
def index(request):

    result = None

    if request.method == "POST":

        date_str = request.POST.get("date")
        target_date = datetime.strptime(date_str, "%Y-%m-%d")

        is_holiday = request.POST.get("is_holiday") == "on"
        is_vacation = request.POST.get("is_vacation") == "on"

        # 最新歷史資料
        latest_history_date = history_df["date"].max()

        lag1 = float(
            history_df[history_df["date"] == latest_history_date]["kwh"].iloc[0]
        )

        # 預測開始日
        start_date = latest_history_date + timedelta(days=1)

        predictions = recursive_prediction(
            lag1,
            start_date,
            target_date,
            is_holiday,
            is_vacation
        )

        # 歷史七天異常檢測
        last_7_days = check_last_7_days_anomaly()

        result = {
            "target_date": target_date.strftime("%Y-%m-%d"),
            "pred_kwh": predictions[-1]["pred_kwh"],
            "last_7_days": last_7_days
        }

    return render(request, "predictor/index.html", {"result": result})