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

# ==========================
# 讀取假日資料
# ==========================
# CSV 範例：date,is_holiday,is_vacation
# 2026-03-10,0,0
# 2026-03-14,1,0
HOLIDAY_PATH = "predictor/holiday.csv"
holiday_df = pd.read_csv(HOLIDAY_PATH)
holiday_df["date"] = pd.to_datetime(holiday_df["date"])

# ==========================
# 遞迴預測未來（自動抓假日）
# ==========================
def recursive_prediction(lag1, start_date, end_date):
    results = []
    current_date = start_date

    while current_date <= end_date:
        avg_temp = get_temp_for_date(current_date)

        # 自動抓假日/寒暑假
        row = holiday_df[holiday_df["date"] == current_date]
        is_holiday = int(row["is_holiday"].values[0]) if not row.empty else 0
        is_vacation = int(row["is_vacation"].values[0]) if not row.empty else 0

        pred_kwh, _ = predict_energy(
            current_date,
            lag1,
            avg_temp,
            is_holiday,
            is_vacation
        )

        print(f"{current_date.strftime('%m/%d')} "
              f"溫度: {avg_temp} "
              f"放假: {is_holiday} "
              f"寒暑假: {is_vacation} "
              f"預測結果: {round(pred_kwh, 2)}")

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

        # 使用 Autoencoder 檢測異常（只用實際用電量）
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


        print(f"\n[DEBUG] 收到預測請求，目標日期: {date_str}")
        # 最新歷史資料
        latest_history_date = history_df["date"].max()
        lag1 = float(
            history_df[history_df["date"] == latest_history_date]["kwh"].iloc[0]
        )

        # 預測開始日
        start_date = latest_history_date + timedelta(days=1)

        # 遞迴預測
        predictions = recursive_prediction(
            lag1,
            start_date,
            target_date
        )

        target_row = holiday_df[holiday_df["date"] == target_date]
        if not target_row.empty:
            h_status = target_row["is_holiday"].values[0]
            v_status = target_row["is_vacation"].values[0]
            print(f"確認目標日資料：")
            print(f" >> 日期: {target_date.strftime('%Y-%m-%d')}")
            print(f" >> 是否假日: {h_status} (1為是, 0為否)")
            print(f" >> 是否寒暑假: {v_status} (1為是, 0為否)")
        else:
            print(f"⚠️ 警告: holiday.csv 中找不到 {date_str} 的資料！")


        # 歷史七天異常檢測
        last_7_days = check_last_7_days_anomaly()

        result = {
            "target_date": target_date.strftime("%Y-%m-%d"),
            "pred_kwh": predictions[-1]["pred_kwh"],
            "last_7_days": last_7_days
        }

    return render(request, "predictor/index.html", {"result": result})