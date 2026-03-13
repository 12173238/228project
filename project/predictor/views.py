from django.shortcuts import render
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pandas import Timestamp

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
# ==========================
# 歷史七天異常檢測 (修正版：傳入當天數值比對殘差)
# ==========================
def check_last_7_days_anomaly():
    # 這裡我們取 8 筆，因為第 7 天需要第 8 天的 kwh 當作 lag1
    last_8_days = history_df.sort_values("date", ascending=False).head(8)
    results = []

    # 轉成 list 方便用索引抓前後兩天的資料
    data_list = last_8_days.to_dict('records')

    # 只檢查最近的 7 天
    for i in range(len(data_list) - 1):
        target_row = data_list[i]    # 當天 (要檢查的那天)
        prev_row = data_list[i+1]    # 前一天 (提供 lag1)

        target_date = target_row["date"]
        actual_kwh = target_row["kwh"]
        lag1_kwh = prev_row["kwh"] # 這是當天的前一天實際用電量

        # 準備特徵
        avg_temp = get_temp_for_date(target_date)
        h_row = holiday_df[holiday_df["date"] == target_date]
        is_h = int(h_row["is_holiday"].values[0]) if not h_row.empty else 0
        is_v = int(h_row["is_vacation"].values[0]) if not h_row.empty else 0

        # --- 【關鍵修改點】 ---
        # 不要在這裡自己算 scaler.transform
        # 而是呼叫 predict_energy，並把 actual_kwh 傳進去
        _, is_anomaly = predict_energy(
            target_date, 
            lag1_kwh, 
            avg_temp, 
            is_h, 
            is_v,
            actual_kwh=actual_kwh  # 讓模型知道當天實際是 1,000,000
        )

        results.append({
            "date": target_date.strftime("%Y-%m-%d"),
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
        target_date = Timestamp(datetime.strptime(date_str, "%Y-%m-%d"))  # <-- Pandas Timestamp
        today = Timestamp.today().normalize()  

        if target_date < today:
            result = {"error": "目標日不能是過去日期！"}
            return render(request, "predictor/index.html", {"result": result})
        elif target_date > today + pd.Timedelta(days=5):
            result = {"error": f"目標日超過5天 (今天 {today.strftime('%m/%d')}，最多可預測至 {(today + pd.Timedelta(days=5)).strftime('%m/%d')})"}
            return render(request, "predictor/index.html", {"result": result})


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