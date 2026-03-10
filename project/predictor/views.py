from django.shortcuts import render
from datetime import datetime, timedelta
import pandas as pd

from .model_service import predict_energy  # 你的模型函式
from .weather_service import get_temp_for_date  # 你的天氣 API

# ==========================
# 讀取歷史資料 CSV（啟動就讀一次）
# ==========================
HISTORY_PATH = "predictor/history.csv"
history_df = pd.read_csv(HISTORY_PATH)
history_df["date"] = pd.to_datetime(history_df["date"])

if history_df.empty:
    print("[DEBUG] 歷史資料讀取失敗，DataFrame 為空")
else:
    print(f"[DEBUG] 歷史資料讀取完成，共 {len(history_df)} 筆資料")
    print(history_df.head())  # 顯示前五筆確認

# ==========================
# 遞迴預測（不檢查異常）
# ==========================
def recursive_prediction(lag1, start_date, end_date, is_holiday, is_vacation):
    results = []
    current_date = start_date

    while current_date <= end_date:
        avg_temp = get_temp_for_date(current_date)
        print(f"[DEBUG] 溫度 API - {current_date.strftime('%Y-%m-%d')}: {avg_temp} °C")

        pred_kwh, _ = predict_energy(
            current_date,
            lag1,
            avg_temp,
            is_holiday,
            is_vacation
        )

        print(f"[DEBUG] 遞迴預測 - {current_date.strftime('%Y-%m-%d')}: {round(pred_kwh,2)} kWh")

        results.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "pred_kwh": round(pred_kwh, 2)
        })

        lag1 = pred_kwh
        current_date += timedelta(days=1)

    return results

# ==========================
# 取得歷史最近七天資料
# ==========================
def get_last_7_days_history():
    last_7_days = history_df.sort_values(by="date", ascending=False).head(7)
    print("[DEBUG] 歷史最近七天用電量:")
    for _, row in last_7_days.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['kwh']} kWh")
    return last_7_days

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

        print(f"[DEBUG] 使用者選擇目標日期: {target_date.strftime('%Y-%m-%d')}")
        print(f"[DEBUG] 放假: {is_holiday}, 寒暑假: {is_vacation}")

        # 最新歷史資料日期
        latest_history_date = history_df["date"].max()
        # lag1 取最新歷史資料的用電量
        lag1 = float(history_df[history_df["date"] == latest_history_date]["kwh"].iloc[0])
        print(f"[DEBUG] 用 {latest_history_date.strftime('%Y-%m-%d')} 的歷史資料作為 lag1: {lag1} kWh")

        # 從最新歷史資料的下一天開始預測
        start_date = latest_history_date + timedelta(days=1)
        print(f"[DEBUG] 遞迴預測起始日期: {start_date.strftime('%Y-%m-%d')}")
        predictions = recursive_prediction(lag1, start_date, target_date, is_holiday, is_vacation)

        # 歷史最近七天資料
        last_7_days = get_last_7_days_history()

        result = {
            "target_date": target_date.strftime("%Y-%m-%d"),
            "pred_kwh": predictions[-1]["pred_kwh"],  # 目標日期預測用電量
            "last_7_days": [
                {"date": row["date"].strftime("%Y-%m-%d"), "kwh": row["kwh"]}
                for _, row in last_7_days.iterrows()
            ]
        }

    return render(request, "predictor/index.html", {"result": result})