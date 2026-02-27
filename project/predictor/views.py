from django.shortcuts import render

from datetime import datetime

from .model_service import predict_energy, predict_next_7_days

def index(request):

    result = None

    if request.method == "POST":

        # ===== 取得使用者輸入 =====
        date_str = request.POST.get("date")
        lag1 = float(request.POST.get("lag1"))
        temp = float(request.POST.get("temp"))

        is_holiday = request.POST.get("is_holiday") == "on"

        # ===== 轉換日期 =====
        target_date = datetime.strptime(date_str, "%Y-%m-%d")

        # ===== 呼叫單日預測 =====
        pred_kwh, is_anomaly = predict_energy(
            target_date,
            lag1,
            temp,
            is_holiday
        )

        # ===== 呼叫七日預測 =====
        weekly_predictions = predict_next_7_days(
            target_date,
            lag1,
            temp,
            is_holiday
        )

        result = {
            "next_day": round(pred_kwh, 2),
            "is_anomaly": is_anomaly,
            "next_week": weekly_predictions
        }

    return render(request, "predictor/index.html", {"result": result})