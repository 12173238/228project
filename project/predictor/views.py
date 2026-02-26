from django.shortcuts import render

from datetime import datetime

from .model_service import predict_energy


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

        # ===== 呼叫預測模型 =====
        pred_kwh, is_anomaly = predict_energy(
            target_date,
            lag1,
            temp,
            is_holiday
        )

        result = {
            "prediction": round(pred_kwh, 2),
            "is_anomaly": is_anomaly
        }

    return render(request, "predictor/index.html", {"result": result})