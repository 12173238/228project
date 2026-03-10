import requests
import pandas as pd
from datetime import datetime

API_KEY = "8241283a4ec48f41fcad7a4e6dcd42a4"
CITY = "Taipei,TW"  # 建議加上國家代碼
URL = "https://api.openweathermap.org/data/2.5/forecast"

def get_temp_for_date(target_date: datetime):
    params = {"q": CITY, "appid": API_KEY, "units": "metric"}
    res = requests.get(URL, params=params)
    data = res.json()

    if "list" not in data:
        print("⚠️ API 回傳錯誤:", data)
        return 25.0  # fallback 溫度
        

    rows = []
    for item in data["list"]:
        date = item["dt_txt"].split(" ")[0]
        temp = item["main"]["temp"]
        rows.append({"date": date, "temp": temp})

    df = pd.DataFrame(rows)
    result = df.groupby("date")["temp"].agg(temp_min="min", temp_max="max", temp_avg="mean").reset_index()
    result = result.round(2)

    row = result[result["date"] == target_date.strftime("%Y-%m-%d")]
    if not row.empty:
        return float(row["temp_avg"])
    else:
        return 25.0  # fallback