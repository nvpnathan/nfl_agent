import requests
from typing import Optional

OUTDOOR_STADIUM_COORDS = {
    "BUF": (42.774, -78.787), "KC": (39.049, -94.484), "GB": (44.501, -88.062),
    "CHI": (41.862, -87.617), "PIT": (40.447, -80.016), "CLE": (41.517, -81.700),
    "CIN": (39.095, -84.516), "BAL": (39.278, -76.623), "NE": (42.091, -71.264),
    "NYG": (40.813, -74.074), "NYJ": (40.813, -74.074), "PHI": (39.901, -75.168),
    "WAS": (38.908, -76.864), "DAL": (32.748, -97.093), "CAR": (35.225, -80.853),
    "JAX": (30.324, -81.638), "TEN": (36.166, -86.771), "HOU": (29.685, -95.411),
    "IND": (39.760, -86.164), "MIA": (25.958, -80.239), "DEN": (39.744, -105.020),
    "LV": (36.091, -115.184), "SEA": (47.595, -122.332), "SF": (37.403, -121.970),
    "LAR": (33.953, -118.339), "LAC": (33.953, -118.339), "ARI": (33.528, -112.263),
    "MIN": (44.974, -93.258), "DET": (42.340, -83.045), "TB": (27.976, -82.503),
    "NO": (29.951, -90.081), "ATL": (33.755, -84.401),
}

INDOOR_TEAMS = {"NO", "ATL", "MIN", "DET", "IND", "HOU", "LAR", "LAC", "ARI", "LV", "DAL"}

def get_stadium_weather(home_team: str, game_date: str) -> Optional[dict]:
    if home_team in INDOOR_TEAMS:
        return {"temperature": 72, "wind_speed": 0, "is_outdoor": False}
    coords = OUTDOOR_STADIUM_COORDS.get(home_team)
    if not coords:
        return None
    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,wind_speed_10m_max"
        f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
        f"&start_date={game_date}&end_date={game_date}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        temp = daily.get("temperature_2m_max", [None])[0]
        wind = daily.get("wind_speed_10m_max", [None])[0]
        return {"temperature": temp, "wind_speed": wind, "is_outdoor": True}
    except Exception:
        return None

def estimate_weather_impact(is_outdoor: bool, temperature: Optional[float],
                              wind_speed: Optional[float]) -> float:
    if not is_outdoor:
        return 0.0
    impact = 0.0
    if temperature is not None and temperature < 32:
        impact -= 0.02
    if temperature is not None and temperature < 20:
        impact -= 0.02
    if wind_speed is not None and wind_speed > 20:
        impact -= 0.015
    if wind_speed is not None and wind_speed > 30:
        impact -= 0.015
    return impact
