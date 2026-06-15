"""
Open-Meteo historical weather API wrapper.

Endpoint: https://archive-api.open-meteo.com/v1/archive
No API key required. Responses cached to data/cache/weather/ as JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "weather"
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily variables fetched per request
_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_max",
]

# Hourly variables (fetched for afternoon kick-off window, 14:00-18:00 local)
_HOURLY_VARS = [
    "temperature_2m",
    "relativehumidity_2m",
    "windspeed_10m",
    "precipitation",
]


def _cache_key(lat: float, lon: float, date: str) -> str:
    return f"{lat:.3f}_{lon:.3f}_{date}.json"


def _load_cache(key: str) -> dict | None:
    path = CACHE_DIR / key
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_cache(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / key).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def get_match_weather(
    lat: float,
    lon: float,
    date: str,
    timeout: int = 30,
    retries: int = 3,
) -> dict:
    """
    Return weather conditions for a given venue and match date.

    Parameters
    ----------
    lat, lon : venue coordinates
    date     : ISO date string "YYYY-MM-DD"

    Returns
    -------
    dict with keys:
      temp_max_c, temp_min_c, temp_avg_c,
      humidity_max_pct, precip_mm, wind_max_kmh,
      afternoon_temp_c, afternoon_humidity_pct   (avg 14-18 local hour)
    """
    key = _cache_key(lat, lon, date)
    cached = _load_cache(key)
    if cached:
        return cached

    # Open-Meteo archive only has data up to ~5 days ago; skip API for future dates
    from datetime import date as _date, datetime
    try:
        match_date = datetime.strptime(date, "%Y-%m-%d").date()
        if match_date > _date.today():
            return _fallback_weather(lat)
    except ValueError:
        pass

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": date,
        "end_date":   date,
        "daily":      ",".join(_DAILY_VARS),
        "hourly":     ",".join(_HOURLY_VARS),
        "timezone":   "auto",
    }

    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=params, timeout=timeout)
            r.raise_for_status()
            raw = r.json()
            break
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  [weather] Failed for ({lat},{lon}) {date}: {exc}")
                return _fallback_weather(lat)
            time.sleep(2 ** attempt)

    daily = raw.get("daily", {})
    hourly = raw.get("hourly", {})

    # Afternoon window: hours 14-18 (indices depend on hourly resolution)
    # Open-Meteo returns 24 hourly values; index 14-18 = 14:00-18:00 local
    def _hourly_avg(var: str, start_h: int = 14, end_h: int = 18) -> float:
        vals = hourly.get(var, [None] * 24)
        window = [v for v in vals[start_h:end_h] if v is not None]
        return round(sum(window) / len(window), 1) if window else 20.0

    def _first(key: str, default=0.0) -> float:
        vals = daily.get(key, [default])
        v = vals[0] if vals else default
        return float(v) if v is not None else default

    t_max = _first("temperature_2m_max", 22.0)
    t_min = _first("temperature_2m_min", 14.0)

    result = {
        "temp_max_c":          round(t_max, 1),
        "temp_min_c":          round(t_min, 1),
        "temp_avg_c":          round((t_max + t_min) / 2, 1),
        "humidity_max_pct":    round(_first("relative_humidity_2m_max", 60.0), 1),
        "precip_mm":           round(_first("precipitation_sum", 0.0), 2),
        "wind_max_kmh":        round(_first("windspeed_10m_max", 15.0), 1),
        "afternoon_temp_c":    _hourly_avg("temperature_2m"),
        "afternoon_humidity_pct": _hourly_avg("relativehumidity_2m"),
    }

    _save_cache(key, result)
    return result


def _fallback_weather(lat: float) -> dict:
    """Return reasonable defaults when API is unavailable."""
    # Rough June average by latitude band
    if lat > 45:    # Canada/Northern US
        temp = 20.0; hum = 65.0
    elif lat > 35:  # Most of US, Europe
        temp = 27.0; hum = 60.0
    elif lat > 20:  # Mexico, Southern US, North Africa
        temp = 31.0; hum = 55.0
    else:            # Tropical
        temp = 30.0; hum = 75.0
    return {
        "temp_max_c": temp + 3, "temp_min_c": temp - 5, "temp_avg_c": temp,
        "humidity_max_pct": hum, "precip_mm": 2.0, "wind_max_kmh": 15.0,
        "afternoon_temp_c": temp, "afternoon_humidity_pct": hum,
    }
