"""
Real-world data ingestion from open-source APIs.
Fetches weather, transit ETA, and pricing data for commute decision modeling.
"""
from __future__ import annotations

import os
import json
import datetime as dt
from typing import Optional
from pathlib import Path
import math

import requests
import pandas as pd


# ============================================================================
# WEATHER API - OpenWeatherMap (free tier)
# Sign up at https://openweathermap.org/api
# ============================================================================

def fetch_weather(
    latitude: float = 43.6634,  # M9A 0C9, Toronto default
    longitude: float = -79.4500,
    api_key: Optional[str] = None,
) -> dict[str, float]:
    """
    Fetch current weather from OpenWeatherMap free API.
    
    Args:
        latitude: Location latitude (default: Toronto)
        longitude: Location longitude (default: Toronto)
        api_key: OpenWeatherMap API key (or use OPENWEATHER_API_KEY env var)
    
    Returns:
        dict with keys: temp_c, precip_mm, wind_kph, humidity
    """
    api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENWEATHER_API_KEY not set. "
            "Get free key at https://openweathermap.org/api"
        )

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": api_key,
        "units": "metric",
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        weather = {
            "temp_c": float(data["main"]["temp"]),
            "precip_mm": float(data.get("rain", {}).get("1h", 0.0)),
            "wind_kph": float(data["wind"]["speed"]) * 3.6,  # m/s to km/h
            "humidity": float(data["main"]["humidity"]),
        }
        return weather
    except Exception as e:
        raise RuntimeError(f"Failed to fetch weather: {e}")


# ============================================================================
# TRANSIT ETA - Google Maps Distance Matrix API (limited free tier)
# Alternative: Use static typical times + day/time estimation
# ============================================================================

def estimate_ttc_eta(
    origin: tuple[float, float] = (43.6634, -79.4500),  # M9A 0C9 default
    destination: tuple[float, float] = (43.6629, -79.3957),  # Myhal Building default
    departure_time: Optional[dt.datetime] = None,
    api_key: Optional[str] = None,
) -> dict[str, float]:
    """
    Estimate TTC (Toronto Transit) ETA and transfers.
    Falls back to estimation if no API key provided.
    
    Args:
        origin: (lat, lon) tuple
        destination: (lat, lon) tuple
        departure_time: when departing (default: now)
        api_key: Google Maps API key (optional, GOOGLE_MAPS_API_KEY env var)
    
    Returns:
        dict with keys: ttc_eta_min, ttc_transfers, ttc_walk_min
    """
    if not departure_time:
        departure_time = dt.datetime.now()

    api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")

    # Try Google Maps first if key available
    if api_key:
        try:
            return _fetch_ttc_from_google(
                origin, destination, departure_time, api_key
            )
        except Exception as e:
            print(f"[WARNING] Google Maps API failed ({e}), falling back to estimation")

    # Fallback: Estimate based on distance + time of day
    return _estimate_ttc_heuristic(origin, destination, departure_time)


def _fetch_ttc_from_google(
    origin: tuple[float, float],
    destination: tuple[float, float],
    departure_time: dt.datetime,
    api_key: str,
) -> dict[str, float]:
    """Query Google Maps Distance Matrix API for transit times."""
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": f"{origin[0]},{origin[1]}",
        "destinations": f"{destination[0]},{destination[1]}",
        "mode": "transit",
        "key": api_key,
        "departure_time": int(departure_time.timestamp()),
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data["status"] != "OK":
        raise RuntimeError(f"Google Maps API error: {data.get('error_message')}")

    element = data["rows"][0]["elements"][0]
    if element["status"] != "OK":
        raise RuntimeError(f"Route not found: {element.get('status')}")

    duration_sec = element["duration"]["value"]
    distance_m = element["distance"]["value"]

    # Rough estimation: 1 transfer per 15 minutes
    ttc_transfers = max(0, int(duration_sec / 900))
    # Estimate walking as 20% of total time
    ttc_walk_min = (duration_sec / 60) * 0.2

    return {
        "ttc_eta_min": duration_sec / 60,
        "ttc_transfers": ttc_transfers,
        "ttc_walk_min": ttc_walk_min,
    }


def _estimate_ttc_heuristic(
    origin: tuple[float, float],
    destination: tuple[float, float],
    departure_time: dt.datetime,
) -> dict[str, float]:
    """
    Estimate TTC metrics based on straight-line distance + time of day.
    Toronto-specific heuristics.
    """
    # Haversine distance in km
    lat1, lon1 = origin
    lat2, lon2 = destination
    R = 6371  # Earth radius in km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    distance_km = R * 2 * math.asin(math.sqrt(a))

    # Base: 1.2 km per 5 min on TTC average
    base_eta = (distance_km / 1.2) * 5

    # Rush hour multiplier (7-9am, 4-7pm weekdays)
    hour = departure_time.hour
    is_weekday = departure_time.weekday() < 5
    if is_weekday and (7 <= hour <= 9 or 16 <= hour <= 19):
        base_eta *= 1.4

    # Estimate transfers: 1 per ~2.5km
    ttc_transfers = max(0, int(distance_km / 2.5))
    
    # Walking: ~10% of total time
    ttc_walk_min = base_eta * 0.1

    return {
        "ttc_eta_min": base_eta,
        "ttc_transfers": ttc_transfers,
        "ttc_walk_min": ttc_walk_min,
    }


# ============================================================================
# RIDE-HAILING PRICING - Using distance-based model (Uber API deprecated)
# We use straight-line distance with Toronto-specific pricing multipliers
# ============================================================================

def estimate_uber_price(
    origin: tuple[float, float] = (43.6634, -79.4500),  # M9A 0C9 default
    destination: tuple[float, float] = (43.6629, -79.3957),  # Myhal Building default
    departure_time: Optional[dt.datetime] = None,
    demand_multiplier: float = 1.0,
) -> dict[str, float]:
    """
    Estimate Uber price based on distance and demand.
    Uses Toronto-specific pricing model.
    
    Args:
        origin: (lat, lon) tuple
        destination: (lat, lon) tuple
        departure_time: for surge pricing estimation
        demand_multiplier: surge pricing multiplier (1.0 = normal)
    
    Returns:
        dict with keys: uber_eta_min, uber_price_cad
    """
    if not departure_time:
        departure_time = dt.datetime.now()

    # Calculate distance
    lat1, lon1 = origin
    lat2, lon2 = destination
    R = 6371

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    distance_km = R * 2 * math.asin(math.sqrt(a))

    # Toronto UberX pricing model (approximate):
    # Base: $3.15 CAD
    # Per km: $2.00 CAD
    # Per minute: $0.45 CAD
    base_fare = 3.15
    per_km_rate = 2.00
    per_min_rate = 0.45
    service_fee = 0.50  # Service fee

    # ETA: ~4 min per km in Toronto
    eta_min = distance_km * 4 + 2  # +2 for pickup

    distance_cost = distance_km * per_km_rate
    time_cost = eta_min * per_min_rate
    subtotal = base_fare + distance_cost + time_cost + service_fee

    # Apply demand multiplier (surge pricing)
    total_cad = subtotal * demand_multiplier

    return {
        "uber_eta_min": eta_min,
        "uber_price_cad": round(total_cad, 2),
    }


def estimate_demand_multiplier(
    departure_time: Optional[dt.datetime] = None,
) -> float:
    """
    Estimate demand/surge multiplier based on time.
    Higher during rush hours.
    """
    if not departure_time:
        departure_time = dt.datetime.now()

    hour = departure_time.hour
    is_weekday = departure_time.weekday() < 5

    # Rush hour surge: 1.3-1.5x
    if is_weekday:
        if 7 <= hour <= 9:  # Morning rush
            return 1.3
        if 16 <= hour <= 19:  # Evening rush
            return 1.5
    
    # Late night surge: 1.2x (11pm-6am)
    if 23 <= hour or hour <= 6:
        return 1.2

    # Normal demand
    return 1.0


# ============================================================================
# COMBINED DATA FETCHING
# ============================================================================

def fetch_commute_data(
    origin: tuple[float, float] = (43.6634, -79.4500),  # M9A 0C9
    destination: tuple[float, float] = (43.6629, -79.3957),  # Myhal Building
    departure_time: Optional[dt.datetime] = None,
    openweather_key: Optional[str] = None,
    google_maps_key: Optional[str] = None,
) -> dict:
    """
    Fetch all commute data from APIs and estimation models.
    
    Returns:
        dict with all features ready for model prediction
    """
    if not departure_time:
        departure_time = dt.datetime.now()

    print("[*] Fetching real-world commute data...")

    # Weather
    try:
        weather = fetch_weather(origin[0], origin[1], openweather_key)
        print(f"  [+] Weather: {weather['temp_c']}°C, {weather['precip_mm']}mm rain")
    except Exception as e:
        print(f"  [!] Weather failed: {e}")
        weather = {
            "temp_c": 15.0,
            "precip_mm": 0.0,
            "humidity": 50.0,
            "wind_kph": 10.0,
        }

    # Transit ETA
    try:
        ttc = estimate_ttc_eta(origin, destination, departure_time, google_maps_key)
        print(f"  [+] TTC: {ttc['ttc_eta_min']:.0f}min, "
              f"{ttc['ttc_transfers']} transfers")
    except Exception as e:
        print(f"  [!] TTC estimation failed: {e}")
        ttc = {
            "ttc_eta_min": 25.0,
            "ttc_transfers": 1,
            "ttc_walk_min": 5.0,
        }

    # Uber pricing & ETA
    demand = estimate_demand_multiplier(departure_time)
    uber = estimate_uber_price(origin, destination, departure_time, demand)
    print(f"  [+] Uber: {uber['uber_eta_min']:.0f}min, ${uber['uber_price_cad']:.2f} CAD")

    # Combine all data
    data = {
        "timestamp": departure_time.isoformat(timespec="seconds"),
        "is_weekday": int(departure_time.weekday() < 5),
        "hour": departure_time.hour,
        **weather,
        **ttc,
        **uber,
    }

    return data


def save_commute_log(
    data: dict,
    choice: str,
    annoyance: int = 3,
    log_path: Path = Path("data/trips.csv"),
) -> None:
    """
    Save logged commute to CSV with timestamp.
    
    Args:
        data: dict from fetch_commute_data()
        choice: "ttc" or "uber"
        annoyance: 1-5 rating
        log_path: where to save CSV
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    choice_normalized = choice.strip().lower()
    if choice_normalized not in {"ttc", "uber"}:
        raise ValueError("choice must be 'ttc' or 'uber'")

    row = {
        **data,
        "choice": 1 if choice_normalized == "uber" else 0,
        "annoyance": int(annoyance),
    }

    # Reorder columns to match FEATURES in cli.py
    feature_order = [
        "timestamp",
        "precip_mm",
        "temp_c",
        "wind_kph",
        "is_weekday",
        "hour",
        "ttc_eta_min",
        "ttc_transfers",
        "ttc_walk_min",
        "humidity",
        "uber_eta_min",
        "uber_price_cad",
        "choice",
        "annoyance",
    ]
    row = {k: row.get(k, 0) for k in feature_order}

    if log_path.exists():
        df = pd.read_csv(log_path)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(log_path, index=False)

    print(f"[+] Logged {choice.upper()} trip (annoyance={annoyance})")
