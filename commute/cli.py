from __future__ import annotations 
import datetime as dt
from pathlib import Path
import typer
import pandas as pd

from .model import train_model, predict_choice
from .data import (
    fetch_commute_data,
    save_commute_log,
    fetch_weather,
    estimate_ttc_eta,
    estimate_uber_price,
    estimate_demand_multiplier,
)

app = typer.Typer(no_args_is_help=True)
DATA_PATH = Path("data/trips.csv")
MODEL_DIR = Path("data/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "precip_mm", "temp_c", "wind_kph", "is_weekday", "hour",
    "ttc_eta_min", "ttc_transfers", "ttc_walk_min",
    "uber_eta_min", "uber_price_cad",
]

@app.command()
def fetch(
    origin_lat: float = typer.Option(43.6634, help="Origin latitude (default: M9A 0C9)"),
    origin_lon: float = typer.Option(-79.4500, help="Origin longitude"),
    dest_lat: float = typer.Option(43.6629, help="Destination latitude (default: Myhal Building)"),
    dest_lon: float = typer.Option(-79.3957, help="Destination longitude"),
    choice: str = typer.Option(..., help="ttc or uber"),
    annoyance: int = typer.Option(3, help="1-5 rating of how annoying the trip felt"),
):
    """
    Fetch real-world commute data from APIs and log your choice.
    
    Requires environment variables:
    - OPENWEATHER_API_KEY (get free at https://openweathermap.org/api)
    - GOOGLE_MAPS_API_KEY (optional, for better transit estimates)
    
    Example:
        python -m commute.cli fetch --choice=uber
    """
    try:
        origin = (origin_lat, origin_lon)
        destination = (dest_lat, dest_lon)
        
        # Fetch all real data from APIs
        data = fetch_commute_data(origin, destination)
        
        # Save to CSV
        save_commute_log(data, choice, annoyance, DATA_PATH)
        
        typer.echo("\n" + "="*60)
        typer.echo(f"Trip Summary:")
        typer.echo(f"  Weather: {data['temp_c']}°C, {data['precip_mm']:.1f}mm rain, {data['wind_kph']:.1f} km/h wind")
        typer.echo(f"  TTC: {data['ttc_eta_min']:.0f} min ({int(data['ttc_transfers'])} transfers, {data['ttc_walk_min']:.0f} min walk)")
        typer.echo(f"  Uber: {data['uber_eta_min']:.0f} min, ${data['uber_price_cad']:.2f} CAD")
        typer.echo(f"  Your choice: {choice.upper()}")
        typer.echo(f"  Annoyance rating: {annoyance}/5")
        typer.echo("="*60 + "\n")
        
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def log(
    precip_mm: float = typer.Option(...),
    temp_c: float = typer.Option(...),
    wind_kph: float = typer.Option(...),
    ttc_eta_min: float = typer.Option(...),
    ttc_transfers: int = typer.Option(...),
    ttc_walk_min: float = typer.Option(...),
    uber_eta_min: float = typer.Option(...),
    uber_price_cad: float = typer.Option(...),
    choice: str = typer.Option(..., help="ttc or uber"),
    annoyance: int = typer.Option(3, help="1-5 rating of how annoying the trip felt"),
):
    choice = choice.strip().lower()
    if choice not in {"ttc", "uber"}:
        raise typer.BadParameter("choice must be 'ttc' or 'uber'")

    now = dt.datetime.now()
    row = {
        "timestamp": now.isoformat(timespec="seconds"),
        "precip_mm": precip_mm,
        "temp_c": temp_c,
        "wind_kph": wind_kph,
        "is_weekday": int(now.weekday() < 5),
        "hour": now.hour,
        "ttc_eta_min": ttc_eta_min,
        "ttc_transfers": int(ttc_transfers),
        "ttc_walk_min": ttc_walk_min,
        "uber_eta_min": uber_eta_min,
        "uber_price_cad": uber_price_cad,
        "choice": 1 if choice == "uber" else 0,
        "annoyance": int(annoyance),
    }

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Create it with the header row first.")

    df = pd.read_csv(DATA_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    typer.echo(f"Logged trip at {row['timestamp']} (choice={choice.upper()})")

@app.command()
def estimate(
    origin_lat: float = typer.Option(43.6634, help="Origin latitude (default: M9A 0C9)"),
    origin_lon: float = typer.Option(-79.4500, help="Origin longitude"),
    dest_lat: float = typer.Option(43.6629, help="Destination latitude (default: Myhal Building)"),
    dest_lon: float = typer.Option(-79.3957, help="Destination longitude"),
):
    """
    Get real-time commute estimates (weather, transit, pricing) without logging.
    
    Example:
        python -m commute.cli estimate
    """
    try:
        origin = (origin_lat, origin_lon)
        destination = (dest_lat, dest_lon)
        
        data = fetch_commute_data(origin, destination)
        
        typer.echo("\n" + "="*60)
        typer.echo("COMMUTE OPTIONS")
        typer.echo("="*60)
        typer.echo(f"\nCurrent conditions:")
        typer.echo(f"   Temperature: {data['temp_c']}°C")
        typer.echo(f"   Precipitation: {data['precip_mm']:.1f}mm")
        typer.echo(f"   Wind: {data['wind_kph']:.1f} km/h")
        
        ttc_info = f"   ETA: {data['ttc_eta_min']:.0f} min | Transfers: {int(data['ttc_transfers'])} | Walking: {data['ttc_walk_min']:.0f} min"
        uber_info = f"   ETA: {data['uber_eta_min']:.0f} min | Cost: ${data['uber_price_cad']:.2f} CAD"
        
        typer.echo(f"\nTTC Option:")
        typer.echo(ttc_info)
        
        typer.echo(f"\nUber Option:")
        typer.echo(uber_info)
        
        # Quick recommendation based on price
        if data['uber_price_cad'] < 15:
            typer.echo("\nUber is affordable today!")
        elif data['ttc_eta_min'] < 20:
            typer.echo("\nTTC is fast today!")
        
        typer.echo("="*60 + "\n")
        
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def train(device: str = typer.Option("auto", help="auto|cpu|cuda")):
    """
    Train a small neural net to predict Uber(1) vs TTC(0).
    Saves model to data/model/
    """
    df = pd.read_csv(DATA_PATH)
    if len(df) < 10:
        typer.echo("You have fewer than 10 rows. Log more trips first for a meaningful model.")
        raise typer.Exit(code=1)

    train_model(df, FEATURES, MODEL_DIR, device=device)
    typer.echo("Training complete. Model saved in data/model/")

@app.command()
def predict(
    origin_lat: float = typer.Option(43.6634, help="Origin latitude (default: M9A 0C9)"),
    origin_lon: float = typer.Option(-79.4500, help="Origin longitude"),
    dest_lat: float = typer.Option(43.6629, help="Destination latitude (default: Myhal Building)"),
    dest_lon: float = typer.Option(-79.3957, help="Destination longitude"),
):
    """
    Predict whether you'll prefer TTC or Uber using your trained model.
    Fetches real-time data from APIs (weather, transit, pricing).
    
    Example:
        python -m commute.cli predict
    """
    try:
        origin = (origin_lat, origin_lon)
        destination = (dest_lat, dest_lon)
        
        typer.echo("Using real-time data for prediction...\n")
        data = fetch_commute_data(origin, destination)
        
        x = {
            "precip_mm": data["precip_mm"],
            "temp_c": data["temp_c"],
            "wind_kph": data["wind_kph"],
            "is_weekday": data["is_weekday"],
            "hour": data["hour"],
            "ttc_eta_min": data["ttc_eta_min"],
            "ttc_transfers": int(data["ttc_transfers"]),
            "ttc_walk_min": data["ttc_walk_min"],
            "uber_eta_min": data["uber_eta_min"],
            "uber_price_cad": data["uber_price_cad"],
        }

        result = predict_choice(x, FEATURES, MODEL_DIR)
        typer.echo("\n" + result)
        
    except Exception as e:
        typer.echo(f"[ERROR] Prediction failed: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()