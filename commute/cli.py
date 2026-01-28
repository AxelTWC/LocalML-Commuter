from __future__ import annotations 
import datetime as dt
from pathlib import Path
import typer
import pandas as pd

from .model import train_model, predict_choice

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
    precip_mm: float = typer.Option(...),
    temp_c: float = typer.Option(...),
    wind_kph: float = typer.Option(...),
    ttc_eta_min: float = typer.Option(...),
    ttc_transfers: int = typer.Option(...),
    ttc_walk_min: float = typer.Option(...),
    uber_eta_min: float = typer.Option(...),
    uber_price_cad: float = typer.Option(...),
):
    """
    Predict whether you'll prefer TTC or Uber for this situation.
    """
    now = dt.datetime.now()
    x = {
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
    }

    result = predict_choice(x, FEATURES, MODEL_DIR)
    typer.echo(result)

if __name__ == "__main__":
    app()    