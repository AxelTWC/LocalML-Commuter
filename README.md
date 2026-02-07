# LocalML-Commuter

A lightweight CLI that ingests real-world commute signals and uses machine learning to predict optimal transportation choices.

## Overview

**LocalML-Commuter** combines:
- **Real-time commute signals**: Weather APIs, transit ETAs, ride-hailing pricing
- **Feature engineering**: Extracts patterns from weather, timing, and transportation options
- **TinyNet model**: A lightweight neural network (16 hidden units) trained to predict Uber vs TTC decisions
- **Data logging pipeline**: Continuously grows dataset with personal outcomes + external API data

## Key Features

[+] **Real-world data integration**:
  - Weather from OpenWeatherMap API (free tier)
  - Transit ETA estimation using heuristics + optional Google Maps API
  - Uber pricing model based on distance and surge pricing

[+] **Feature extraction**:
  - Weather: temperature, precipitation, wind speed, humidity
  - Timing: weekday/weekend, hour of day (rush hour detection)
  - Transit: ETA, number of transfers, walking distance
  - Pricing: Uber fare estimate with demand multiplier

[+] **Lightweight ML model**:
  - TinyNet: Linear → ReLU → Linear (16 hidden units)
  - 80%+ reliability target on validation data
  - GPU-optional (CPU fallback)
  - Standardized input features for stability

[+] **Continuous dataset growth**:
  - CLI automatically logs real-world outcomes
  - Pairs user decisions with captured environmental conditions
  - Enable iterative retraining for model improvement

## Installation

```bash
cd LocalML-Commuter

# Install dependencies
pip install -r requirements.txt
# Or manually:
# pip install typer pandas torch requests

# Set up environment variables for APIs
export OPENWEATHER_API_KEY="your_key_here"
export GOOGLE_MAPS_API_KEY="your_key_here"  # optional
```

### Getting API Keys

**OpenWeatherMap** (required, free tier available):
1. Visit https://openweathermap.org/api
2. Sign up for free account
3. Generate API key in account settings
4. Set: `export OPENWEATHER_API_KEY="your_key"`

**Google Maps** (optional, for better transit estimates):
1. Visit https://console.cloud.google.com/
2. Enable "Distance Matrix API"
3. Create API key
4. Set: `export GOOGLE_MAPS_API_KEY="your_key"`

## CLI Commands

### 1. **fetch** — Log a trip with real-time data

Fetches live weather, transit times, and pricing, then logs your choice:

```bash
python -m commute.cli fetch --choice=uber --annoyance=3

# Custom origin/destination:
python -m commute.cli fetch \
  --origin-lat=43.6634 \
  --origin-lon=-79.4500 \
  --dest-lat=43.6629 \
  --dest-lon=-79.3957 \
  --choice=ttc \
  --annoyance=2
```

**Output**:
```
[*] Fetching real-world commute data...
  [+] Weather: 5°C, 2.3mm rain
  [+] TTC: 28 min, 2 transfers
  [+] Uber: 18 min, $12.50 CAD

Trip Summary:
  Weather: 5°C, 2.3mm rain, 15.4 km/h wind
  TTC: 28 min (2 transfers, 3 min walk)
  Uber: 18 min, $12.50 CAD
  Your choice: TTC
  Annoyance rating: 2/5
```

### 2. **estimate** — View commute options without logging

Quick view of all transportation options (weather, transit, pricing):

```bash
python -m commute.cli estimate

# Custom location:
python -m commute.cli estimate \
  --origin-lat=43.6700 --origin-lon=-79.3950 \
  --dest-lat=43.6450 --dest-lon=-79.3900
```

**Output**:
```
============================================================
COMMUTE OPTIONS
============================================================

Current conditions:
   Temperature: 8°C
   Precipitation: 0.0mm
   Wind: 12.3 km/h

TTC Option:
   ETA: 25 min | Transfers: 2 | Walking: 2.5 min

Uber Option:
   ETA: 15 min | Cost: $11.75 CAD

Uber is affordable today!
============================================================
```

### 3. **predict** — Get model recommendation using real data

Uses your trained model to predict best option:

```bash
python -m commute.cli predict

# Custom location:
python -m commute.cli predict \
  --origin-lat=43.6634 --origin-lon=-79.4500 \
  --dest-lat=43.6629 --dest-lon=-79.3957
```

**Output**:
```
Using real-time data for prediction...

[*] Fetching real-world commute data...
  [+] Weather: 3°C, 0mm rain
  [+] TTC: 32 min, 2 transfers
  [+] Uber: 20 min, $14.25 CAD

Recommendation: UBER  (confidence ~ 85%)
Top drivers (rough):
- uber_price_cad is high/low vs your usual
- ttc_transfers is high/low vs your usual
- hour is high/low vs your usual
```

### 4. **train** — Train the TinyNet model

Trains on logged trips to predict your preferences:

```bash
# Train on CPU
python -m commute.cli train

# Use GPU if available
python -m commute.cli train --device=cuda

# Force CPU
python -m commute.cli train --device=cpu
```

Saves to `data/model/` with:
- `model.pt`: Neural network weights + preprocessing stats
- `run_metadata.json`: Training metadata (device, row count, etc.)

## Data Pipeline

### Features Used (10 input features):

1. **precip_mm** — Precipitation in mm
2. **temp_c** — Temperature in Celsius
3. **wind_kph** — Wind speed in km/h
4. **is_weekday** — Binary (1=weekday, 0=weekend)
5. **hour** — Hour of day (0-23)
6. **ttc_eta_min** — Estimated transit time in minutes
7. **ttc_transfers** — Number of transfers needed
8. **ttc_walk_min** — Walking distance in minutes
9. **uber_eta_min** — Estimated Uber ETA in minutes
10. **uber_price_cad** — Estimated fare in CAD

### Data File Structure

`data/trips.csv`:
```csv
timestamp,precip_mm,temp_c,wind_kph,is_weekday,hour,ttc_eta_min,ttc_transfers,ttc_walk_min,humidity,uber_eta_min,uber_price_cad,choice,annoyance
2026-02-07T08:30:00,0.0,5.2,12.3,1,8,28,2,3.0,55.0,18,12.50,1,3
2026-02-07T17:45:00,2.1,3.8,15.0,1,17,35,3,4.5,62.0,22,15.75,0,2
```

- **choice**: 1=Uber, 0=TTC
- **annoyance**: User rating 1-5 (optional feedback)

## Model Architecture

**TinyNet**:
```
Input (10 features)
  ↓
Linear(10 → 16)
  ↓
ReLU activation
  ↓
Linear(16 → 1)
  ↓
Sigmoid (via BCEWithLogitsLoss)
  ↓
Output: P(Uber choice)
```

**Training**:
- Loss function: Binary Cross-Entropy with Logits
- Optimizer: Adam (lr=0.01)
- Epochs: 200
- Preprocessing: Z-score standardization

## Workflow Example

```bash
# 1. Set up API key
export OPENWEATHER_API_KEY="your_key"

# 2. Log first commute (fetches real data automatically)
python -m commute.cli fetch --choice=uber --annoyance=4

# 3. Log more trips over time
python -m commute.cli fetch --choice=ttc --annoyance=2
python -m commute.cli fetch --choice=uber --annoyance=3
# ... continue logging ...

# 4. Once you have 10+ trips, train the model
python -m commute.cli train

# 5. Get recommendations on new days
python -m commute.cli predict

# 6. Continue logging outcomes (model improves!)
python -m commute.cli fetch --choice=uber --annoyance=2
```

## API Data Sources

| Data | Source | Free Tier | Coverage |
|------|--------|-----------|----------|
| Weather | OpenWeatherMap | [+] Yes (1000 calls/day) | Global |
| Transit ETA | Google Maps (optional) | [+] Limited (25,000 calls/month) | Global |
| Transit ETA (fallback) | Heuristic estimation | [+] Always | Toronto |
| Uber pricing | Distance-based model | [+] Yes | Toronto area |

## Continuous Improvement

Each logged trip improves the model:

1. **Collect**: `fetch` command captures real conditions + your choice
2. **Accumulate**: Data saved to `trips.csv` with timestamp
3. **Retrain**: `train` command updates weights based on new data
4. **Improve**: Next `predict` uses latest patterns

Track improvement via annoyance ratings — are recommendations getting better?

## Project Structure

```
LocalML-Commuter/
├── commute/
│   ├── __init__.py
│   ├── cli.py          # CLI commands (fetch, estimate, predict, train)
│   ├── data.py         # API integrations & feature engineering
│   └── model.py        # TinyNet architecture & training
├── data/
│   ├── trips.csv       # Logged commute data
│   └── model/
│       ├── model.pt    # Trained weights + preprocessing stats
│       └── run_metadata.json  # Training metadata
└── README.md
```

## Performance Targets

- **Model reliability**: 80%+ confidence on validation split
- **Feature coverage**: 10 engineered features from weather + timing + pricing
- **Latency**: <2 seconds for API fetching + prediction
- **Data growth**: Continuous via CLI logging system

## Troubleshooting

**"OPENWEATHER_API_KEY not set"**
```bash
export OPENWEATHER_API_KEY="your_api_key"
# Or run with env var inline:
OPENWEATHER_API_KEY="key" python -m commute.cli fetch --choice=uber
```

**"Failed to fetch weather"**
- Check internet connection
- Verify API key is valid
- Check OpenWeatherMap rate limits

**"CUDA not available, using CPU"**
- This is normal if no GPU detected
- Model trains fine on CPU (still fast due to small size)
- No action needed unless you want GPU acceleration

**"You have fewer than 10 rows"**
- Log at least 10 trips before training
- Use `fetch` command multiple times

## Future Enhancements

- [ ] Support multiple routes/destinations
- [ ] Weather extremes alerting (snow, storms)
- [ ] Historical pattern analysis (best days to take Uber)
- [ ] Integration with calendar (meeting times)
- [ ] Cost tracking over time
- [ ] Export reports

## License

MIT (use freely!)
