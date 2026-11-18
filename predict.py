"""Load the saved model and predict consumption for one trip."""
import json
from pathlib import Path
import joblib
import pandas as pd

HERE = Path(__file__).parent
_model = None


def _load():
    global _model
    if _model is None:
        _model = joblib.load(HERE / "models" / "best_model.joblib")
    return _model


def predict_trip(speed, distance, temp, mass, elevation, style, ac_on, battery_kwh):
    row = pd.DataFrame([{
        "avg_speed_kmh": speed, "trip_distance_km": distance, "temperature_c": temp,
        "mass_kg": mass, "elevation_change_m": elevation,
        "driving_style": style, "ac_on": int(bool(ac_on)),
    }])
    c = float(_load().predict(row)[0])          # kWh/100 km
    return {"consumption": c, "trip_energy": c * distance / 100,
            "range_km": battery_kwh / c * 100}


def metrics():
    return json.load(open(HERE / "models" / "metrics.json"))


if __name__ == "__main__":
    print(predict_trip(80, 60, 25, 1800, 0, "Normal", True, 60))
