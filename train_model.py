"""Train and compare Linear Regression, Random Forest and Gradient Boosting."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["avg_speed_kmh", "trip_distance_km", "temperature_c", "mass_kg",
            "elevation_change_m", "driving_style", "ac_on"]
NUM = [f for f in FEATURES if f != "driving_style"]
TARGET = "consumption_kwh_100km"


def make_pipe(model, scale):
    pre = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUM),
        ("cat", OneHotEncoder(categories=[["Eco", "Normal", "Aggressive"]]), ["driving_style"]),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = pd.read_csv("data/ev_trips.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[TARGET],
                                              test_size=0.2, random_state=42)
    models = {
        "Linear Regression": make_pipe(LinearRegression(), True),
        "Random Forest": make_pipe(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), False),
        "Gradient Boosting": make_pipe(GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                                 learning_rate=0.08, random_state=42), False),
    }
    rows, fitted = [], {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        rows.append({"Model": name, "MAE": round(mean_absolute_error(y_te, p), 3),
                     "RMSE": round(float(np.sqrt(mean_squared_error(y_te, p))), 3),
                     "R2": round(r2_score(y_te, p), 3)})
        fitted[name] = pipe
        print(rows[-1])

    best = min(rows, key=lambda r: r["RMSE"])["Model"]
    pipe = fitted[best]
    # feature importance (grouped back to the 7 original features)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = getattr(pipe.named_steps["model"], "feature_importances_", None)
    grouped = {}
    if imp is not None:
        for n, v in zip(names, imp):
            key = "driving_style" if "driving_style" in n else n.split("__")[1]
            grouped[key] = grouped.get(key, 0) + float(v)
    joblib.dump(pipe, "models/best_model.joblib")
    json.dump({"best_model": best, "n_train": len(X_tr), "n_test": len(X_te),
               "comparison": rows, "feature_importance": grouped},
              open("models/metrics.json", "w"), indent=2)
    print("Best:", best); print(json.dumps(grouped, indent=1))


if __name__ == "__main__":
    main()
