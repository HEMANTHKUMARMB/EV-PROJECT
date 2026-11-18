"""Generate a simulated EV trip dataset from vehicle physics.

Target: consumption in kWh/100 km.
Inputs (7): avg_speed, trip_distance, temperature, mass, elevation_change,
            driving_style, ac_on.  (battery capacity is used only for range.)
"""
import numpy as np
import pandas as pd

RHO, G = 1.2, 9.81          # air density, gravity
CD_A = 0.62                  # drag coefficient x frontal area (m^2)
C_RR = 0.010                 # rolling resistance
DRIVE_EFF = 0.88             # battery -> wheel efficiency
REGEN = 0.60                 # share of downhill energy recovered
STYLE_FACTOR = {"Eco": 0.92, "Normal": 1.00, "Aggressive": 1.18}


def consumption_kwh_per_100km(speed, dist, temp, mass, elev, style_factor, ac, rng=None):
    v = speed / 3.6                                   # m/s
    f_aero = 0.5 * RHO * CD_A * v ** 2
    f_roll = C_RR * mass * G
    wheel_kwh_km = (f_aero + f_roll) * 1000 / 3.6e6   # kWh per km at wheels

    # stop-and-go: low average speed means more accel/brake cycles
    stop_go = 0.012 * np.clip(70 - speed, 0, None) / 45 * (mass / 1800)
    wheel_kwh_km = (wheel_kwh_km + stop_go) * style_factor

    # grade: climbing costs energy, descending returns some via regen
    grade_kwh_km = mass * G * (elev / (dist * 1000)) * 1000 / 3.6e6
    grade_kwh_km = np.where(grade_kwh_km >= 0, grade_kwh_km, grade_kwh_km * REGEN)

    # cold batteries are less efficient
    cold_penalty = 1 + 0.012 * np.clip(15 - temp, 0, None)
    batt_kwh_km = (wheel_kwh_km / DRIVE_EFF) * cold_penalty + grade_kwh_km / DRIVE_EFF

    # climate control draw (kW) x trip time per km
    aux_kw = np.where(ac == 1, 0.4 + 0.13 * np.abs(temp - 22), 0.25)
    aux_kwh_km = aux_kw / speed

    total = (batt_kwh_km + aux_kwh_km) * 100
    if rng is not None:
        total = total * rng.lognormal(0, 0.03, size=np.shape(total))
    return np.clip(total, 4.0, None)


def generate(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    speed = np.clip(rng.normal(65, 25, n), 25, 120)
    dist = np.clip(rng.gamma(2.2, 22, n), 5, 150)
    temp = np.clip(rng.normal(24, 8, n), -5, 42)
    mass = np.clip(rng.normal(1800, 180, n), 1400, 2200)
    elev = np.clip(rng.normal(0, 150, n), -500, 500)
    elev = np.clip(elev, -40 * dist, 40 * dist)   # keep grade within +/-4%
    style = rng.choice(list(STYLE_FACTOR), n, p=[0.25, 0.55, 0.20])
    # A/C or heater is far more likely when it is hot or cold
    p_ac = np.clip(0.25 + 0.035 * np.abs(temp - 22), 0.2, 0.95)
    ac = (rng.random(n) < p_ac).astype(int)
    battery = rng.choice([40, 50, 60, 75, 82], n)

    y = consumption_kwh_per_100km(speed, dist, temp, mass, elev,
                                  np.array([STYLE_FACTOR[s] for s in style]), ac, rng)
    df = pd.DataFrame({
        "avg_speed_kmh": speed.round(1), "trip_distance_km": dist.round(1),
        "temperature_c": temp.round(1), "mass_kg": mass.round(0),
        "elevation_change_m": elev.round(0), "driving_style": style,
        "ac_on": ac, "battery_kwh": battery,
        "consumption_kwh_100km": y.round(2),
    })
    df["trip_energy_kwh"] = (df.consumption_kwh_100km * df.trip_distance_km / 100).round(2)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/ev_trips.csv", index=False)
    print(df.shape); print(df.describe().round(2).T)
