"""Streamlit app: EV Consumption Predictor.   Run: streamlit run app.py"""
import pandas as pd
import streamlit as st
from predict import metrics, predict_trip

st.set_page_config(page_title="EV Consumption Predictor", page_icon="⚡", layout="centered")
st.title("⚡ EV Consumption Predictor")
st.caption("Enter the trip details. The trained model estimates the energy the electric vehicle will use.")

tab_predict, tab_models = st.tabs(["Predict", "Model comparison"])

with tab_predict:
    c1, c2 = st.columns(2)
    speed = c1.slider("Average speed (km/h)", 25, 120, 80)
    dist = c2.slider("Trip distance (km)", 5, 150, 60)
    temp = c1.slider("Outside temperature (°C)", -5, 42, 25)
    mass = c2.slider("Vehicle + load mass (kg)", 1400, 2200, 1800)
    elev = c1.slider("Net elevation change (m)", -500, 500, 0)
    style = c2.selectbox("Driving style", ["Eco", "Normal", "Aggressive"], index=1)
    ac = c1.checkbox("Air-conditioning / heater on", value=True)
    battery = c2.number_input("Battery capacity (kWh)", 20.0, 120.0, 60.0, 1.0)

    if st.button("Predict consumption", type="primary"):
        r = predict_trip(speed, dist, temp, mass, elev, style, ac, battery)
        terrain = "flat road" if elev == 0 else f"{elev:+d} m elevation"
        st.write(f"Trip: {speed} km/h, {dist} km, {temp} °C, {mass} kg, {terrain}, "
                 f"{style} style, A/C {'on' if ac else 'off'}, {battery:g} kWh battery")
        a, b, c = st.columns(3)
        a.metric("Consumption", f"{r['consumption']:.1f} kWh/100 km")
        b.metric("Trip energy", f"{r['trip_energy']:.1f} kWh")
        c.metric("Estimated range", f"{r['range_km']:.0f} km")
        if r["trip_energy"] > battery:
            st.warning("This trip needs more energy than the battery holds. Plan a charge stop.")
        st.success(f"Prediction made by the saved best model ({metrics()['best_model']}).")

with tab_models:
    m = metrics()
    df = pd.DataFrame(m["comparison"]).set_index("Model")
    st.dataframe(df, use_container_width=True)
    st.subheader("RMSE (lower is better)")
    st.bar_chart(df["RMSE"])
    st.write(f"Best model: **{m['best_model']}**, trained on {m['n_train']} trips, tested on {m['n_test']}.")
    st.subheader("Feature importance")
    fi = pd.Series(m["feature_importance"]).sort_values(ascending=False)
    st.bar_chart(fi)
    st.caption("Data is simulated from EV physics, so results show the pipeline works, not real-world accuracy.")
