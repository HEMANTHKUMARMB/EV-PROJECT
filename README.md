# EV Consumption Prediction

Predicts EV energy use (kWh/100 km) from 7 trip inputs, using a physics-simulated 6,000-trip dataset.

## Run
    pip install -r requirements.txt
    python data_generation.py   # -> data/ev_trips.csv
    python train_model.py       # -> models/best_model.joblib, models/metrics.json
    streamlit run app.py
    pip install pandas numpy scikit-learn joblib streamlit
    cd ev_consumption




## Files
- data_generation.py - physics simulator (aero drag, rolling resistance, grade + regen, cold-battery loss, A/C load, driving style, noise)
- train_model.py - 80/20 split, compares Linear Regression / Random Forest / Gradient Boosting (MAE, RMSE, R2), saves the best
- predict.py - loads the model; returns consumption, trip energy and range
- app.py - Streamlit UI (Predict + Model comparison tabs)

Data is simulated: results show the pipeline works, not real-world accuracy.
