"""Used-car price estimator (AutoScout24 Germany, 2021 listings).

Loads the pipeline exported by the notebook (car_price_model.joblib) and the input
metadata (meta.json). The pipeline contains its own encoder, so the app only has to
build one raw row with the same columns the model was trained on.

Inputs are a cascade (make > model > fuel > transmission > power) restricted to the
combinations that actually occur in the training data, so impossible cars cannot be entered.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent

# Test-set log-RMSE of the tuned XGBoost (notebook section 12). Used for the rough error band only.
TEST_LOG_RMSE = 0.1416
# Fuel categories that are scrape junk in the training data; hidden from the UI.
HIDDEN_FUELS = {"-/- (Fuel)", "Others"}
MIN_FREQ = 3          # models with fewer listings than this were grouped as "other" by the encoder
FEW_LISTINGS = 5      # warn when an exact configuration has fewer training listings than this

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗")


@st.cache_resource
def load_artifacts():
    model = joblib.load(HERE / "car_price_model.joblib")
    with open(HERE / "meta.json") as f:
        meta = json.load(f)
    return model, meta


def options_with_default(pairs):
    """pairs: (value, weight). Returns sorted unique values and the index of the heaviest (most common) one."""
    weights = {}
    for value, weight in pairs:
        weights[value] = weights.get(value, 0) + weight
    options = sorted(weights)
    return options, options.index(max(options, key=lambda o: weights[o]))


model, meta = load_artifacts()
REF_YEAR = meta["reference_year"]
year_min, year_max = meta["year_range"]
mileage_min, mileage_max = meta["mileage_range"]

st.title("🚗 Used Car Price Estimator")
st.caption(
    "Trained on about 38,000 German AutoScout24 listings from 2011-2021. "
    "Estimates are **2021-era asking prices in EUR**, not today's market values."
)

variants = meta.get("variants")
if not variants:
    st.error("meta.json has no 'variants' section. Re-export meta.json from the notebook, then restart the app.")
    st.stop()


def visible(make_model):
    """Variants of a model that the UI may offer (junk fuel categories removed)."""
    return [v for v in variants.get(make_model, []) if v[0] not in HIDDEN_FUELS]


# Only offer makes/models that have at least one usable variant
offer = {}
for mk, model_names in meta["makes"].items():
    names = [m for m in model_names if visible(f"{mk} {m}")]
    if names:
        offer[mk] = names

# Cascade of dropdowns. Keys include the parent choices, so changing a parent resets the children
# to that parent's most common option.
make = st.selectbox("Make", sorted(offer))
car_model = st.selectbox("Model", offer[make])
make_model = f"{make} {car_model}"           # must match how it was built in training
vs = visible(make_model)                      # rows: [fuel, gear, hp, n_listings]

fuels, i = options_with_default((v[0], v[3]) for v in vs)
fuel = st.selectbox("Fuel type", fuels, index=i, key=f"fuel|{make_model}")

gears, i = options_with_default((v[1], v[3]) for v in vs if v[0] == fuel)
gear = st.selectbox("Transmission", gears, index=i, key=f"gear|{make_model}|{fuel}")

powers, i = options_with_default((v[2], v[3]) for v in vs if v[0] == fuel and v[1] == gear)
hp = st.selectbox("Power (hp)", powers, index=i, format_func=lambda x: f"{x:g} hp",
                  key=f"hp|{make_model}|{fuel}|{gear}")

years = meta.get("years")
if years is None:
    st.warning("meta.json has no 'years' section, so the year range is not restricted per model. "
               "Re-export meta.json from the notebook and restart the app.")
y_lo, y_hi = years.get(make_model, [year_min, year_max]) if years else (year_min, year_max)

col1, col2 = st.columns(2)
with col1:
    if y_lo < y_hi:
        # offered range = the years this model actually appears in the training data
        year = st.slider("Registration year", y_lo, y_hi, (y_lo + y_hi) // 2, key=f"year|{make_model}")
    else:
        year = y_lo
        st.write(f"Registration year: **{year}** (the only year this model appears in the training data)")
with col2:
    mileage = st.number_input("Mileage (km)", min_value=mileage_min, max_value=mileage_max,
                              value=min(80_000, mileage_max), step=1_000)

# How much training data backs this exact configuration?
n_exact = sum(v[3] for v in vs if v[0] == fuel and v[1] == gear and v[2] == hp)
n_model = sum(v[3] for v in variants[make_model])
if n_model < MIN_FREQ:
    st.warning(f"Only {n_model} listing(s) of this model in the training data, so it was treated as a generic "
               "'other model': the estimate relies on make and specs only.")
elif n_exact < FEW_LISTINGS:
    st.warning(f"Only {n_exact} training listing(s) match this exact configuration, so the estimate is rough.")

if st.button("Estimate price", type="primary"):
    row = pd.DataFrame([{
        "age": REF_YEAR - year,                 # relative to 2021, the year the prices come from
        "mileage": mileage,
        "hp": float(hp),
        "fuel": fuel,
        "gear": gear,
        "make": make,
        "make_model": make_model,
    }])
    pred = float(np.exp(model.predict(row)[0]))  # the model predicts log(price)
    spread = float(np.exp(TEST_LOG_RMSE))

    st.metric("Estimated price", f"€{pred:,.0f}")
    st.caption(
        f"Typical error band: €{pred / spread:,.0f} to €{pred * spread:,.0f}. "
        "This is one standard error of the test set, not a guarantee."
    )

with st.expander("About this model"):
    st.write(
        "Tuned XGBoost on 7 features: age, mileage, horsepower, fuel, transmission, make and model. "
        "On a held-out test set of about 7,700 cars its typical error was about €1,460 "
        "(roughly 10%), against about €1,590 for a plain linear model. "
        "Error grows with age, from roughly 8% for 1-year-old cars to roughly 14% for 10-year-old cars."
    )
    st.write(
        "**Limits:** the model cannot see trim level, condition, accident history or options, "
        "so an individual car can differ a lot from the estimate. The data are asking prices, not sale prices, "
        "and cover cars up to about €100,000 only. The fuel, transmission and power options are limited to "
        "combinations and years that appear in the training data, and rare models have few examples, so estimates for them "
        "are rougher."
    )