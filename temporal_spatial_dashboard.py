from __future__ import annotations

from datetime import date

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from tn_model_utils import (
    CROP_GROWTH_STAGES,
    IRRIGATION_TYPES,
    MODEL_ARTIFACT_PATH,
    SOIL_TYPES,
    TN_CROPS,
    TN_CENTER,
    TN_DISTRICTS,
    build_input_frame,
    district_to_region,
    fetch_live_context,
    generate_suitability_note,
    infer_agri_season,
    load_trained_artifacts,
    prediction_with_confidence,
    soil_defaults,
)


def _init_state() -> None:
    if "district" not in st.session_state:
        st.session_state.district = "Chennai"
    if "latitude" not in st.session_state:
        st.session_state.latitude = TN_CENTER[0]
    if "longitude" not in st.session_state:
        st.session_state.longitude = TN_CENTER[1]
    if "region" not in st.session_state:
        st.session_state.region = district_to_region(st.session_state.district)
    if "enable_map_click" not in st.session_state:
        st.session_state.enable_map_click = False
    if "context_source" not in st.session_state:
        st.session_state.context_source = "TN heuristics"

    defaults = soil_defaults("Loamy")
    if "soil_type" not in st.session_state:
        st.session_state.soil_type = "Loamy"
    if "soil_ph" not in st.session_state:
        st.session_state.soil_ph = defaults["soil_ph"]
    if "soil_moisture" not in st.session_state:
        st.session_state.soil_moisture = defaults["soil_moisture"]
    if "organic_carbon" not in st.session_state:
        st.session_state.organic_carbon = defaults["organic_carbon"]
    if "electrical_conductivity" not in st.session_state:
        st.session_state.electrical_conductivity = defaults["ec"]
    if "temperature" not in st.session_state:
        st.session_state.temperature = 30.0
    if "humidity" not in st.session_state:
        st.session_state.humidity = 70.0
    if "rainfall" not in st.session_state:
        st.session_state.rainfall = 1100.0
    if "pending_lat" not in st.session_state:
        st.session_state.pending_lat = None
    if "pending_lon" not in st.session_state:
        st.session_state.pending_lon = None


def _apply_soil_defaults(soil_type: str) -> None:
    defaults = soil_defaults(soil_type)
    st.session_state.soil_type = soil_type
    st.session_state.soil_ph = defaults["soil_ph"]
    st.session_state.soil_moisture = defaults["soil_moisture"]
    st.session_state.organic_carbon = defaults["organic_carbon"]
    st.session_state.electrical_conductivity = defaults["ec"]


def _update_context_from_location(latitude: float, longitude: float, selected_date: date) -> None:
    season = infer_agri_season(selected_date.month)
    weather = fetch_live_context(latitude, longitude, st.session_state.region, season, st.session_state.soil_type)
    st.session_state.latitude = round(latitude, 6)
    st.session_state.longitude = round(longitude, 6)
    st.session_state.temperature = weather["temperature"]
    st.session_state.humidity = weather["humidity"]
    st.session_state.rainfall = weather["rainfall"]
    st.session_state.soil_moisture = weather["soil_moisture"]
    st.session_state.soil_ph = weather["soil_ph"]
    st.session_state.organic_carbon = weather["organic_carbon"]
    st.session_state.electrical_conductivity = weather["electrical_conductivity"]
    st.session_state.context_source = weather["source"]


def _district_map() -> None:
    map_obj = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB positron")

    for district, coords in TN_DISTRICTS.items():
        folium.CircleMarker(
            location=[coords[0], coords[1]],
            radius=4,
            fill=True,
            color="#117a65",
            fill_opacity=0.85,
            tooltip=district,
        ).add_to(map_obj)

    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        tooltip="Selected location",
        icon=folium.Icon(color="red", icon="map-marker"),
    ).add_to(map_obj)

    map_data = st_folium(map_obj, height=380, width=None, key="tn-map", returned_objects=["last_clicked"])
    clicked = map_data.get("last_clicked") if map_data else None
    if clicked:
        st.session_state.pending_lat = clicked["lat"]
        st.session_state.pending_lon = clicked["lng"]


def _manual_context_controls(selected_date: date) -> None:
    st.markdown("### Soil and Climate Context")
    st.caption("Auto-filled from district or map selection, but you can manually change any field.")

    left, right = st.columns(2)
    with left:
        st.session_state.latitude = st.number_input("Latitude", min_value=8.0, max_value=13.6, value=float(st.session_state.latitude), step=0.0001, format="%.6f")
        st.session_state.temperature = st.number_input("Temperature (C)", min_value=12.0, max_value=45.0, value=float(st.session_state.temperature), step=0.1, format="%.1f")
        st.session_state.soil_ph = st.number_input("Soil pH", min_value=4.5, max_value=8.8, value=float(st.session_state.soil_ph), step=0.01, format="%.2f")
        st.session_state.organic_carbon = st.number_input("Organic Carbon", min_value=0.0, max_value=2.5, value=float(st.session_state.organic_carbon), step=0.01, format="%.2f")
        st.session_state.soil_moisture = st.number_input("Soil Moisture", min_value=5.0, max_value=95.0, value=float(st.session_state.soil_moisture), step=0.1, format="%.1f")
    with right:
        st.session_state.longitude = st.number_input("Longitude", min_value=76.0, max_value=80.6, value=float(st.session_state.longitude), step=0.0001, format="%.6f")
        st.session_state.humidity = st.number_input("Humidity (%)", min_value=15.0, max_value=99.0, value=float(st.session_state.humidity), step=0.1, format="%.1f")
        st.session_state.rainfall = st.number_input("Rainfall (mm)", min_value=300.0, max_value=2800.0, value=float(st.session_state.rainfall), step=10.0, format="%.1f")
        st.session_state.electrical_conductivity = st.number_input("Electrical Conductivity", min_value=0.0, max_value=3.5, value=float(st.session_state.electrical_conductivity), step=0.01, format="%.2f")
        st.metric("Region", st.session_state.region)
        st.metric("Data Source", st.session_state.context_source)

def main() -> None:
    st.set_page_config(page_title="Tamil Nadu Fertilizer Dashboard", layout="wide")
    _init_state()

    st.title("Tamil Nadu Fertilizer Recommendation Dashboard")
    st.caption("Pre-trained model for Tamil Nadu using richer fertilizer data and practical field inputs.")

    if not MODEL_ARTIFACT_PATH.exists():
        st.error("Pre-trained model not found. Run `python pretrain_tn_model.py` once first.")
        st.stop()

    artifacts = load_trained_artifacts(MODEL_ARTIFACT_PATH)

    current_accuracy = float(artifacts.get("accuracy", 0.0)) * 100.0
    baseline_accuracy = {
        "Current Spark RF": round(current_accuracy, 2),
        "Historical MLP": 14.24,
        "Historical RF": 13.97,
    }
    st.markdown("### Model Accuracy Graph")
    st.bar_chart(pd.DataFrame.from_dict(baseline_accuracy, orient="index", columns=["Accuracy (%)"]))

    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Location")
        district = st.selectbox("District", options=list(TN_DISTRICTS.keys()), index=list(TN_DISTRICTS.keys()).index(st.session_state.district))
        st.session_state.district = district
        st.session_state.region = district_to_region(district)

        if st.button("Use district coordinates", width="stretch"):
            lat, lon = TN_DISTRICTS[district]
            _update_context_from_location(lat, lon, date.today())

        st.session_state.enable_map_click = st.toggle(
            "Enable map click selection",
            value=st.session_state.enable_map_click,
            help="Keep this off for a faster UI, turn it on only when selecting a point on the map.",
        )

        if st.session_state.enable_map_click:
            _district_map()
            if st.session_state.pending_lat is None:
                st.caption("Click a point on the map to capture coordinates.")
        else:
            st.caption("Map click mode is off for speed. Use district selection or manual fields.")

    with right:
        st.subheader("Farm Inputs")
        with st.form("prediction_form"):
            selected_date = st.date_input("Sowing / Planning Date", value=date.today())
            crop_type = st.selectbox("Crop Type", options=TN_CROPS)
            crop_growth_stage = st.selectbox("Crop Growth Stage", options=CROP_GROWTH_STAGES, index=1)
            soil_type = st.selectbox("Soil Type", options=SOIL_TYPES, index=SOIL_TYPES.index(st.session_state.soil_type))
            irrigation_type = st.selectbox("Irrigation Type", options=IRRIGATION_TYPES, index=IRRIGATION_TYPES.index("Rainfed"))

            if soil_type != st.session_state.soil_type:
                _apply_soil_defaults(soil_type)

            nitrogen_level = st.slider("Nitrogen Level", min_value=0, max_value=200, value=60)
            phosphorus_level = st.slider("Phosphorus Level", min_value=0, max_value=150, value=40)
            potassium_level = st.slider("Potassium Level", min_value=0, max_value=150, value=40)

            _manual_context_controls(selected_date)
            run_prediction = st.form_submit_button("Predict Fertilizer", type="primary")

        if st.session_state.pending_lat is not None and st.session_state.pending_lon is not None:
            st.info(f"Clicked point captured: {st.session_state.pending_lat:.6f}, {st.session_state.pending_lon:.6f}")
            if st.button("Apply clicked location", width="stretch"):
                _update_context_from_location(st.session_state.pending_lat, st.session_state.pending_lon, selected_date)
                st.session_state.pending_lat = None
                st.session_state.pending_lon = None

    if run_prediction:
        features = build_input_frame(
            soil_type=soil_type,
            soil_ph=st.session_state.soil_ph,
            soil_moisture=st.session_state.soil_moisture,
            organic_carbon=st.session_state.organic_carbon,
            electrical_conductivity=st.session_state.electrical_conductivity,
            nitrogen_level=nitrogen_level,
            phosphorus_level=phosphorus_level,
            potassium_level=potassium_level,
            temperature=st.session_state.temperature,
            humidity=st.session_state.humidity,
            rainfall=st.session_state.rainfall,
            crop_type=crop_type,
            crop_growth_stage=crop_growth_stage,
            irrigation_type=irrigation_type,
            region=st.session_state.region,
            selected_date=selected_date,
        )

        recommended, confidence_table = prediction_with_confidence(artifacts["model"], features)
        season = infer_agri_season(selected_date.month)
        suitability_note = generate_suitability_note(
            fertilizer_name=recommended,
            crop_type=crop_type,
            soil_type=soil_type,
            soil_ph=st.session_state.soil_ph,
            soil_moisture=st.session_state.soil_moisture,
            rainfall=st.session_state.rainfall,
            nitrogen_level=nitrogen_level,
            phosphorus_level=phosphorus_level,
            potassium_level=potassium_level,
            crop_growth_stage=crop_growth_stage,
            irrigation_type=irrigation_type,
            region=st.session_state.region,
            season=season,
        )

        st.success(f"Recommended Fertilizer: {recommended}")
        st.info(f"Why suitable: {suitability_note}")
        st.dataframe(confidence_table, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
