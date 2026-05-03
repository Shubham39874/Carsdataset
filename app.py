import streamlit as st
import pickle
import pandas as pd
import os
import subprocess

st.set_page_config(page_title="Cars24 Advanced Predictor", layout="wide")

if not os.path.exists("models/car_pred"):
    st.info("Model not found. Training now... please wait.")
    subprocess.run(["python", "train.py"])
    st.success("Training complete!")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model_path = "models/car_pred"
    loc_path = "models/location_scores.pkl"
    data_path = "model/cars_24_combined.csv"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(loc_path, 'rb') as f:
        loc_scores = pickle.load(f)
        
    sample_df = pd.read_csv(data_path)
    return model, loc_scores, sample_df

try:
    reg_model, loc_map, full_df = load_assets()
except:
    st.error("Please run 'train.py' first to generate model files!")
    st.stop()

# --- UI ---
st.title("🚗 Cars24 Valuation")
st.write("Exhaustive AI model incorporating Vehicle Type and Regional Pricing.")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", 2010, 2026, 2021)
    distance = st.number_input("Kms Driven", 0, 500000, 20000)
    location = st.selectbox("Location", sorted(list(loc_map.keys())))

with col2:
    v_type = st.selectbox("Vehicle Type", ["HatchBack", "Sedan", "SUV", "Lux_sedan", "Lux_SUV"])
    fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

with col3:
    drive = st.selectbox("Drive", ["Manual", "Automatic"])
    owner = st.radio("Ownership", [1, 2, 3, 4], horizontal=True)

# --- CALCULATION ---
if st.button("Predict Absolute Price", use_container_width=True):
    # Mapping
    fuel_map = {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5}
    drive_map = {'Manual': 1, 'Automatic': 2}
    type_map = {'HatchBack': 1, 'Sedan': 2, 'SUV': 3, 'Lux_sedan': 4, 'Lux_SUV': 5}
    
    # Get the score for the selected location (default to global mean if missing)
    loc_score = loc_map.get(location, full_df['Price'].mean())

    input_data = [[
        year, distance, fuel_map[fuel], drive_map[drive], owner, type_map[v_type], loc_score
    ]]
    
    prediction = reg_model.predict(input_data)[0]
    
    st.divider()
    st.success(f"### Estimated Price: ₹ {int(prediction):,}")
    
    st.metric("Valuation", f"₹ {int(prediction):,}")

# --- SAMPLE DATA SECTION ---
st.divider()
st.subheader("📊 Dataset Preview")
st.dataframe(full_df.tail(10), use_container_width=True)