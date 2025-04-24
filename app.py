import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Title
st.set_page_config(page_title="Green Space Wellbeing Predictor", layout="centered")
st.title("🌿 University Green Space & Wellbeing Predictor")

# Load dataset
df = pd.read_csv("university_student_wellbeing_synthetic.csv")

# Define features and target
features = ["NDVI Score", "Green Space Area (sq.meters)",
            "Walking Distance (mins)", "Shade Coverage (%)", "Academic Stress Level"]
target = "Predicted Wellbeing Score"

# Sidebar - Input Parameters
st.sidebar.header("Input Green Space and Student Parameters")

ndvi = st.sidebar.slider("NDVI Score", 0.3, 0.9, 0.6)
area = st.sidebar.number_input("Green Space Area (sq. meters)", 100, 10000, 5000)
walk = st.sidebar.slider("Walking Distance (mins)", 1, 30, 10)
shade = st.sidebar.slider("Shade Coverage (%)", 0.0, 100.0, 50.0)
stress = st.sidebar.slider("Academic Stress Level", 1, 10, 5)

# Prepare input
input_df = pd.DataFrame([[ndvi, area, walk, shade, stress]], columns=features)

# Train model
X = df[features]
y = df[target]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Make prediction
prediction = model.predict(input_df)[0]

# Display prediction
st.subheader("🎯 Predicted Wellbeing Score")
st.metric(label="Wellbeing Score", value=round(prediction, 2))

# Show inputs
st.subheader("📋 Your Input Summary")
st.dataframe(input_df)
