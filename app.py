# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🧬 Disease Prediction System")

# Step 1: Load Static CSV
file_path = "dataset_for_project.csv"  # Make sure this is in the same folder
df = pd.read_csv(file_path)

# Step 2: Clean + Encode
df.fillna(0, inplace=True)

le_sym1 = LabelEncoder()
le_sym2 = LabelEncoder()
le_sym3 = LabelEncoder()
le_disease = LabelEncoder()

df['Symptom_1_en'] = le_sym1.fit_transform(df['Symptom_1'])
df['Symptom_2_en'] = le_sym2.fit_transform(df['Symptom_2'])
df['Symptom_3_en'] = le_sym3.fit_transform(df['Symptom_3'])
df['Disease_en'] = le_disease.fit_transform(df['Disease'])

# Step 3: Train the model
X = df[['Symptom_1_en', 'Symptom_2_en', 'Symptom_3_en']]
y = df['Disease_en']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders (optional)
joblib.dump(model, 'disease_predictor_model.h5')
joblib.dump(le_sym1, 'le_sym1.pkl')
joblib.dump(le_sym2, 'le_sym2.pkl')
joblib.dump(le_sym3, 'le_sym3.pkl')
joblib.dump(le_disease, 'le_disease.pkl')

# Step 4: Evaluation
st.subheader("📈 Model Evaluation")
acc = accuracy_score(y_test, model.predict(X_test))
st.write(f"✅ Accuracy: `{acc:.2f}`")

with st.expander("📄 Show Classification Report"):
    report = classification_report(y_test, model.predict(X_test), target_names=le_disease.classes_, zero_division=0)
    st.text(report)

# Step 5: Prediction UI
st.subheader("🎯 Predict Disease from Symptoms")

sym1 = st.selectbox("Select Symptom 1", le_sym1.classes_)
sym2 = st.selectbox("Select Symptom 2", le_sym2.classes_)
sym3 = st.selectbox("Select Symptom 3", le_sym3.classes_)

if st.button("🔍 Predict"):
    input_array = np.array([
        le_sym1.transform([sym1])[0],
        le_sym2.transform([sym2])[0],
        le_sym3.transform([sym3])[0]
    ]).reshape(1, -1)

    pred_encoded = model.predict(input_array)[0]
    predicted_disease = le_disease.inverse_transform([pred_encoded])[0]

    st.success(f"🩺 Predicted Disease: **{predicted_disease}**")