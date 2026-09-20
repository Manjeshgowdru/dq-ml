from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("dental_model.pkl")

@app.get("/")
def home():
    return {"message": "DentalIQ API is running!"}

@app.post("/predict")
def predict(appointment: dict):
    try:
        features = pd.DataFrame([{
            "Age":            appointment["age"],
            "age_group":      appointment["age_group"],
            "lead_time_days": appointment["lead_time_days"],
            "day_of_week":    appointment["day_of_week"],
            "SMS_received":   appointment["sms_received"],
            "Scholarship":    appointment["scholarship"],
            "Hipertension":   appointment["hipertension"],
            "Diabetes":       appointment["diabetes"],
            "Alcoholism":     appointment["alcoholism"],
            "Handcap":        appointment["handcap"]
        }])

        prob = float(model.predict_proba(features)[0][1])

        if prob >= 0.55:
            risk = "CRITICAL"
            action = "Call + WhatsApp + Request Deposit"
        elif prob >= 0.38:
            risk = "HIGH"
            action = "Send WhatsApp Reminder"
        elif prob >= 0.22:
            risk = "MODERATE"
            action = "Send SMS Confirmation"
        else:
            risk = "LOW"
            action = "Monitor Only"

        return {
            "noshow_probability": round(prob, 2),
            "risk_tier": risk,
            "action": action
        }
    except Exception as e:
        return {"error": str(e)}