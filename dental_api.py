import os
from datetime import datetime, timedelta
import joblib
import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pydantic import BaseModel

app = FastAPI(title="DentalIQ & Calendar Engine")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 1. Optional ML Model Setup
# -------------------------------------------------------------
model = None
if os.path.exists("dental_model.pkl"):
    try:
        model = joblib.load("dental_model.pkl")
    except Exception as e:
        print(f"Warning: Could not load dental_model.pkl: {e}")

# -------------------------------------------------------------
# 2. Calendar Configuration (Environment or Defaults)
# -------------------------------------------------------------
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TIMEZONE = os.getenv("TIMEZONE", "Europe/Bratislava")

# Clinic calendar target email
CLINIC_CALENDAR_ID = os.getenv("CLINIC_CALENDAR_ID", "gowdrislabs@gmail.com")

CLINIC_OPEN_HOUR = int(os.getenv("CLINIC_OPEN_HOUR", "8"))
CLINIC_CLOSE_HOUR = int(os.getenv("CLINIC_CLOSE_HOUR", "17"))
SLOT_DURATION_MINS = 30


def get_calendar_service():
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(f"Missing {SERVICE_ACCOUNT_FILE} in project root.")
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("calendar", "v3", credentials=creds)


class SlotBookingRequest(BaseModel):
    date: str
    time: str
    patient_name: str
    patient_phone: str
    treatment_type: str = "Vstupné vyšetrenie"


# -------------------------------------------------------------
# 3. Model & Health Endpoints
# -------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "DentalIQ API is running!"}


@app.post("/predict")
def predict(appointment: dict):
    if not model:
        raise HTTPException(status_code=503, detail="Model file dental_model.pkl not loaded.")
    try:
        features = pd.DataFrame([{
            "Age": appointment["age"],
            "age_group": appointment["age_group"],
            "lead_time_days": appointment["lead_time_days"],
            "day_of_week": appointment["day_of_week"],
            "SMS_received": appointment["sms_received"],
            "Scholarship": appointment["scholarship"],
            "Hipertension": appointment["hipertension"],
            "Diabetes": appointment["diabetes"],
            "Alcoholism": appointment["alcoholism"],
            "Handcap": appointment["handcap"]
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


# -------------------------------------------------------------
# 4. Google Calendar Endpoints
# -------------------------------------------------------------
@app.get("/api/calendar/events")
def get_daily_events(date: str = None):
    """
    Fetches scheduled appointments from Google Calendar for the target date (YYYY-MM-DD).
    Defaults to today if none is provided.
    """
    try:
        service = get_calendar_service()
        tz = pytz.timezone(TIMEZONE)
        target_date_str = date or datetime.now(tz).strftime("%Y-%m-%d")

        day_start = tz.localize(datetime.strptime(f"{target_date_str} 00:00:00", "%Y-%m-%d %H:%M:%S"))
        day_end = tz.localize(datetime.strptime(f"{target_date_str} 23:59:59", "%Y-%m-%d %H:%M:%S"))

        events_result = service.events().list(
            calendarId=CLINIC_CALENDAR_ID,
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        raw_events = events_result.get("items", [])
        formatted_events = []
        now = datetime.now(tz)

        for item in raw_events:
            start_iso = item["start"].get("dateTime", item["start"].get("date"))
            dt = datetime.fromisoformat(start_iso)
            time_str = dt.strftime("%H:%M")

            summary = item.get("summary", "Zubné vyšetrenie")
            patient_name = summary
            proc = "Vstupné vyšetrenie"
            if ":" in summary:
                parts = summary.split(":", 1)[1].strip()
                if "(" in parts and parts.endswith(")"):
                    patient_name = parts.split("(")[0].strip()
                    proc = parts.split("(")[1].replace(")", "").strip()
                else:
                    patient_name = parts

            # Extract phone number from description if stored
            description = item.get("description", "")
            extracted_phone = ""
            for line in description.split("\n"):
                if "Telefón:" in line:
                    extracted_phone = line.replace("Telefón:", "").strip()

            formatted_events.append({
                "id": item.get("id"),
                "time": time_str,
                "name": patient_name,
                "proc": proc,
                "provider": "Dr. Patel",
                "fee": 150,
                "prob": 0.15,
                "phone": extracted_phone,
                "status": "Upcoming" if dt > now else "Completed"
            })

        return {
            "date": target_date_str,
            "events": formatted_events
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/api/calendar/book-or-resolve")
def book_or_resolve(req: SlotBookingRequest):
    try:
        service = get_calendar_service()
        tz = pytz.timezone(TIMEZONE)

        start_naive = datetime.strptime(f"{req.date} {req.time}", "%Y-%m-%d %H:%M")
        start_dt = tz.localize(start_naive)
        end_dt = start_dt + timedelta(minutes=SLOT_DURATION_MINS)

        day_start = tz.localize(datetime.strptime(f"{req.date} {CLINIC_OPEN_HOUR:02d}:00", "%Y-%m-%d %H:%M"))
        day_end = tz.localize(datetime.strptime(f"{req.date} {CLINIC_CLOSE_HOUR:02d}:00", "%Y-%m-%d %H:%M"))

        body = {
            "timeMin": day_start.isoformat(),
            "timeMax": day_end.isoformat(),
            "timeZone": TIMEZONE,
            "items": [{"id": CLINIC_CALENDAR_ID}]
        }
        fb_result = service.freebusy().query(body=body).execute()
        busy_slots = fb_result.get("calendars", {}).get(CLINIC_CALENDAR_ID, {}).get("busy", [])

        # Check for collision
        for busy in busy_slots:
            busy_start = datetime.fromisoformat(busy["start"])
            busy_end = datetime.fromisoformat(busy["end"])
            if max(start_dt, busy_start) < min(end_dt, busy_end):
                alternatives = []
                curr = day_start
                while curr + timedelta(minutes=SLOT_DURATION_MINS) <= day_end and len(alternatives) < 2:
                    c_end = curr + timedelta(minutes=SLOT_DURATION_MINS)
                    is_busy = any(
                        max(curr, datetime.fromisoformat(b["start"])) < min(c_end, datetime.fromisoformat(b["end"]))
                        for b in busy_slots
                    )
                    c_str = curr.strftime("%H:%M")
                    if not is_busy and c_str != req.time:
                        alternatives.append(c_str)
                    curr += timedelta(minutes=SLOT_DURATION_MINS)

                return {
                    "status": "conflict",
                    "requested_time": req.time,
                    "suggested_alternatives": alternatives,
                    "message": f"Slot {req.time} is already booked."
                }

        # Slot is available: insert appointment
        event = {
            "summary": f"Zubné ošetrenie: {req.patient_name} ({req.treatment_type})",
            "description": f"Pacient: {req.patient_name}\nTelefón: {req.patient_phone}\nRezervované cez DentalIQ",
            "start": {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": TIMEZONE},
        }
        created_event = service.events().insert(calendarId=CLINIC_CALENDAR_ID, body=event).execute()
        return {
            "status": "booked",
            "event_id": created_event.get("id"),
            "date": req.date,
            "time": req.time,
            "message": f"Appointment successfully confirmed for {req.patient_name}."
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=repr(e))


@app.delete("/api/calendar/events/{event_id}")
def delete_event(event_id: str):
    """
    Deletes an event directly from Google Calendar using its unique event ID.
    """
    try:
        service = get_calendar_service()
        service.events().delete(
            calendarId=CLINIC_CALENDAR_ID,
            eventId=event_id
        ).execute()

        return {
            "status": "deleted",
            "event_id": event_id,
            "message": "Appointment successfully removed from Google Calendar."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=repr(e))