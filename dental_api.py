import os
import json
import re
from datetime import datetime, date, time, timedelta
from typing import Optional, List

# Local development: read secrets (GEMINI_API_KEY, GOOGLE_CALENDAR_ID, ...) from .env next to this file.
# On Render these come from the service's Environment settings instead.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import clinic_data
from clinic_api import router as clinic_router
from voice_api import router as voice_router

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
SCOPES = ["https://www.googleapis.com/auth/calendar"]
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "service_account.json")
CALENDAR_ID = os.environ.get("GOOGLE_CALENDAR_ID", "primary")
TIMEZONE = os.environ.get("CLINIC_TIMEZONE", "Europe/Bratislava")

def clinic_tz():
    from zoneinfo import ZoneInfo
    return ZoneInfo(TIMEZONE)


def local_rfc3339(naive_local: datetime) -> str:
    """Clinic wall-clock time -> RFC 3339 with the clinic's real UTC offset (e.g. +02:00 in summer)."""
    return naive_local.replace(tzinfo=clinic_tz()).isoformat()


def to_clinic_local(rfc3339: str) -> datetime:
    """Any RFC 3339 time from Google -> naive clinic wall-clock time."""
    return datetime.fromisoformat(rfc3339.replace("Z", "+00:00")).astimezone(clinic_tz()).replace(tzinfo=None)


CLINIC_OPEN_HOUR = 8       # 08:00
CLINIC_CLOSE_HOUR = 17     # 17:00
SLOT_DURATION_MINUTES = 30 # 30 min per slot

app = FastAPI(
    title="DentalIQ Intelligent Revenue & Calendar Engine",
    description="Backend API powering DentalIQ scheduling, no-show protection, and calendar synchronization.",
    version="1.0.0"
)

# Enable CORS for local dev and cloud frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clinic modules (patients, predictions, financials, stock, procurement, equipment, plans, billing, labs, reports, settings)
app.include_router(clinic_router)
# AI voice receptionist (Gemini Live tokens, phone-verified find/cancel)
app.include_router(voice_router)


@app.on_event("startup")
def warm_up_clinic_data():
    """Build the clinic data and load the ML model at startup so the first page load is fast."""
    clinic_data.ensure_data()


# ==========================================
# AUTHENTICATION & GOOGLE CLIENT
# ==========================================
def get_calendar_service():
    """
    Initializes and returns an authorized Google Calendar service instance.
    Checks cloud environment variables first, falling back to local file.
    """
    # 1. Cloud deployment: check for raw JSON credentials in environment variables
    env_creds = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if env_creds:
        try:
            creds_info = json.loads(env_creds)
            creds = service_account.Credentials.from_service_account_info(
                creds_info, scopes=SCOPES
            )
            return build("calendar", "v3", credentials=creds, cache_discovery=False)
        except Exception as e:
            print("Error parsing GOOGLE_SERVICE_ACCOUNT_JSON:", e)

    # 2. Local development fallback: load from service_account.json file
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        return build("calendar", "v3", credentials=creds, cache_discovery=False)

    raise FileNotFoundError(
        f"Google credentials not found. Set GOOGLE_SERVICE_ACCOUNT_JSON env var or place file at: {SERVICE_ACCOUNT_FILE}"
    )


# ==========================================
# DATE & TIME HELPER UTILITIES
# ==========================================
def parse_date_flexible(val: str) -> date:
    """
    Parses date strings flexibly across multiple formats:
    - YYYY-MM-DD
    - YYYY-MM-DDTHH:MM:SS...
    - DD - MM - YYYY
    - DD-MM-YYYY
    - DD/MM/YYYY
    """
    if not val:
        return date.today()

    val_str = str(val).strip()

    # If it contains an ISO timestamp 'T', extract the date segment
    if "T" in val_str:
        val_str = val_str.split("T")[0]

    # Clean whitespace around hyphens (e.g. '27 - 09 - 2026' -> '27-09-2026')
    val_str = re.sub(r"\s*-\s*", "-", val_str)

    patterns = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
    ]

    for fmt in patterns:
        try:
            return datetime.strptime(val_str, fmt).date()
        except ValueError:
            continue

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unable to parse date '{val}'. Expected format YYYY-MM-DD or DD-MM-YYYY."
    )


def parse_time_flexible(val: str) -> time:
    """Parses time strings such as '09:00', '9:00', or '09:00:00'."""
    val_str = str(val).strip()
    patterns = ["%H:%M", "%H:%M:%S", "%I:%M %p"]
    for fmt in patterns:
        try:
            return datetime.strptime(val_str, fmt).time()
        except ValueError:
            continue

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unable to parse time '{val}'. Expected format HH:MM (e.g. 10:00)."
    )


def generate_clinic_slots(target_date: date) -> List[str]:
    """Generates all working slots for a given day in HH:MM format, following the opening hours in Settings."""
    return clinic_data.day_slots(target_date)


# ==========================================
# PYDANTIC DATA MODELS
# ==========================================
class BookingRequest(BaseModel):
    date: str
    time: str
    patient_name: Optional[str] = None
    patient: Optional[str] = None
    patient_phone: Optional[str] = None
    phone: Optional[str] = None
    treatment_type: Optional[str] = None
    procedure: Optional[str] = None
    provider: Optional[str] = "Dr. Novak"
    fee: Optional[float] = 0.0
    risk_level: Optional[str] = "Low"

    @property
    def resolved_patient_name(self) -> str:
        return (self.patient_name or self.patient or "Valued Patient").strip()

    @property
    def resolved_phone(self) -> str:
        raw = (self.patient_phone or self.phone or "").strip()
        return clinic_data.normalize_phone(raw) or raw

    @property
    def resolved_procedure(self) -> str:
        return (self.treatment_type or self.procedure or "General Consultation").strip()


# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "DentalIQ Engine",
        "version": "1.0.0",
        "auth_mode": "cloud_env" if os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON") else "local_file"
    }


@app.get("/api/calendar/events")
def list_calendar_events(date: str = Query(..., description="Target date in YYYY-MM-DD or DD-MM-YYYY format")):
    """
    Fetches all calendar events for a specific day and formats them
    for the receptionist dashboard.
    """
    target_date = parse_date_flexible(date)
    start_dt = local_rfc3339(datetime.combine(target_date, time.min))
    end_dt = local_rfc3339(datetime.combine(target_date, time(23, 59, 59)))

    try:
        service = get_calendar_service()
        events_result = service.events().list(
            calendarId=CALENDAR_ID,
            timeMin=start_dt,
            timeMax=end_dt,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        raw_items = events_result.get("items", [])
        formatted_events = []

        for item in raw_items:
            start_info = item.get("start", {})
            start_val = start_info.get("dateTime", start_info.get("date", ""))
            
            event_time = "09:00"
            if "T" in start_val:
                time_part = start_val.split("T")[1]
                event_time = time_part[:5]

            summary = item.get("summary", "Dental Appointment")
            description = item.get("description", "")

            # Extract metadata from summary/description
            patient_name = summary.split(" - ")[0] if " - " in summary else summary
            procedure = summary.split(" - ")[1] if " - " in summary else "Checkup"

            # Parse phone and fee if stored in description
            phone = ""
            fee = 150
            for line in description.split("\n"):
                if "Phone:" in line:
                    phone = line.replace("Phone:", "").strip()
                elif "Fee: €" in line:
                    try:
                        fee = float(line.replace("Fee: €", "").strip())
                    except ValueError:
                        pass

            formatted_events.append({
                "id": item.get("id"),
                "time": event_time,
                "name": patient_name,
                "patient": patient_name,
                "proc": procedure,
                "procedure": procedure,
                "prob": 0.15,
                "status": "Upcoming",
                "provider": "Dr. Novak",
                "fee": fee,
                "phone": phone
            })

        return {"status": "success", "date": target_date.isoformat(), "events": formatted_events}

    except Exception as e:
        print("Calendar list error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/calendar/available-slots")
def get_available_slots(date: str = Query(..., description="Target date in YYYY-MM-DD or DD-MM-YYYY format")):
    """
    Returns available 30-minute booking intervals for the selected date.
    Used by PublicBooking.jsx and the Voice Receptionist Bot.
    """
    target_date = parse_date_flexible(date)
    all_slots = generate_clinic_slots(target_date)

    if not all_slots:
        return {"status": "success", "date": target_date.isoformat(), "available_slots": []}
    start_iso = local_rfc3339(datetime.combine(target_date, time.min))
    end_iso = local_rfc3339(datetime.combine(target_date, time(23, 59, 59)))

    try:
        service = get_calendar_service()
        freebusy_req = {
            "timeMin": start_iso,
            "timeMax": end_iso,
            "timeZone": TIMEZONE,
            "items": [{"id": CALENDAR_ID}]
        }
        freebusy_res = service.freebusy().query(body=freebusy_req).execute()
        busy_spans = freebusy_res.get("calendars", {}).get(CALENDAR_ID, {}).get("busy", [])

        # Filter out slots that overlap with any busy span
        open_slots = []
        for slot_str in all_slots:
            slot_t = parse_time_flexible(slot_str)
            slot_start = datetime.combine(target_date, slot_t)
            slot_end = slot_start + timedelta(minutes=SLOT_DURATION_MINUTES)

            is_busy = False
            for span in busy_spans:
                busy_start = to_clinic_local(span["start"])
                busy_end = to_clinic_local(span["end"])

                # Check if times overlap: max(start1, start2) < min(end1, end2)
                if max(slot_start, busy_start) < min(slot_end, busy_end):
                    is_busy = True
                    break

            if not is_busy:
                open_slots.append(slot_str)

        return {
            "status": "success",
            "date": target_date.isoformat(),
            "available_slots": open_slots
        }

    except Exception as e:
        print("Available slots error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calendar/book-or-resolve")
def book_or_resolve_appointment(req: BookingRequest):
    """
    Schedules an appointment on Google Calendar.
    Checks for conflicts; if the slot is occupied, returns status: 'conflict'
    with alternative available slots.
    """
    target_date = parse_date_flexible(req.date)
    target_time = parse_time_flexible(req.time)

    slot_start = datetime.combine(target_date, target_time)
    slot_end = slot_start + timedelta(minutes=SLOT_DURATION_MINUTES)

    # Booking rules enforced on the server (website, staff and voice receptionist alike)
    open_slots = generate_clinic_slots(target_date)
    if not open_slots:
        raise HTTPException(status_code=400, detail="The clinic is closed on that day. Please choose another date.")
    if slot_start.strftime("%H:%M") not in open_slots:
        raise HTTPException(status_code=400, detail=f"{target_time.strftime('%H:%M')} is not a bookable time. Bookable times are {open_slots[0]}–{open_slots[-1]} on the half hour.")
    if slot_start <= clinic_data.clinic_now():
        raise HTTPException(status_code=400, detail="That time has already passed. Please choose a later time.")

    service = get_calendar_service()

    # 1. Collision check via freebusy query
    start_iso = local_rfc3339(slot_start)
    end_iso = local_rfc3339(slot_end)

    freebusy_req = {
        "timeMin": start_iso,
        "timeMax": end_iso,
        "timeZone": TIMEZONE,
        "items": [{"id": CALENDAR_ID}]
    }

    try:
        freebusy_res = service.freebusy().query(body=freebusy_req).execute()
        busy_spans = freebusy_res.get("calendars", {}).get(CALENDAR_ID, {}).get("busy", [])

        if busy_spans:
            # Slot is taken: compute 3 nearby open alternatives
            avail_res = get_available_slots(date=target_date.isoformat())
            open_slots = avail_res.get("available_slots", [])
            alternatives = [s for s in open_slots if s != req.time][:3]

            return {
                "status": "conflict",
                "requested_time": req.time,
                "suggested_alternatives": alternatives,
                "message": f"Requested slot {req.time} is already booked."
            }

        # 2. Insert appointment event into Google Calendar
        event_body = {
            "summary": f"{req.resolved_patient_name} - {req.resolved_procedure}",
            "description": (
                f"DentalIQ Automated Booking\n"
                f"Patient: {req.resolved_patient_name}\n"
                f"Phone: {req.resolved_phone}\n"
                f"Procedure: {req.resolved_procedure}\n"
                f"Fee: €{req.fee}\n"
                f"Provider: {req.provider}"
            ),
            "start": {
                "dateTime": slot_start.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": TIMEZONE
            },
            "end": {
                "dateTime": slot_end.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": TIMEZONE
            }
        }

        created_event = service.events().insert(calendarId=CALENDAR_ID, body=event_body).execute()

        return {
            "status": "success",
            "event_id": created_event.get("id"),
            "html_link": created_event.get("htmlLink"),
            "message": "Appointment booked successfully in Google Calendar!"
        }

    except Exception as e:
        print("Booking creation error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/calendar/events/{event_id}")
def delete_calendar_event(event_id: str):
    """Deletes an appointment event directly from Google Calendar."""
    try:
        service = get_calendar_service()
        service.events().delete(calendarId=CALENDAR_ID, eventId=event_id).execute()
        return {"status": "success", "deleted_id": event_id, "message": "Event deleted from Google Calendar"}
    except HttpError as e:
        if e.resp.status == 404:
            raise HTTPException(status_code=404, detail="Event not found in Google Calendar")
        raise HTTPException(status_code=e.resp.status, detail=str(e))
    except Exception as e:
        print("Event deletion error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))