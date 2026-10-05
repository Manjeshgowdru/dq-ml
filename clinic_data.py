"""
DentalIQ clinic data store.

Generates one consistent demo clinic (patients, appointments, invoices, treatment plans,
stock, suppliers, purchase orders, equipment, lab orders) so every module reads the same data.
Data is regenerated on startup (deterministic seed, anchored to today's date) and kept in memory.
Clinic settings are persisted to clinic_settings.json.

Later this module is replaced by a real database (Supabase); the API shape stays the same.
"""
import copy
import json
import os
import random
import threading
import unicodedata
from datetime import date, datetime, time, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, "clinic_settings.json")
MODEL_PATH = os.path.join(HERE, "no_show_model_v2.pkl")
MODEL_META_PATH = os.path.join(HERE, "no_show_model_v2.json")

LOCK = threading.RLock()


def clinic_now() -> datetime:
    """Current wall-clock time in the clinic's timezone (naive)."""
    tz_name = os.environ.get("CLINIC_TIMEZONE", "Europe/Bratislava")
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz_name)).replace(tzinfo=None)
    except Exception:
        return datetime.now()


# ==========================================
# DEFAULT SETTINGS
# ==========================================
DEFAULT_TREATMENTS = [
    {"code": "CONSULT", "name": "Initial Dental Consultation", "name_sk": "Vstupné vyšetrenie", "category": "General", "duration": 30, "price": 40, "online": True, "icon": "🩺"},
    {"code": "CHECKUP", "name": "Preventive Check-up", "name_sk": "Preventívna prehliadka", "category": "General", "duration": 30, "price": 35, "online": True, "icon": "📋"},
    {"code": "HYGIENE", "name": "Dental Hygiene", "name_sk": "Dentálna hygiena", "category": "Hygiene", "duration": 45, "price": 75, "online": True, "icon": "✨"},
    {"code": "FILLING", "name": "Composite Filling", "name_sk": "Kompozitná výplň", "category": "Restorative", "duration": 45, "price": 95, "online": True, "icon": "🦷"},
    {"code": "ROOT_CANAL", "name": "Root Canal Treatment", "name_sk": "Ošetrenie koreňového kanálika", "category": "Endodontics", "duration": 90, "price": 320, "online": False, "icon": "🔬"},
    {"code": "EXTRACTION", "name": "Tooth Extraction", "name_sk": "Extrakcia zuba", "category": "Surgery", "duration": 30, "price": 90, "online": False, "icon": "🩹"},
    {"code": "CROWN", "name": "Zirconia Crown", "name_sk": "Zirkónová korunka", "category": "Prosthetics", "duration": 60, "price": 450, "online": False, "icon": "👑"},
    {"code": "IMPLANT_CONSULT", "name": "Implant Consultation", "name_sk": "Konzultácia implantátu", "category": "Implants", "duration": 30, "price": 50, "online": True, "icon": "🔩"},
    {"code": "IMPLANT", "name": "Dental Implant", "name_sk": "Zubný implantát", "category": "Implants", "duration": 90, "price": 1100, "online": False, "icon": "🦾"},
    {"code": "WHITENING", "name": "Teeth Whitening", "name_sk": "Bielenie zubov", "category": "Cosmetic", "duration": 60, "price": 250, "online": True, "icon": "💎"},
    {"code": "ALIGNERS", "name": "Clear Aligner Check", "name_sk": "Kontrola alignerov", "category": "Orthodontics", "duration": 30, "price": 60, "online": False, "icon": "😁"},
    {"code": "EMERGENCY", "name": "Emergency Pain Relief", "name_sk": "Akútna bolesť", "category": "General", "duration": 30, "price": 60, "online": True, "icon": "🚨"},
]

DEFAULT_TEAM = [
    {"id": "U1", "name": "Dr. Peter Novák", "role": "Dentist", "specialty": "General dentistry", "email": "novak@smilesdental.example", "color": "#4F5BD5", "active": True},
    {"id": "U2", "name": "Dr. Zuzana Kollárová", "role": "Dentist", "specialty": "Implants & prosthetics", "email": "kollarova@smilesdental.example", "color": "#7C3AED", "active": True},
    {"id": "U3", "name": "Dr. Tomáš Szabó", "role": "Dentist", "specialty": "Endodontics & surgery", "email": "szabo@smilesdental.example", "color": "#0891B2", "active": True},
    {"id": "U4", "name": "Mgr. Lucia Lukáčová", "role": "Hygienist", "specialty": "Dental hygiene", "email": "lukacova@smilesdental.example", "color": "#059669", "active": True},
    {"id": "U5", "name": "Jana Blahová", "role": "Receptionist", "specialty": "Front desk", "email": "recepcia@smilesdental.example", "color": "#B45309", "active": True},
    {"id": "U6", "name": "Manjesh Gowda", "role": "Admin", "specialty": "Practice management", "email": "admin@smilesdental.example", "color": "#0F172A", "active": True},
]

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

DEFAULT_SETTINGS = {
    "clinic": {
        "name": "Smiles Dental Clinic",
        "legal_name": "Smiles Dental s.r.o.",
        "address": "Hlavná 42, 040 01 Košice, Slovakia",
        "phone": "+421 55 123 4567",
        "email": "recepcia@smilesdental.example",
        "website": "",
        "company_id": "",
        "vat_id": "",
        "currency": "EUR",
        "timezone": "Europe/Bratislava",
        "default_language": "sk",
    },
    "hours": {
        **{d: {"open": "08:00", "close": "17:00", "closed": False} for d in WEEKDAYS[:5]},
        "sat": {"open": "08:00", "close": "12:00", "closed": True},
        "sun": {"open": "08:00", "close": "12:00", "closed": True},
    },
    "slot_minutes": 30,
    "chairs": 4,
    "treatments": DEFAULT_TREATMENTS,
    "team": DEFAULT_TEAM,
    "risk": {
        "moderate": 0.20,
        "high": 0.32,
        "critical": 0.45,
        "deposit_eur": 30,
        "actions": {
            "low": "Monitor only",
            "moderate": "SMS confirmation",
            "high": "WhatsApp reminder",
            "critical": "Call + WhatsApp + deposit request",
        },
    },
    "reminders": {
        "sms_hours_before": 48,
        "whatsapp_hours_before": 24,
        "template_sk": "Dobrý deň {name}, pripomíname Vám termín v {clinic} dňa {date} o {time}. Ak termín nemôžete dodržať, prosím kontaktujte nás.",
        "template_en": "Hello {name}, this is a reminder of your appointment at {clinic} on {date} at {time}. If you cannot attend, please let us know.",
    },
    "voice": {
        "enabled": True,
        "provider": "Gemini Live",
        "languages": ["sk", "en"],
        "greeting_sk": "Dobrý deň, dovolali ste sa do {clinic}. Som virtuálna asistentka s umelou inteligenciou. Ako Vám môžem pomôcť?",
        "greeting_en": "Hello, you have reached {clinic}. I am an AI virtual assistant. How can I help you?",
        "ai_disclosure": True,
        "can_book": True,
        "can_cancel": True,
        "handoff_phone": "+421 55 123 4567",
        "max_call_minutes": 5,
    },
    "booking": {
        "enabled": True,
        "days_ahead": 60,
        "min_notice_hours": 2,
        "allow_same_day": True,
        "show_prices": False,
        "require_consent": True,
        "consent_text_en": "I agree that Smiles Dental Clinic processes my name and phone number to manage my appointment.",
        "consent_text_sk": "Súhlasím so spracovaním môjho mena a telefónneho čísla za účelom správy termínu.",
    },
    "notifications": {
        "low_stock_alert": True,
        "overdue_lab_alert": True,
        "overdue_invoice_days": 30,
        "daily_summary_email": False,
        "summary_email": "",
    },
    "privacy": {
        "data_region": "EU",
        "retention_years": 20,
        "record_calls": False,
        "dpo_email": "",
    },
}

# ==========================================
# DEMO DATA VOCABULARY
# ==========================================
FIRST_M = ["Peter", "Martin", "Tomáš", "Michal", "Jakub", "Lukáš", "Marek", "Ján", "Juraj", "Matej", "Filip", "Adam", "Róbert", "Patrik", "Dávid", "Milan", "Jozef", "Ondrej"]
FIRST_F = ["Eva", "Anna", "Lucia", "Jana", "Zuzana", "Mária", "Katarína", "Veronika", "Simona", "Petra", "Monika", "Barbora", "Kristína", "Martina", "Lenka", "Ivana", "Dominika", "Nikola"]
LAST = [("Kováč", "Kováčová"), ("Horváth", "Horváthová"), ("Varga", "Vargová"), ("Tóth", "Tóthová"), ("Nagy", "Nagyová"),
        ("Baláž", "Balážová"), ("Molnár", "Molnárová"), ("Novák", "Nováková"), ("Lukáč", "Lukáčová"), ("Kollár", "Kollárová"),
        ("Mikuš", "Mikušová"), ("Šimko", "Šimková"), ("Polák", "Poláková"), ("Hudák", "Hudáková"), ("Gajdoš", "Gajdošová"),
        ("Oravec", "Oravcová"), ("Kráľ", "Kráľová"), ("Blaho", "Blahová"), ("Urban", "Urbanová"), ("Sloboda", "Slobodová"),
        ("Bartoš", "Bartošová"), ("Fedor", "Fedorová"), ("Hric", "Hricová"), ("Jurko", "Jurková")]
CITIES = ["Košice", "Košice", "Košice", "Prešov", "Michalovce", "Spišská Nová Ves", "Rožňava", "Trebišov"]
INSURERS = ["VšZP", "Dôvera", "Union"]

TREATMENT_WEIGHTS = {"CONSULT": 8, "CHECKUP": 22, "HYGIENE": 20, "FILLING": 18, "ROOT_CANAL": 5, "EXTRACTION": 5,
                     "CROWN": 5, "IMPLANT_CONSULT": 4, "IMPLANT": 2, "WHITENING": 4, "ALIGNERS": 4, "EMERGENCY": 3}
PROVIDER_FOR_CATEGORY = {"General": ["U1", "U2", "U3"], "Hygiene": ["U4"], "Restorative": ["U1", "U3"], "Endodontics": ["U3"],
                         "Surgery": ["U3", "U2"], "Prosthetics": ["U2"], "Implants": ["U2"], "Cosmetic": ["U1", "U4"], "Orthodontics": ["U1"]}
# Public insurance covers part of basic care in Slovakia
INSURANCE_SHARE = {"CHECKUP": 1.0, "CONSULT": 0.5, "FILLING": 0.35, "EXTRACTION": 0.6, "ROOT_CANAL": 0.3, "EMERGENCY": 0.5}

SUPPLIERS = [
    {"id": "S1", "name": "DentaMed Distribution s.r.o.", "category": "Consumables", "email": "orders@dentamed.example", "phone": "+421 2 555 0101", "lead_days": 2, "rating": 4.6},
    {"id": "S2", "name": "ProDent Supply SK", "category": "Restorative & endo", "email": "obchod@prodent.example", "phone": "+421 55 555 0202", "lead_days": 3, "rating": 4.4},
    {"id": "S3", "name": "SteriClean Medical", "category": "Infection control", "email": "sales@stericlean.example", "phone": "+421 2 555 0303", "lead_days": 1, "rating": 4.8},
    {"id": "S4", "name": "ImplantLine Europe", "category": "Implants", "email": "orders@implantline.example", "phone": "+420 2 555 0404", "lead_days": 5, "rating": 4.5},
    {"id": "S5", "name": "OrthoSmile Partners", "category": "Orthodontics & hygiene", "email": "info@orthosmile.example", "phone": "+421 51 555 0505", "lead_days": 4, "rating": 4.2},
]

STOCK_CATALOG = [
    # name, category, unit, min, reorder, unit_cost, supplier, weekly_use, has_expiry
    ("Composite Resin A2 (4g syringe)", "Restorative", "syringe", 10, 20, 18.5, "S2", 6, True),
    ("Composite Resin A3 (4g syringe)", "Restorative", "syringe", 8, 16, 18.5, "S2", 4, True),
    ("Bonding Agent (5ml)", "Restorative", "bottle", 3, 6, 42.0, "S2", 1, True),
    ("Glass Ionomer Cement", "Restorative", "kit", 2, 4, 55.0, "S2", 0.5, True),
    ("Articaine 4% Anaesthetic (50 carp.)", "Anaesthesia", "box", 6, 10, 38.0, "S1", 2, True),
    ("Topical Anaesthetic Gel", "Anaesthesia", "jar", 2, 4, 12.0, "S1", 0.5, True),
    ("Nitrile Gloves M (100)", "Infection control", "box", 30, 60, 6.9, "S3", 14, False),
    ("Nitrile Gloves S (100)", "Infection control", "box", 20, 40, 6.9, "S3", 9, False),
    ("Face Masks Type IIR (50)", "Infection control", "box", 15, 30, 7.5, "S3", 6, False),
    ("Surface Disinfectant (1L)", "Infection control", "bottle", 8, 16, 11.0, "S3", 3, True),
    ("Sterilisation Pouches (200)", "Infection control", "box", 6, 12, 14.0, "S3", 2, False),
    ("Saliva Ejectors (100)", "Disposables", "bag", 10, 20, 4.2, "S1", 4, False),
    ("Suction Tips (100)", "Disposables", "bag", 10, 20, 9.5, "S1", 3, False),
    ("Patient Bibs (500)", "Disposables", "box", 3, 6, 19.0, "S1", 1, False),
    ("Alginate Impression Powder (500g)", "Impression", "bag", 4, 8, 13.0, "S1", 1.5, True),
    ("Silicone Impression Material", "Impression", "cartridge", 6, 12, 24.0, "S2", 2, True),
    ("Endo Files K-Type (6)", "Endodontics", "pack", 8, 16, 9.0, "S2", 2, False),
    ("Gutta-percha Points", "Endodontics", "box", 4, 8, 16.0, "S2", 1, False),
    ("Sodium Hypochlorite 3% (500ml)", "Endodontics", "bottle", 3, 6, 8.5, "S2", 1, True),
    ("Prophy Paste (200 cups)", "Hygiene", "box", 3, 6, 29.0, "S5", 1, True),
    ("Airflow Powder", "Hygiene", "bottle", 4, 8, 34.0, "S5", 1.5, True),
    ("Whitening Gel Kit", "Cosmetic", "kit", 3, 6, 65.0, "S5", 0.7, True),
    ("Implant Fixture 4.0x10mm", "Implants", "piece", 4, 8, 145.0, "S4", 0.6, False),
    ("Healing Abutment", "Implants", "piece", 4, 8, 38.0, "S4", 0.6, False),
    ("Bone Graft Material (0.5g)", "Implants", "vial", 2, 4, 89.0, "S4", 0.3, True),
    ("Sutures 4-0 (12)", "Surgery", "box", 2, 4, 32.0, "S1", 0.5, True),
]

EQUIPMENT_CATALOG = [
    # name, type, location, interval_days, uses (treatment codes needing it)
    ("Dental Chair 1", "Chair", "Room 1", 180, None),
    ("Dental Chair 2", "Chair", "Room 2", 180, None),
    ("Dental Chair 3", "Chair", "Room 3", 180, None),
    ("Dental Chair 4", "Chair", "Hygiene room", 180, None),
    ("Intraoral X-ray", "Imaging", "Room 2", 365, ["ROOT_CANAL", "EXTRACTION", "EMERGENCY"]),
    ("CBCT / OPG Scanner", "Imaging", "Imaging room", 365, ["IMPLANT_CONSULT", "IMPLANT"]),
    ("Intraoral 3D Scanner", "Digital", "Mobile", 365, ["CROWN", "ALIGNERS"]),
    ("Autoclave A", "Sterilisation", "Sterilisation room", 90, None),
    ("Autoclave B", "Sterilisation", "Sterilisation room", 90, None),
    ("Air Compressor", "Utility", "Technical room", 180, None),
    ("Surgical Motor & Implant Unit", "Surgical", "Room 3", 365, ["IMPLANT"]),
    ("LED Curing Light (x4)", "Restorative", "All rooms", 365, ["FILLING"]),
]

LAB_TYPES = [("Zirconia crown", 180, 7), ("E.max veneer", 220, 8), ("3-unit bridge", 520, 10), ("Partial denture", 380, 12),
             ("Night guard", 90, 5), ("Clear aligner set", 650, 14), ("Implant abutment (custom)", 210, 9), ("Inlay / onlay", 160, 7)]
LABS = ["Zubná technika Košice", "DentLab Prešov", "CeramArt Laboratory"]
SHADES = ["A1", "A2", "A3", "A3.5", "B1", "B2", "C1", "D2"]

STORE = {}


# ==========================================
# SETTINGS
# ==========================================
def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings() -> dict:
    stored = {}
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                stored = json.load(f)
        except Exception as e:
            print("Could not read clinic_settings.json, using defaults:", e)
    return _deep_merge(DEFAULT_SETTINGS, stored)


def get_settings() -> dict:
    with LOCK:
        if "settings" not in STORE:
            STORE["settings"] = load_settings()
        return STORE["settings"]


def save_settings(new_settings: dict) -> dict:
    with LOCK:
        merged = _deep_merge(DEFAULT_SETTINGS, new_settings)
        STORE["settings"] = merged
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Could not persist settings:", e)
        return merged


def reset_settings() -> dict:
    with LOCK:
        if os.path.exists(SETTINGS_PATH):
            os.remove(SETTINGS_PATH)
        STORE["settings"] = copy.deepcopy(DEFAULT_SETTINGS)
        return STORE["settings"]


def treatment_map() -> dict:
    return {t["code"]: t for t in get_settings()["treatments"]}


def team_map() -> dict:
    return {u["id"]: u for u in get_settings()["team"]}


def day_hours(d: date):
    """(open, close) times for a date, or None when the clinic is closed."""
    s = get_settings()
    h = s["hours"].get(WEEKDAYS[d.weekday()])
    if not h or h.get("closed"):
        return None
    return datetime.strptime(h["open"], "%H:%M").time(), datetime.strptime(h["close"], "%H:%M").time()


def day_slots(d: date):
    hours = day_hours(d)
    if not hours:
        return []
    step = int(get_settings().get("slot_minutes", 30))
    cur, end = datetime.combine(d, hours[0]), datetime.combine(d, hours[1])
    slots = []
    while cur + timedelta(minutes=step) <= end:
        slots.append(cur.strftime("%H:%M"))
        cur += timedelta(minutes=step)
    return slots


def normalize_phone(raw: str, default_country: str = "421"):
    """
    Turns what people actually type into +<country><number>, or None if it can't be a phone number.
    Accepts spaces, dashes, dots, brackets, 00-prefix and Slovak local format:
    '0905 123 456' -> '+421905123456', '00421 905...' -> '+421905...', '+421 0905...' -> '+421905...'.
    """
    if not raw:
        return None
    s = "".join(ch for ch in str(raw).strip() if ch.isdigit() or ch == "+")
    if s.startswith("00"):
        s = "+" + s[2:]
    if not s.startswith("+"):
        s = "+" + default_country + (s[1:] if s.startswith("0") else s)
    if s.startswith("+" + default_country + "0"):
        s = "+" + default_country + s[len(default_country) + 2:]
    digits = s[1:]
    if not digits.isdigit() or not (9 <= len(digits) <= 15) or digits.startswith("0"):
        return None
    return s


def risk_level(prob: float) -> str:
    r = get_settings()["risk"]
    if prob >= r["critical"]:
        return "critical"
    if prob >= r["high"]:
        return "high"
    if prob >= r["moderate"]:
        return "moderate"
    return "low"


# ==========================================
# ML MODEL
# ==========================================
MODEL = {"model": None, "meta": None, "error": None}

FEATURE_TEXT = {
    "lead_days": lambda v: "Booked same day" if v == 0 else f"Booked {int(v)} days ahead",
    "same_day": lambda v: "Same-day booking" if v else "Not a same-day booking",
    "day_of_week": lambda v: f"{['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][int(v)]} appointment",
    "booked_hour": lambda v: f"Booked at {int(v):02d}:00",
    "age": lambda v: f"Age {int(v)}",
    "male": lambda v: "Male patient" if v else "Female patient",
    "sms_reminder": lambda v: "SMS reminder sent" if v else "No SMS reminder sent",
    "hypertension": lambda v: "Hypertension" if v else "No hypertension",
    "diabetes": lambda v: "Diabetes" if v else "No diabetes",
    "alcoholism": lambda v: "Alcohol use flagged" if v else "No alcohol flag",
    "handicap": lambda v: "Mobility / disability flag" if v else "No disability flag",
    "prev_appts": lambda v: "First visit" if v == 0 else f"{int(v)} previous visits",
    "prev_noshows": lambda v: f"{int(v)} previous no-show{'s' if v != 1 else ''}",
    "prev_noshow_rate": lambda v: "No visit history" if v < 0 else f"Past no-show rate {v:.0%}",
}


def load_model():
    if MODEL["model"] is not None or MODEL["error"]:
        return MODEL
    try:
        import joblib
        MODEL["model"] = joblib.load(MODEL_PATH)
        with open(MODEL_META_PATH, encoding="utf-8") as f:
            MODEL["meta"] = json.load(f)
    except Exception as e:
        MODEL["error"] = f"{type(e).__name__}: {e}"
        print("No-show model could not be loaded, using heuristic fallback:", MODEL["error"])
    return MODEL


def score_rows(rows):
    """rows: list of feature dicts -> list of (prob, factors)."""
    if not rows:
        return []
    m = load_model()
    if m["model"] is None:
        out = []
        for r in rows:
            p = 0.04 if r["same_day"] else min(0.6, 0.12 + 0.004 * min(r["lead_days"], 60) + 0.15 * max(r["prev_noshow_rate"], 0) - (0.03 if r["sms_reminder"] else 0))
            out.append((round(p, 4), [{"text": FEATURE_TEXT["lead_days"](r["lead_days"]), "effect": "up" if r["lead_days"] > 7 else "down"}]))
        return out

    import pandas as pd
    import xgboost as xgb
    feats = m["meta"]["features"]
    X = pd.DataFrame(rows)[feats].astype(float)
    probs = m["model"].predict_proba(X)[:, 1]
    contribs = m["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    out = []
    for i, p in enumerate(probs):
        c = sorted(((contribs[i][j], f) for j, f in enumerate(feats)), key=lambda t: -abs(t[0]))
        factors = []
        for val, f in c:
            if abs(val) < 0.05:
                continue
            factors.append({"text": FEATURE_TEXT[f](X.iloc[i][f]), "effect": "up" if val > 0 else "down", "weight": round(float(val), 3)})
            if len(factors) == 4:
                break
        out.append((round(float(p), 4), factors))
    return out


# ==========================================
# DEMO DATA GENERATION
# ==========================================
def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _weighted(rng, mapping):
    keys = list(mapping)
    return rng.choices(keys, weights=[mapping[k] for k in keys])[0]


def generate(seed: int = 42):
    rng = random.Random(seed)
    settings = get_settings()
    tmap = treatment_map()
    now = clinic_now()
    today = now.date()

    # ---------- Patients ----------
    patients = []
    window_start = today - timedelta(days=185)
    for i in range(1200):
        male = rng.random() < 0.46
        first = rng.choice(FIRST_M if male else FIRST_F)
        last_m, last_f = rng.choice(LAST)
        last = last_m if male else last_f
        age = int(min(88, max(6, rng.gauss(41, 17))))
        dob = today - timedelta(days=age * 365 + rng.randint(0, 364))
        propensity = min(0.9, rng.betavariate(1.1, 7))  # personal tendency to miss
        if rng.random() < 0.08:
            propensity = min(0.9, propensity + rng.uniform(0.25, 0.45))
        patients.append({
            "id": f"P-{1001 + i}",
            "first_name": first,
            "last_name": last,
            "name": f"{first} {last}",
            "gender": "M" if male else "F",
            "dob": dob.isoformat(),
            "age": age,
            "phone": "+4219" + "".join(str(rng.randint(0, 9)) for _ in range(8)),
            "email": f"{_ascii(first)}.{_ascii(last)}@{rng.choice(['gmail.com', 'azet.sk', 'centrum.sk', 'post.sk'])}",
            "city": rng.choice(CITIES),
            "insurer": rng.choice(INSURERS),
            "hypertension": int(rng.random() < (0.05 + age / 220)),
            "diabetes": int(rng.random() < (0.02 + age / 600)),
            "smoker": int(rng.random() < 0.22),
            "handicap": int(rng.random() < 0.02),
            "allergies": rng.choice(["None", "None", "None", "None", "Penicillin", "Latex", "Ibuprofen"]),
            "sms_opt_in": rng.random() < 0.86,
            "gdpr_consent": True,
            "registered": (today - timedelta(days=rng.randint(200, 2500)) if rng.random() < 0.8 else window_start + timedelta(days=rng.randint(0, 184))).isoformat(),
            "notes": "",
            "_propensity": propensity,
            "_weight": rng.choice([1, 1, 1, 2, 2, 3]),
        })

    # ---------- Appointments ----------
    appointments = []
    chairs = int(settings.get("chairs", 4))
    aid = 50001
    start_day = today - timedelta(days=185)
    for offset in range((today + timedelta(days=21) - start_day).days):
        d = start_day + timedelta(days=offset)
        slots = day_slots(d)
        if not slots:
            continue
        days_out = (d - today).days
        if days_out <= 0:
            fill = rng.uniform(0.75, 0.95)
        else:
            fill = max(0.08, 0.72 - 0.035 * days_out) * rng.uniform(0.85, 1.1)
        cells = [(s, c) for s in slots for c in range(1, chairs + 1)]
        n = int(len(cells) * fill * 0.58)  # most treatments take > 1 slot
        iso = d.isoformat()
        eligible = [p for p in patients if p["registered"] <= iso]
        eligible_w = [p["_weight"] for p in eligible]
        step = int(settings.get("slot_minutes", 30))
        slot_pos = {sl: i for i, sl in enumerate(slots)}
        chair_busy, provider_busy = set(), set()
        placed = 0
        rng.shuffle(cells)
        for slot, chair in cells:
            if placed >= n:
                break
            code = _weighted(rng, TREATMENT_WEIGHTS)
            t = tmap.get(code) or DEFAULT_TREATMENTS[0]
            provider = rng.choice(PROVIDER_FOR_CATEGORY.get(t["category"], ["U1"]))
            if code == "HYGIENE":
                chair = min(4, chairs)
            # No double-booking: the chair and the provider must be free for the whole treatment
            start_i = slot_pos[slot]
            needed = range(start_i, start_i + max(1, -(-t["duration"] // step)))
            if needed[-1] >= len(slots) or any((chair, i) in chair_busy or (provider, i) in provider_busy for i in needed):
                continue
            chair_busy.update((chair, i) for i in needed)
            provider_busy.update((provider, i) for i in needed)
            placed += 1
            patient = rng.choices(eligible, weights=eligible_w)[0]
            # Lead time: ~30% same day (emergencies, walk-ins), rest skewed
            if code == "EMERGENCY" or rng.random() < 0.22:
                lead = 0
            else:
                lead = int(min(90, rng.expovariate(1 / 14)) + 1)
            booked_at = datetime.combine(d - timedelta(days=lead), time(rng.randint(7, 19), rng.choice([0, 15, 30, 45])))
            if booked_at > now:
                booked_at = now - timedelta(hours=rng.randint(1, 48))
                lead = max(0, (d - booked_at.date()).days)
            sms = patient["sms_opt_in"] and lead >= 2 and rng.random() < 0.72
            fee = round(t["price"] * rng.choice([1, 1, 1, 1, 1.15, 1.3]) if code == "FILLING" else t["price"], 2)
            appointments.append({
                "id": f"A-{aid}",
                "patient_id": patient["id"],
                "date": d.isoformat(),
                "time": slot,
                "duration": t["duration"],
                "treatment": code,
                "provider_id": provider,
                "chair": chair,
                "fee": fee,
                "lead_days": lead,
                "booked_at": booked_at.isoformat(timespec="minutes"),
                "sms_reminder": int(sms),
                "status": "Upcoming",
                "rating": None,
                "prob": None,
                "factors": [],
            })
            aid += 1

    appointments.sort(key=lambda a: (a["date"], a["time"], a["chair"]))
    pmap = {p["id"]: p for p in patients}

    # Outcomes for past appointments
    for a in appointments:
        start = datetime.combine(date.fromisoformat(a["date"]), datetime.strptime(a["time"], "%H:%M").time())
        if start >= now:
            continue
        p = pmap[a["patient_id"]]
        if a["lead_days"] == 0:
            p_ns = 0.02 + p["_propensity"] * 0.15
        else:
            p_ns = 0.05 + p["_propensity"] * 0.7 + 0.004 * min(a["lead_days"], 60) - (0.05 if a["sms_reminder"] else 0)
            p_ns += 0.04 if p["age"] < 30 else 0
        r = rng.random()
        if r < 0.05:
            a["status"] = "Cancelled"
        elif r < 0.05 + max(0.01, min(p_ns, 0.85)):
            a["status"] = "No-show"
        else:
            a["status"] = "Completed"
            if rng.random() < 0.45:
                a["rating"] = rng.choices([5, 4, 3, 2, 1], weights=[62, 25, 8, 3, 2])[0]

    # ---------- Model scoring for upcoming appointments ----------
    history = {}
    upcoming = []
    for a in appointments:
        h = history.setdefault(a["patient_id"], [0, 0])
        if a["status"] == "Upcoming":
            upcoming.append((a, h[0], h[1]))
        elif a["status"] in ("Completed", "No-show"):
            h[0] += 1
            h[1] += a["status"] == "No-show"
    rows = []
    for a, prev, prev_ns in upcoming:
        p = pmap[a["patient_id"]]
        d = date.fromisoformat(a["date"])
        rows.append({
            "age": p["age"], "lead_days": a["lead_days"], "same_day": int(a["lead_days"] == 0),
            "day_of_week": d.weekday(), "booked_hour": int(a["booked_at"][11:13]), "male": int(p["gender"] == "M"),
            "sms_reminder": a["sms_reminder"], "hypertension": p["hypertension"], "diabetes": p["diabetes"],
            "alcoholism": 0, "handicap": p["handicap"], "prev_appts": prev, "prev_noshows": prev_ns,
            "prev_noshow_rate": (prev_ns / prev) if prev else -1.0,
        })
    for (a, _, _), (prob, factors) in zip(upcoming, score_rows(rows)):
        a["prob"] = prob
        a["factors"] = factors

    # ---------- Invoices ----------
    invoices = []
    seq = {}
    for a in appointments:
        if a["status"] != "Completed":
            continue
        d = date.fromisoformat(a["date"])
        seq[d.year] = seq.get(d.year, 1400) + 1
        insurance = round(a["fee"] * INSURANCE_SHARE.get(a["treatment"], 0), 2)
        age_days = (today - d).days
        if age_days > 90:
            status = "Paid" if rng.random() < 0.997 else "Overdue"
        elif age_days > 30:
            status = "Paid" if rng.random() < 0.985 else "Overdue"
        elif age_days > 14:
            status = "Paid" if rng.random() < 0.88 else "Unpaid"
        else:
            status = "Paid" if rng.random() < 0.78 else "Unpaid"
        patient_part = round(a["fee"] - insurance, 2)
        if patient_part <= 0:
            status = "Paid"
        invoices.append({
            "id": f"INV-{d.year}-{seq[d.year]:05d}",
            "patient_id": a["patient_id"],
            "appointment_id": a["id"],
            "date": a["date"],
            "due_date": (d + timedelta(days=14)).isoformat(),
            "items": [{"description": tmap.get(a["treatment"], {}).get("name", a["treatment"]), "amount": a["fee"]}],
            "amount": a["fee"],
            "insurance_amount": insurance,
            "patient_amount": patient_part,
            "status": status,
            "method": rng.choice(["Card", "Card", "Cash", "Bank transfer"]) if status == "Paid" else None,
            "paid_date": (d + timedelta(days=rng.choice([0, 0, 0, 2, 7]))).isoformat() if status == "Paid" else None,
        })

    # ---------- Treatment plans ----------
    plan_templates = [
        ["CROWN"], ["ROOT_CANAL", "CROWN"], ["IMPLANT_CONSULT", "IMPLANT", "CROWN"], ["FILLING", "FILLING", "HYGIENE"],
        ["HYGIENE", "WHITENING"], ["EXTRACTION", "IMPLANT", "CROWN"], ["FILLING", "FILLING", "FILLING"], ["ALIGNERS", "ALIGNERS", "WHITENING"],
    ]
    plans = []
    teeth = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]
    for i in range(48):
        patient = rng.choice(patients)
        codes = rng.choice(plan_templates)
        status = rng.choices(["Proposed", "Accepted", "In progress", "Completed", "Declined"], weights=[24, 20, 22, 22, 12])[0]
        created = today - timedelta(days=rng.randint(2, 150))
        items = []
        for k, code in enumerate(codes):
            t = tmap.get(code, DEFAULT_TREATMENTS[0])
            done = status == "Completed" or (status == "In progress" and k < max(1, len(codes) - 1))
            items.append({"code": code, "name": t["name"], "tooth": rng.choice(teeth) if code not in ("HYGIENE", "WHITENING", "ALIGNERS") else None,
                          "price": t["price"], "done": done})
        provider = rng.choice(PROVIDER_FOR_CATEGORY.get(tmap.get(codes[-1], DEFAULT_TREATMENTS[0])["category"], ["U1"]))
        plans.append({
            "id": f"TP-{3001 + i}",
            "patient_id": patient["id"],
            "provider_id": provider,
            "title": " + ".join(dict.fromkeys(it["name"] for it in items)),
            "created": created.isoformat(),
            "status": status,
            "items": items,
            "total": round(sum(it["price"] for it in items), 2),
            "notes": "",
        })

    # ---------- Stock ----------
    stock = []
    for i, (name, cat, unit, mn, reorder, cost, sup, weekly, expires) in enumerate(STOCK_CATALOG):
        r = rng.random()
        qty = rng.randint(0, mn - 1) if r < 0.22 else rng.randint(mn, mn * 3)
        stock.append({
            "id": f"ST-{101 + i}", "name": name, "category": cat, "unit": unit, "qty": qty, "min_qty": mn,
            "reorder_qty": reorder, "unit_cost": cost, "supplier_id": sup, "weekly_usage": weekly,
            "expiry": (today + timedelta(days=rng.randint(20, 540))).isoformat() if expires else None,
            "location": rng.choice(["Store room A", "Store room B", "Sterilisation room", "Surgery cabinet"]),
        })

    # ---------- Purchase orders ----------
    purchase_orders = []
    for i in range(14):
        sup = rng.choice(SUPPLIERS)
        sup_items = [s for s in stock if s["supplier_id"] == sup["id"]]
        lines = [{"stock_id": s["id"], "name": s["name"], "qty": s["reorder_qty"], "unit_cost": s["unit_cost"]}
                 for s in rng.sample(sup_items, min(len(sup_items), rng.randint(1, 4)))]
        created = today - timedelta(days=rng.randint(0, 120))
        age = (today - created).days
        status = "Received" if age > sup["lead_days"] + 2 else rng.choice(["Ordered", "Ordered", "Draft"])
        purchase_orders.append({
            "id": f"PO-{created.year}-{701 + i}", "supplier_id": sup["id"], "created": created.isoformat(),
            "expected": (created + timedelta(days=sup["lead_days"])).isoformat(), "status": status,
            "received_date": (created + timedelta(days=sup["lead_days"])).isoformat() if status == "Received" else None,
            "lines": lines, "total": round(sum(l["qty"] * l["unit_cost"] for l in lines), 2), "notes": "",
        })
    purchase_orders.sort(key=lambda p: p["created"], reverse=True)

    # ---------- Equipment ----------
    equipment = []
    for i, (name, typ, loc, interval, uses) in enumerate(EQUIPMENT_CATALOG):
        last = today - timedelta(days=rng.randint(5, interval + 20))
        status = "Operational"
        if name.startswith("CBCT") and rng.random() < 0.5:
            status = "Maintenance"
        equipment.append({
            "id": f"EQ-{i + 1:02d}", "name": name, "type": typ, "location": loc,
            "serial": f"SN-{rng.randint(100000, 999999)}", "status": status,
            "installed": (today - timedelta(days=rng.randint(400, 2600))).isoformat(),
            "last_service": last.isoformat(), "service_interval_days": interval,
            "next_service": (last + timedelta(days=interval)).isoformat(), "uses": uses or [],
            "chair": int(name[-1]) if typ == "Chair" else None, "service_log": [{"date": last.isoformat(), "note": "Scheduled service"}],
        })

    # ---------- Lab orders ----------
    lab_orders = []
    crown_patients = [pl for pl in plans if any(it["code"] in ("CROWN", "IMPLANT") for it in pl["items"]) and pl["status"] in ("Accepted", "In progress", "Completed")]
    for i in range(26):
        lt, cost, turnaround = rng.choice(LAB_TYPES)
        plan = crown_patients[i % len(crown_patients)] if crown_patients and rng.random() < 0.7 else None
        patient_id = plan["patient_id"] if plan else rng.choice(patients)["id"]
        sent = today - timedelta(days=rng.randint(0, 45))
        due = sent + timedelta(days=turnaround)
        if due < today - timedelta(days=5):
            status = rng.choices(["Fitted", "Received", "Remake"], weights=[75, 20, 5])[0]
        elif due < today:
            status = rng.choices(["Received", "Shipped", "In production"], weights=[50, 25, 25])[0]
        else:
            status = rng.choice(["Sent", "In production", "In production", "Shipped"])
        lab_orders.append({
            "id": f"LAB-{4001 + i}", "patient_id": patient_id, "plan_id": plan["id"] if plan else None,
            "provider_id": plan["provider_id"] if plan else "U2", "type": lt, "lab": rng.choice(LABS),
            "tooth": rng.choice(teeth) if "aligner" not in lt.lower() and "guard" not in lt.lower() else None,
            "shade": rng.choice(SHADES) if "crown" in lt.lower() or "veneer" in lt.lower() or "bridge" in lt.lower() else None,
            "sent_date": sent.isoformat(), "due_date": due.isoformat(), "status": status, "cost": cost, "notes": "",
        })
    lab_orders.sort(key=lambda l: l["sent_date"], reverse=True)

    for p in patients:
        p.pop("_weight", None)

    with LOCK:
        STORE.update({
            "generated_at": now.isoformat(timespec="seconds"),
            "patients": patients, "appointments": appointments, "invoices": invoices, "plans": plans,
            "stock": stock, "suppliers": copy.deepcopy(SUPPLIERS), "purchase_orders": purchase_orders,
            "equipment": equipment, "lab_orders": lab_orders,
        })
    print(f"Demo clinic generated: {len(patients)} patients, {len(appointments)} appointments, {len(invoices)} invoices")


def ensure_data():
    with LOCK:
        if "patients" not in STORE:
            generate()
    return STORE
