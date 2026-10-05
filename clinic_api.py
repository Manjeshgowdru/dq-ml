"""
DentalIQ clinic module API: patients, AI predictions, financials, stock, procurement,
equipment, treatment plans, billing, lab orders, reports, dashboard and settings.
"""
import os
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import clinic_data as cd

router = APIRouter(prefix="/api")

PAST_STATUSES = ("Completed", "No-show")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FIXED_MONTHLY_EXPENSES = {"Salaries": 18500, "Rent": 2400, "Utilities": 650, "Software & IT": 320, "Insurance": 260, "Marketing": 0}


# ==========================================
# HELPERS
# ==========================================
def store():
    return cd.ensure_data()


def today() -> date:
    return cd.clinic_now().date()


def find(collection: str, item_id: str) -> dict:
    for item in store()[collection]:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"{item_id} not found")


def patient_names():
    return {p["id"]: p["name"] for p in store()["patients"]}


def enrich_appointment(a: dict, names=None, tmap=None, team=None) -> dict:
    names = names or patient_names()
    tmap = tmap or cd.treatment_map()
    team = team or cd.team_map()
    t = tmap.get(a["treatment"], {})
    out = {k: v for k, v in a.items()}
    out["patient"] = names.get(a["patient_id"], a["patient_id"])
    out["treatment_name"] = t.get("name", a["treatment"])
    out["category"] = t.get("category", "Other")
    out["provider"] = team.get(a["provider_id"], {}).get("name", a["provider_id"])
    if a.get("prob") is not None:
        level = cd.risk_level(a["prob"])
        out["risk_level"] = level
        out["action"] = cd.get_settings()["risk"]["actions"][level]
    return out


def month_key(iso: str) -> str:
    return iso[:7]


def month_label(key: str) -> str:
    y, m = key.split("-")
    return f"{MONTHS[int(m) - 1]} {y[2:]}"


def last_months(n: int):
    t = today().replace(day=1)
    keys = []
    for _ in range(n):
        keys.append(t.strftime("%Y-%m"))
        t = (t - timedelta(days=1)).replace(day=1)
    return list(reversed(keys))


def working_days_in_month(key: str) -> int:
    y, m = map(int, key.split("-"))
    d = date(y, m, 1)
    count = 0
    while d.month == m:
        count += cd.day_hours(d) is not None
        d += timedelta(days=1)
    return max(count, 1)


def workdays_elapsed(key: str) -> int:
    """Open clinic days in month `key` up to and including today (whole month for past months)."""
    y, m = map(int, key.split("-"))
    d, t, count = date(y, m, 1), today(), 0
    while d.month == m and d <= t:
        count += cd.day_hours(d) is not None
        d += timedelta(days=1)
    return max(count, 1)


def revenue_pace(month_rev: float, prev_rev: float) -> float:
    """Current month revenue per working day vs last month's, as a ratio."""
    if not prev_rev:
        return 1.0
    cur_key, prev_key = last_months(2)[1], last_months(2)[0]
    return (month_rev / workdays_elapsed(cur_key)) / (prev_rev / working_days_in_month(prev_key))


def expenses_by_month(keys, prorate=True):
    s = store()
    lab = defaultdict(float)
    for l in s["lab_orders"]:
        lab[month_key(l["sent_date"])] += l["cost"]
    supplies = defaultdict(float)
    for po in s["purchase_orders"]:
        if po["status"] == "Received" and po["received_date"]:
            supplies[month_key(po["received_date"])] += po["total"]
    result = {}
    for k in keys:
        items = dict(FIXED_MONTHLY_EXPENSES)
        current = prorate and k == today().strftime("%Y-%m")
        if current:  # month to date: prorate fixed costs
            frac = min(1.0, today().day / 30)
            items = {name: round(v * frac, 2) for name, v in items.items()}
        items["Lab fees"] = round(lab.get(k, 0), 2)
        items["Supplies"] = round(supplies.get(k, 0) or 1450 * (min(1.0, today().day / 30) if current else 1), 2)
        result[k] = items
    return result


def upcoming_appointments(days: int):
    now = cd.clinic_now()
    end = (now + timedelta(days=days)).date()
    out = []
    for a in store()["appointments"]:
        if a["status"] != "Upcoming":
            continue
        d = date.fromisoformat(a["date"])
        if d > end:
            continue
        out.append(a)
    return out


# ==========================================
# SYSTEM / SETTINGS
# ==========================================
@router.get("/system/status")
def system_status():
    m = cd.load_model()
    s = store()
    calendar_configured = bool(os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")) or os.path.exists(os.path.join(cd.HERE, "service_account.json"))
    return {
        "api": "online",
        "data_mode": "demo",
        "data_generated_at": s["generated_at"],
        "clinic_time": cd.clinic_now().isoformat(timespec="minutes"),
        "integrations": {
            "google_calendar": {"configured": calendar_configured, "calendar_id": os.environ.get("GOOGLE_CALENDAR_ID", "primary"),
                                "auth_mode": "cloud_env" if os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON") else "local_file"},
            "gemini_voice": {"configured": bool(os.environ.get("GEMINI_API_KEY"))},
            "whatsapp": {"configured": True, "mode": "click-to-chat"},
            "database": {"configured": False, "planned": "Supabase (EU)"},
        },
        "model": {"loaded": m["model"] is not None, "error": m["error"], "meta": m["meta"]},
        "counts": {k: len(s[k]) for k in ("patients", "appointments", "invoices", "plans", "stock", "purchase_orders", "equipment", "lab_orders")},
    }


@router.get("/settings")
def read_settings():
    return cd.get_settings()


@router.put("/settings")
def update_settings(payload: dict):
    r = payload.get("risk", {})
    if r and not (0 < r.get("moderate", 0.2) < r.get("high", 0.32) < r.get("critical", 0.45) < 1):
        raise HTTPException(status_code=400, detail="Risk thresholds must increase: moderate < high < critical (between 0 and 1).")
    slot = payload.get("slot_minutes", 30)
    if slot not in (15, 20, 30, 45, 60):
        raise HTTPException(status_code=400, detail="Slot length must be 15, 20, 30, 45 or 60 minutes.")
    for day, h in (payload.get("hours") or {}).items():
        if not h.get("closed") and h.get("open", "08:00") >= h.get("close", "17:00"):
            raise HTTPException(status_code=400, detail=f"Opening time must be before closing time ({day}).")
    return cd.save_settings(payload)


@router.post("/settings/reset")
def reset_settings():
    return cd.reset_settings()


@router.post("/demo/regenerate")
def regenerate_demo():
    cd.generate()
    return {"status": "success", "generated_at": store()["generated_at"]}


# ==========================================
# DASHBOARD
# ==========================================
@router.get("/dashboard")
def dashboard():
    s = store()
    t = today()
    t_iso = t.isoformat()
    names, tmap, team = patient_names(), cd.treatment_map(), cd.team_map()
    settings = cd.get_settings()

    todays = [a for a in s["appointments"] if a["date"] == t_iso]
    yday = t - timedelta(days=1)
    while cd.day_hours(yday) is None and (t - yday).days < 7:
        yday -= timedelta(days=1)
    rev_by_day = defaultdict(float)
    for inv in s["invoices"]:
        rev_by_day[inv["date"]] += inv["amount"]
    today_rev = rev_by_day.get(t_iso, 0) + sum(a["fee"] for a in todays if a["status"] == "Upcoming")
    y_rev = rev_by_day.get(yday.isoformat(), 0)

    upcoming_today = [a for a in todays if a["status"] == "Upcoming"]
    risky_today = [a for a in upcoming_today if a["prob"] is not None and cd.risk_level(a["prob"]) in ("high", "critical")]
    expected_noshows = sum(a["prob"] or 0 for a in upcoming_today)

    plans = s["plans"]
    decided = [p for p in plans if p["status"] != "Proposed"]
    accepted = [p for p in decided if p["status"] in ("Accepted", "In progress", "Completed")]
    acceptance = len(accepted) / len(decided) if decided else 0

    slots_today = cd.day_slots(t)
    capacity = len(slots_today) * int(settings.get("chairs", 4))
    used = sum(max(1, -(-a["duration"] // int(settings.get("slot_minutes", 30)))) for a in todays if a["status"] != "Cancelled")
    utilization = min(1.0, used / capacity) if capacity else 0

    mkey = t.strftime("%Y-%m")
    month_rev = sum(inv["amount"] for inv in s["invoices"] if month_key(inv["date"]) == mkey)
    month_exp = sum(expenses_by_month([mkey])[mkey].values())

    # 7-day revenue chart (working days)
    series = []
    d = t
    while len(series) < 7 and (t - d).days < 14:
        if cd.day_hours(d) is not None:
            key = d.isoformat()
            exp_day = round(sum(expenses_by_month([month_key(key)], prorate=False)[month_key(key)].values()) / working_days_in_month(month_key(key)))
            revenue = round(rev_by_day.get(key, 0) + (sum(a["fee"] for a in todays if a["status"] == "Upcoming") if key == t_iso else 0))
            series.append({"date": f"{d.day} {MONTHS[d.month - 1]}", "revenue": revenue, "expenses": exp_day, "profit": revenue - exp_day})
        d -= timedelta(days=1)
    series.reverse()

    cat = defaultdict(float)
    for inv in s["invoices"]:
        if month_key(inv["date"]) == mkey:
            a_code = inv["items"][0]["description"]
            cat[a_code] += inv["amount"]
    by_category = defaultdict(float)
    for a in s["appointments"]:
        if a["status"] == "Completed" and month_key(a["date"]) == mkey:
            by_category[tmap.get(a["treatment"], {}).get("category", "Other")] += a["fee"]

    hours = defaultdict(int)
    for a in todays:
        if a["status"] != "Cancelled":
            hours[a["time"][:2]] += 1
    per_hour_capacity = int(settings.get("chairs", 4)) * (60 // int(settings.get("slot_minutes", 30)))
    chair_by_hour = [{"hour": f"{h}:00", "util": round(min(100, hours.get(h, 0) * 2 / per_hour_capacity * 100))}
                     for h in sorted({sl[:2] for sl in slots_today})]

    week_upcoming = upcoming_appointments(7)
    at_risk = sum((a["prob"] or 0) * a["fee"] for a in week_upcoming)
    expected_in = sum((1 - (a["prob"] or 0)) * a["fee"] for a in week_upcoming)
    risk_series, cash_series = [], []
    for i in range(7):
        dd = (t + timedelta(days=i)).isoformat()
        day_apps = [a for a in week_upcoming if a["date"] == dd]
        risk_series.append({"day": dd, "value": round(sum((a["prob"] or 0) * a["fee"] for a in day_apps))})
        cash_series.append({"day": dd, "value": round(sum((1 - (a["prob"] or 0)) * a["fee"] for a in day_apps))})

    top_risk = sorted([a for a in week_upcoming if a["prob"] is not None], key=lambda a: -a["prob"])[:5]
    low_stock = sorted([x for x in s["stock"] if x["qty"] < x["min_qty"]], key=lambda x: x["qty"] / x["min_qty"])[:5]

    proc = defaultdict(lambda: [0, 0.0])
    for a in s["appointments"]:
        if a["status"] == "Completed" and month_key(a["date"]) == mkey:
            proc[a["treatment"]][0] += 1
            proc[a["treatment"]][1] += a["fee"]
    top_procs = sorted(([tmap.get(k, {}).get("name", k), v[0], round(v[1])] for k, v in proc.items()), key=lambda r: -r[2])[:5]

    past90 = [a for a in s["appointments"] if a["status"] in PAST_STATUSES and (t - date.fromisoformat(a["date"])).days <= 90]
    noshow_rate = sum(a["status"] == "No-show" for a in past90) / len(past90) if past90 else 0
    patients_seen = Counter(a["patient_id"] for a in s["appointments"] if a["status"] == "Completed" and (t - date.fromisoformat(a["date"])).days <= 365)
    retention = sum(1 for c in patients_seen.values() if c >= 2) / len(patients_seen) if patients_seen else 0
    prev_key = last_months(2)[0]
    prev_rev = sum(inv["amount"] for inv in s["invoices"] if month_key(inv["date"]) == prev_key)
    growth = revenue_pace(month_rev, prev_rev)
    health = {
        "Revenue growth": round(max(0, min(100, 60 + (growth - 1) * 100))),
        "Patient retention": round(retention * 100),
        "Chair efficiency": round(utilization * 100),
        "No-show control": round(max(0, 100 - noshow_rate * 300)),
    }
    score = round(sum(health.values()) / len(health))

    overdue_labs = [l for l in s["lab_orders"] if l["due_date"] < t_iso and l["status"] in ("Sent", "In production", "Shipped")]
    unpaid = [i for i in s["invoices"] if i["status"] != "Paid"]
    insights = []
    if risky_today:
        insights.append(f"{len(risky_today)} high-risk appointment(s) today — send reminders now to protect €{round(sum(a['fee'] for a in risky_today))}.")
    if low_stock:
        insights.append(f"{len(low_stock)} stock item(s) below minimum — create a reorder in Procurement.")
    if overdue_labs:
        insights.append(f"{len(overdue_labs)} lab order(s) overdue — contact the lab before the fitting appointments.")
    proposed = [p for p in plans if p["status"] == "Proposed"]
    if proposed:
        insights.append(f"{len(proposed)} treatment plans worth €{round(sum(p['total'] for p in proposed)):,} are waiting for patient acceptance.")

    return {
        "date": t_iso,
        "clinic": settings["clinic"]["name"],
        "kpis": {
            "today_revenue": round(today_rev), "today_revenue_change": round((today_rev / y_rev - 1) * 100, 1) if y_rev else None,
            "appointments_today": len([a for a in todays if a["status"] != "Cancelled"]),
            "completed_today": len([a for a in todays if a["status"] == "Completed"]),
            "upcoming_today": len(upcoming_today),
            "high_risk_today": len(risky_today), "expected_noshows_today": round(expected_noshows, 1),
            "treatment_acceptance": round(acceptance * 100, 1), "chair_utilization": round(utilization * 100, 1),
            "net_profit_mtd": round(month_rev - month_exp), "revenue_mtd": round(month_rev),
        },
        "revenue_7d": series,
        "by_category": [{"name": k, "value": round(v)} for k, v in sorted(by_category.items(), key=lambda kv: -kv[1])],
        "chair_by_hour": chair_by_hour,
        "risk_patients": [enrich_appointment(a, names, tmap, team) for a in top_risk],
        "todays_appointments": [enrich_appointment(a, names, tmap, team) for a in sorted(todays, key=lambda a: a["time"])],
        "low_stock": low_stock,
        "equipment": s["equipment"][:6],
        "top_procedures": [{"proc": r[0], "count": r[1], "revenue": r[2]} for r in top_procs],
        "recent_invoices": [{**i, "patient": names.get(i["patient_id"])} for i in sorted(s["invoices"], key=lambda i: i["date"], reverse=True)[:5]],
        "cash_forecast_7d": {"total": round(expected_in), "series": cash_series},
        "revenue_at_risk_7d": {"total": round(at_risk), "series": risk_series},
        "health": {"score": score, "parts": health},
        "insights": insights,
        "unpaid_total": round(sum(i["patient_amount"] for i in unpaid)),
    }


# ==========================================
# PATIENTS
# ==========================================
class PatientIn(BaseModel):
    first_name: str
    last_name: str
    phone: str
    email: Optional[str] = ""
    gender: Optional[str] = "F"
    dob: Optional[str] = None
    city: Optional[str] = "Košice"
    insurer: Optional[str] = "VšZP"
    allergies: Optional[str] = "None"
    hypertension: Optional[int] = 0
    diabetes: Optional[int] = 0
    smoker: Optional[int] = 0
    sms_opt_in: Optional[bool] = True
    notes: Optional[str] = ""


def patient_summary(p, appts_by_patient, invoices_by_patient):
    appts = appts_by_patient.get(p["id"], [])
    past = [a for a in appts if a["status"] in PAST_STATUSES]
    noshows = sum(a["status"] == "No-show" for a in past)
    upcoming = sorted([a for a in appts if a["status"] == "Upcoming"], key=lambda a: (a["date"], a["time"]))
    last_visit = max((a["date"] for a in appts if a["status"] == "Completed"), default=None)
    balance = sum(i["patient_amount"] for i in invoices_by_patient.get(p["id"], []) if i["status"] != "Paid")
    nxt = upcoming[0] if upcoming else None
    # Next visit: model risk. No visit booked: flag only patients with a real history of missing (3+ visits, 30%+ missed)
    risk = cd.risk_level(nxt["prob"]) if nxt and nxt["prob"] is not None else ("high" if len(past) >= 3 and noshows / len(past) >= 0.3 else "low")
    return {
        **{k: v for k, v in p.items() if not k.startswith("_")},
        "visits": len(past), "noshows": noshows, "noshow_rate": round(noshows / len(past), 3) if past else 0,
        "last_visit": last_visit, "next_appointment": {"date": nxt["date"], "time": nxt["time"], "prob": nxt["prob"]} if nxt else None,
        "balance": round(balance, 2), "risk_level": risk,
        "lifetime_value": round(sum(i["amount"] for i in invoices_by_patient.get(p["id"], [])), 2),
    }


@router.get("/patients")
def list_patients(q: Optional[str] = None, risk: Optional[str] = None):
    s = store()
    by_p = defaultdict(list)
    for a in s["appointments"]:
        by_p[a["patient_id"]].append(a)
    inv_p = defaultdict(list)
    for i in s["invoices"]:
        inv_p[i["patient_id"]].append(i)
    rows = [patient_summary(p, by_p, inv_p) for p in s["patients"]]
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in r["name"].lower() or ql in r["phone"] or ql in r["id"].lower() or ql in (r["email"] or "").lower()]
    if risk:
        rows = [r for r in rows if r["risk_level"] == risk]
    rows.sort(key=lambda r: r["last_name"])
    total = len(s["patients"])
    new_90 = sum(1 for p in s["patients"] if (today() - date.fromisoformat(p["registered"])).days <= 90)
    return {
        "patients": rows,
        "summary": {
            "total": total, "new_90d": new_90,
            "with_upcoming": sum(1 for r in rows if r["next_appointment"]),
            "outstanding_balance": round(sum(r["balance"] for r in rows), 2),
            "high_risk": sum(1 for r in rows if r["risk_level"] in ("high", "critical")),
        },
    }


@router.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    s = store()
    p = find("patients", patient_id)
    names, tmap, team = patient_names(), cd.treatment_map(), cd.team_map()
    appts = [a for a in s["appointments"] if a["patient_id"] == patient_id]
    invs = [i for i in s["invoices"] if i["patient_id"] == patient_id]
    by_p, inv_p = {patient_id: appts}, {patient_id: invs}
    return {
        "patient": patient_summary(p, by_p, inv_p),
        "appointments": [enrich_appointment(a, names, tmap, team) for a in sorted(appts, key=lambda a: (a["date"], a["time"]), reverse=True)],
        "invoices": sorted(invs, key=lambda i: i["date"], reverse=True),
        "plans": [pl for pl in s["plans"] if pl["patient_id"] == patient_id],
        "lab_orders": [l for l in s["lab_orders"] if l["patient_id"] == patient_id],
    }


@router.post("/patients")
def create_patient(body: PatientIn):
    s = store()
    with cd.LOCK:
        next_id = max(int(p["id"].split("-")[1]) for p in s["patients"]) + 1
        dob = body.dob or (today() - timedelta(days=35 * 365)).isoformat()
        age = (today() - date.fromisoformat(dob)).days // 365
        p = {**body.model_dump(), "id": f"P-{next_id}", "name": f"{body.first_name.strip()} {body.last_name.strip()}",
             "dob": dob, "age": age, "handicap": 0, "gdpr_consent": True, "registered": today().isoformat(), "_propensity": 0.1}
        s["patients"].append(p)
    return {"status": "success", "patient": {k: v for k, v in p.items() if not k.startswith("_")}}


@router.patch("/patients/{patient_id}")
def update_patient(patient_id: str, body: dict):
    p = find("patients", patient_id)
    allowed = set(PatientIn.model_fields)
    with cd.LOCK:
        for k, v in body.items():
            if k in allowed:
                p[k] = v
        p["name"] = f"{p['first_name']} {p['last_name']}"
    return {"status": "success", "patient": {k: v for k, v in p.items() if not k.startswith("_")}}


# ==========================================
# AI PREDICTIONS
# ==========================================
@router.get("/predictions")
def predictions(days: int = Query(7, ge=1, le=21)):
    names, tmap, team = patient_names(), cd.treatment_map(), cd.team_map()
    pmap = {p["id"]: p for p in store()["patients"]}
    rows = []
    for a in upcoming_appointments(days):
        if a["prob"] is None:
            continue
        e = enrich_appointment(a, names, tmap, team)
        e["phone"] = pmap[a["patient_id"]]["phone"]
        e["revenue_at_risk"] = round(a["prob"] * a["fee"], 2)
        rows.append(e)
    rows.sort(key=lambda r: (-r["prob"]))
    levels = Counter(r["risk_level"] for r in rows)
    by_day = defaultdict(lambda: {"appointments": 0, "expected_noshows": 0.0, "at_risk": 0.0})
    for r in rows:
        d = by_day[r["date"]]
        d["appointments"] += 1
        d["expected_noshows"] += r["prob"]
        d["at_risk"] += r["revenue_at_risk"]
    m = cd.load_model()

    # Historical accuracy: how did no-show rates differ by lead time in the clinic's own history
    past = [a for a in store()["appointments"] if a["status"] in PAST_STATUSES]
    buckets = [("Same day", 0, 0), ("1–7 days", 1, 7), ("8–30 days", 8, 30), ("31+ days", 31, 999)]
    by_lead = []
    for label, lo, hi in buckets:
        grp = [a for a in past if lo <= a["lead_days"] <= hi]
        by_lead.append({"bucket": label, "rate": round(sum(a["status"] == "No-show" for a in grp) / len(grp) * 100, 1) if grp else 0, "count": len(grp)})

    return {
        "days": days,
        "appointments": rows,
        "summary": {
            "total": len(rows), "levels": {k: levels.get(k, 0) for k in ("critical", "high", "moderate", "low")},
            "expected_noshows": round(sum(r["prob"] for r in rows), 1),
            "revenue_at_risk": round(sum(r["revenue_at_risk"] for r in rows), 2),
            "scheduled_revenue": round(sum(r["fee"] for r in rows), 2),
        },
        "by_day": [{"date": k, **{kk: round(vv, 2) for kk, vv in v.items()}} for k, v in sorted(by_day.items())],
        "history_by_lead": by_lead,
        "thresholds": cd.get_settings()["risk"],
        "model": {"loaded": m["model"] is not None, "meta": m["meta"], "error": m["error"]},
    }


class ScoreRequest(BaseModel):
    date: str
    patient_id: Optional[str] = None
    age: Optional[int] = 40
    gender: Optional[str] = "F"
    sms_reminder: Optional[int] = 0


@router.post("/predictions/score")
def score_booking(req: ScoreRequest):
    """Scores a single (new) booking. Used by booking flows and the voice receptionist."""
    d = date.fromisoformat(req.date)
    now = cd.clinic_now()
    prev = prev_ns = 0
    age, male = req.age or 40, int((req.gender or "F") == "M")
    if req.patient_id:
        p = find("patients", req.patient_id)
        age, male = p["age"], int(p["gender"] == "M")
        past = [a for a in store()["appointments"] if a["patient_id"] == req.patient_id and a["status"] in PAST_STATUSES]
        prev, prev_ns = len(past), sum(a["status"] == "No-show" for a in past)
    lead = max(0, (d - now.date()).days)
    row = {"age": age, "lead_days": lead, "same_day": int(lead == 0), "day_of_week": d.weekday(), "booked_hour": now.hour,
           "male": male, "sms_reminder": req.sms_reminder or 0, "hypertension": 0, "diabetes": 0, "alcoholism": 0, "handicap": 0,
           "prev_appts": prev, "prev_noshows": prev_ns, "prev_noshow_rate": prev_ns / prev if prev else -1.0}
    prob, factors = cd.score_rows([row])[0]
    level = cd.risk_level(prob)
    return {"prob": prob, "risk_level": level, "action": cd.get_settings()["risk"]["actions"][level], "factors": factors}


@router.post("/appointments/{appointment_id}/reminder")
def mark_reminder(appointment_id: str):
    a = find("appointments", appointment_id)
    with cd.LOCK:
        a["sms_reminder"] = 1
        a["reminded_at"] = cd.clinic_now().isoformat(timespec="minutes")
    return {"status": "success", "appointment": a}


# ==========================================
# FINANCIAL INTELLIGENCE
# ==========================================
@router.get("/financial")
def financial(months: int = Query(6, ge=3, le=12)):
    s = store()
    keys = last_months(months)
    tmap, team = cd.treatment_map(), cd.team_map()
    expenses = expenses_by_month(keys)
    invoiced, collected, insurance = defaultdict(float), defaultdict(float), defaultdict(float)
    for i in s["invoices"]:
        k = month_key(i["date"])
        invoiced[k] += i["amount"]
        insurance[k] += i["insurance_amount"]
        if i["status"] == "Paid":
            collected[k] += i["amount"]
        else:
            collected[k] += i["insurance_amount"]
    monthly = []
    for k in keys:
        exp = sum(expenses[k].values())
        monthly.append({"month": month_label(k), "key": k, "revenue": round(invoiced[k]), "collected": round(collected[k]),
                        "insurance": round(insurance[k]), "expenses": round(exp), "profit": round(invoiced[k] - exp),
                        "margin": round((invoiced[k] - exp) / invoiced[k] * 100, 1) if invoiced[k] else 0})

    cur, prev = monthly[-1], monthly[-2]
    by_cat, by_provider = defaultdict(float), defaultdict(lambda: {"revenue": 0.0, "visits": 0})
    period_start = keys[0] + "-01"
    for a in s["appointments"]:
        if a["status"] == "Completed" and a["date"] >= period_start:
            by_cat[tmap.get(a["treatment"], {}).get("category", "Other")] += a["fee"]
            pr = by_provider[team.get(a["provider_id"], {}).get("name", a["provider_id"])]
            pr["revenue"] += a["fee"]
            pr["visits"] += 1

    t = today()
    aging = {"0–30 days": 0.0, "31–60 days": 0.0, "61–90 days": 0.0, "90+ days": 0.0}
    for i in s["invoices"]:
        if i["status"] == "Paid":
            continue
        age = (t - date.fromisoformat(i["date"])).days
        bucket = "0–30 days" if age <= 30 else "31–60 days" if age <= 60 else "61–90 days" if age <= 90 else "90+ days"
        aging[bucket] += i["patient_amount"]

    upcoming30 = upcoming_appointments(30)
    forecast = sum(a["fee"] for a in upcoming30)
    at_risk = sum((a["prob"] or 0) * a["fee"] for a in upcoming30)
    total_inv = sum(m["revenue"] for m in monthly)
    total_col = sum(m["collected"] for m in monthly)
    completed = [a for a in s["appointments"] if a["status"] == "Completed" and a["date"] >= period_start]
    noshow_lost = sum(a["fee"] for a in s["appointments"] if a["status"] == "No-show" and a["date"] >= period_start)

    insights = []
    pace_ratio = revenue_pace(cur["revenue"], prev["revenue"])
    if prev["revenue"]:
        insights.append(f"Revenue per working day is {(pace_ratio - 1) * 100:+.0f}% vs last month (projected month: €{round(prev['revenue'] * pace_ratio):,} vs €{prev['revenue']:,}).")
    insights.append(f"No-shows cost €{round(noshow_lost):,} in the last {months} months — AI reminders target the riskiest {len([a for a in upcoming30 if a['prob'] and cd.risk_level(a['prob']) in ('high', 'critical')])} upcoming visits.")
    top_cat = max(by_cat.items(), key=lambda kv: kv[1]) if by_cat else None
    if top_cat:
        insights.append(f"{top_cat[0]} is the largest revenue category ({top_cat[1] / max(1, sum(by_cat.values())) * 100:.0f}% of the period).")
    if sum(aging.values()) > 0:
        insights.append(f"€{round(sum(aging.values())):,} outstanding from patients; €{round(aging['31–60 days'] + aging['61–90 days'] + aging['90+ days']):,} is older than 30 days.")

    return {
        "monthly": monthly,
        "kpis": {
            "revenue_mtd": cur["revenue"], "revenue_prev_month": prev["revenue"], "expenses_mtd": cur["expenses"],
            "pace_change": round((pace_ratio - 1) * 100, 1),
            "profit_mtd": cur["profit"], "margin_mtd": cur["margin"],
            "collection_rate": round(total_col / total_inv * 100, 1) if total_inv else 0,
            "avg_revenue_per_visit": round(sum(a["fee"] for a in completed) / len(completed), 2) if completed else 0,
            "outstanding": round(sum(aging.values()), 2),
            "forecast_30d": round(forecast), "at_risk_30d": round(at_risk), "noshow_lost_period": round(noshow_lost),
        },
        "by_category": [{"name": k, "value": round(v)} for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])],
        "by_provider": [{"name": k, "revenue": round(v["revenue"]), "visits": v["visits"], "per_visit": round(v["revenue"] / v["visits"]) if v["visits"] else 0}
                        for k, v in sorted(by_provider.items(), key=lambda kv: -kv[1]["revenue"])],
        "expenses_breakdown": [{"name": k, "value": round(v)} for k, v in expenses[keys[-1]].items() if v],
        "aging": [{"bucket": k, "value": round(v, 2)} for k, v in aging.items()],
        "insights": insights,
    }


# ==========================================
# BILLING & INVOICES
# ==========================================
class InvoiceLine(BaseModel):
    description: str
    amount: float


class InvoiceIn(BaseModel):
    patient_id: str
    items: List[InvoiceLine]
    insurance_amount: Optional[float] = 0.0
    due_days: Optional[int] = 14


@router.get("/invoices")
def list_invoices(status: Optional[str] = None, q: Optional[str] = None):
    s = store()
    names = patient_names()
    t = today().isoformat()
    rows = []
    for i in s["invoices"]:
        st = i["status"]
        if st == "Unpaid" and i["due_date"] < t:
            st = "Overdue"
        rows.append({**i, "status": st, "patient": names.get(i["patient_id"], i["patient_id"])})
    if status:
        rows = [r for r in rows if r["status"] == status]
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in r["patient"].lower() or ql in r["id"].lower()]
    rows.sort(key=lambda r: (r["date"], r["id"]), reverse=True)
    all_rows = s["invoices"]
    mkey = today().strftime("%Y-%m")
    return {
        "invoices": rows[:400],
        "summary": {
            "count": len(rows),
            "invoiced_mtd": round(sum(i["amount"] for i in all_rows if month_key(i["date"]) == mkey), 2),
            "collected_mtd": round(sum(i["amount"] for i in all_rows if i["status"] == "Paid" and month_key(i.get("paid_date") or i["date"]) == mkey), 2),
            "unpaid": round(sum(i["patient_amount"] for i in all_rows if i["status"] == "Unpaid" and i["due_date"] >= t), 2),
            "overdue": round(sum(i["patient_amount"] for i in all_rows if i["status"] != "Paid" and (i["status"] == "Overdue" or i["due_date"] < t)), 2),
            "insurance_mtd": round(sum(i["insurance_amount"] for i in all_rows if month_key(i["date"]) == mkey), 2),
        },
    }


@router.post("/invoices")
def create_invoice(body: InvoiceIn):
    s = store()
    find("patients", body.patient_id)
    if not body.items:
        raise HTTPException(status_code=400, detail="Add at least one invoice line.")
    with cd.LOCK:
        t = today()
        nums = [int(i["id"].split("-")[2]) for i in s["invoices"] if i["id"].startswith(f"INV-{t.year}-")]
        amount = round(sum(l.amount for l in body.items), 2)
        ins = min(amount, round(body.insurance_amount or 0, 2))
        inv = {"id": f"INV-{t.year}-{(max(nums) if nums else 1400) + 1:05d}", "patient_id": body.patient_id, "appointment_id": None,
               "date": t.isoformat(), "due_date": (t + timedelta(days=body.due_days or 14)).isoformat(),
               "items": [l.model_dump() for l in body.items], "amount": amount, "insurance_amount": ins,
               "patient_amount": round(amount - ins, 2), "status": "Unpaid", "method": None, "paid_date": None}
        s["invoices"].append(inv)
    return {"status": "success", "invoice": inv}


@router.patch("/invoices/{invoice_id}")
def update_invoice(invoice_id: str, body: dict):
    inv = find("invoices", invoice_id)
    with cd.LOCK:
        if body.get("status") == "Paid":
            inv["status"] = "Paid"
            inv["method"] = body.get("method") or "Card"
            inv["paid_date"] = today().isoformat()
        elif body.get("status") in ("Unpaid", "Overdue"):
            inv["status"], inv["method"], inv["paid_date"] = body["status"], None, None
    return {"status": "success", "invoice": inv}


# ==========================================
# TREATMENT PLANS
# ==========================================
class PlanItemIn(BaseModel):
    code: str
    tooth: Optional[int] = None


class PlanIn(BaseModel):
    patient_id: str
    provider_id: Optional[str] = "U1"
    items: List[PlanItemIn]
    notes: Optional[str] = ""


@router.get("/treatment-plans")
def list_plans(status: Optional[str] = None):
    s = store()
    names, team = patient_names(), cd.team_map()
    rows = []
    for pl in s["plans"]:
        done = sum(1 for it in pl["items"] if it["done"])
        rows.append({**pl, "patient": names.get(pl["patient_id"]), "provider": team.get(pl["provider_id"], {}).get("name"),
                     "progress": round(done / len(pl["items"]) * 100) if pl["items"] else 0,
                     "completed_value": round(sum(it["price"] for it in pl["items"] if it["done"]), 2)})
    if status:
        rows = [r for r in rows if r["status"] == status]
    rows.sort(key=lambda r: r["created"], reverse=True)
    allp = s["plans"]
    decided = [p for p in allp if p["status"] != "Proposed"]
    acc = [p for p in decided if p["status"] in ("Accepted", "In progress", "Completed")]
    return {
        "plans": rows,
        "summary": {
            "total": len(allp),
            "acceptance_rate": round(len(acc) / len(decided) * 100, 1) if decided else 0,
            "pipeline_value": round(sum(p["total"] for p in allp if p["status"] == "Proposed"), 2),
            "active_value": round(sum(it["price"] for p in allp if p["status"] in ("Accepted", "In progress") for it in p["items"] if not it["done"]), 2),
            "by_status": dict(Counter(p["status"] for p in allp)),
        },
    }


@router.post("/treatment-plans")
def create_plan(body: PlanIn):
    s = store()
    find("patients", body.patient_id)
    tmap = cd.treatment_map()
    items = []
    for it in body.items:
        t = tmap.get(it.code)
        if not t:
            raise HTTPException(status_code=400, detail=f"Unknown treatment {it.code}")
        items.append({"code": it.code, "name": t["name"], "tooth": it.tooth, "price": t["price"], "done": False})
    if not items:
        raise HTTPException(status_code=400, detail="Add at least one treatment.")
    with cd.LOCK:
        next_id = max(int(p["id"].split("-")[1]) for p in s["plans"]) + 1 if s["plans"] else 3001
        plan = {"id": f"TP-{next_id}", "patient_id": body.patient_id, "provider_id": body.provider_id,
                "title": " + ".join(dict.fromkeys(i["name"] for i in items)), "created": today().isoformat(),
                "status": "Proposed", "items": items, "total": round(sum(i["price"] for i in items), 2), "notes": body.notes}
        s["plans"].append(plan)
    return {"status": "success", "plan": plan}


@router.patch("/treatment-plans/{plan_id}")
def update_plan(plan_id: str, body: dict):
    pl = find("plans", plan_id)
    with cd.LOCK:
        if "status" in body and body["status"] in ("Proposed", "Accepted", "In progress", "Completed", "Declined"):
            pl["status"] = body["status"]
            if body["status"] == "Completed":
                for it in pl["items"]:
                    it["done"] = True
        if "item_index" in body:
            idx = int(body["item_index"])
            if 0 <= idx < len(pl["items"]):
                pl["items"][idx]["done"] = bool(body.get("done", True))
                done = sum(it["done"] for it in pl["items"])
                if done == len(pl["items"]):
                    pl["status"] = "Completed"
                elif done > 0 and pl["status"] in ("Proposed", "Accepted"):
                    pl["status"] = "In progress"
    return {"status": "success", "plan": pl}


# ==========================================
# STOCK MANAGER
# ==========================================
def stock_row(x, suppliers):
    weeks_left = round(x["qty"] / x["weekly_usage"], 1) if x["weekly_usage"] else None
    state = "Out of stock" if x["qty"] == 0 else "Low" if x["qty"] < x["min_qty"] else "OK"
    expiring = bool(x["expiry"]) and (date.fromisoformat(x["expiry"]) - today()).days <= 60
    return {**x, "supplier": suppliers.get(x["supplier_id"], {}).get("name"), "state": state, "weeks_left": weeks_left,
            "value": round(x["qty"] * x["unit_cost"], 2), "expiring_soon": expiring}


@router.get("/stock")
def list_stock():
    s = store()
    suppliers = {sp["id"]: sp for sp in s["suppliers"]}
    rows = [stock_row(x, suppliers) for x in s["stock"]]
    on_order = defaultdict(int)
    for po in s["purchase_orders"]:
        if po["status"] in ("Ordered", "Draft"):
            for l in po["lines"]:
                on_order[l["stock_id"]] += l["qty"]
    for r in rows:
        r["on_order"] = on_order.get(r["id"], 0)
    return {
        "items": sorted(rows, key=lambda r: ({"Out of stock": 0, "Low": 1, "OK": 2}[r["state"]], r["name"])),
        "summary": {
            "items": len(rows), "low": sum(r["state"] == "Low" for r in rows), "out": sum(r["state"] == "Out of stock" for r in rows),
            "expiring": sum(r["expiring_soon"] for r in rows), "value": round(sum(r["value"] for r in rows), 2),
            "categories": sorted({r["category"] for r in rows}),
        },
    }


@router.patch("/stock/{item_id}")
def adjust_stock(item_id: str, body: dict):
    x = find("stock", item_id)
    with cd.LOCK:
        if "adjust" in body:
            x["qty"] = max(0, x["qty"] + int(body["adjust"]))
        if "qty" in body:
            x["qty"] = max(0, int(body["qty"]))
        for k in ("min_qty", "reorder_qty"):
            if k in body:
                x[k] = max(0, int(body[k]))
    return {"status": "success", "item": x}


class StockIn(BaseModel):
    name: str
    category: str
    unit: str = "piece"
    qty: int = 0
    min_qty: int = 5
    reorder_qty: int = 10
    unit_cost: float = 0
    supplier_id: str = "S1"
    weekly_usage: float = 1
    expiry: Optional[str] = None
    location: Optional[str] = "Store room A"


@router.post("/stock")
def create_stock(body: StockIn):
    s = store()
    with cd.LOCK:
        next_id = max(int(x["id"].split("-")[1]) for x in s["stock"]) + 1
        item = {**body.model_dump(), "id": f"ST-{next_id}"}
        s["stock"].append(item)
    return {"status": "success", "item": item}


# ==========================================
# PROCUREMENT
# ==========================================
class POLine(BaseModel):
    stock_id: str
    qty: int


class POIn(BaseModel):
    supplier_id: str
    lines: List[POLine]
    notes: Optional[str] = ""


def po_row(po, suppliers):
    overdue = po["status"] == "Ordered" and po["expected"] < today().isoformat()
    return {**po, "supplier": suppliers.get(po["supplier_id"], {}).get("name"), "overdue": overdue}


@router.get("/procurement")
def procurement():
    s = store()
    suppliers = {sp["id"]: sp for sp in s["suppliers"]}
    orders = [po_row(po, suppliers) for po in s["purchase_orders"]]
    spend = defaultdict(float)
    for po in s["purchase_orders"]:
        if po["status"] == "Received":
            spend[po["supplier_id"]] += po["total"]
    mkey = today().strftime("%Y-%m")
    return {
        "orders": orders,
        "suppliers": [{**sp, "spend": round(spend.get(sp["id"], 0), 2),
                       "open_orders": sum(1 for po in s["purchase_orders"] if po["supplier_id"] == sp["id"] and po["status"] in ("Draft", "Ordered"))}
                      for sp in s["suppliers"]],
        "summary": {
            "open": sum(po["status"] in ("Draft", "Ordered") for po in s["purchase_orders"]),
            "drafts": sum(po["status"] == "Draft" for po in s["purchase_orders"]),
            "overdue": sum(o["overdue"] for o in orders),
            "spend_mtd": round(sum(po["total"] for po in s["purchase_orders"] if po["status"] == "Received" and month_key(po["received_date"] or "") == mkey), 2),
            "low_stock_items": sum(x["qty"] < x["min_qty"] for x in s["stock"]),
        },
    }


def _new_po(supplier_id, lines, notes=""):
    s = store()
    stock = {x["id"]: x for x in s["stock"]}
    po_lines = []
    for l in lines:
        x = stock.get(l["stock_id"])
        if not x:
            raise HTTPException(status_code=400, detail=f"Unknown stock item {l['stock_id']}")
        po_lines.append({"stock_id": x["id"], "name": x["name"], "qty": int(l["qty"]), "unit_cost": x["unit_cost"]})
    nums = [int(po["id"].split("-")[2]) for po in s["purchase_orders"]]
    t = today()
    sup = next((sp for sp in s["suppliers"] if sp["id"] == supplier_id), None)
    if not sup:
        raise HTTPException(status_code=400, detail=f"Unknown supplier {supplier_id}")
    po = {"id": f"PO-{t.year}-{(max(nums) if nums else 700) + 1}", "supplier_id": supplier_id, "created": t.isoformat(),
          "expected": (t + timedelta(days=sup["lead_days"])).isoformat(), "status": "Draft", "received_date": None,
          "lines": po_lines, "total": round(sum(l["qty"] * l["unit_cost"] for l in po_lines), 2), "notes": notes}
    s["purchase_orders"].insert(0, po)
    return po


@router.post("/procurement/orders")
def create_po(body: POIn):
    if not body.lines:
        raise HTTPException(status_code=400, detail="Add at least one item.")
    with cd.LOCK:
        po = _new_po(body.supplier_id, [l.model_dump() for l in body.lines], body.notes)
    return {"status": "success", "order": po}


@router.post("/procurement/auto-reorder")
def auto_reorder():
    """Creates draft purchase orders (grouped by supplier) for every item below minimum stock that is not already on order."""
    s = store()
    on_order = {l["stock_id"] for po in s["purchase_orders"] if po["status"] in ("Draft", "Ordered") for l in po["lines"]}
    by_sup = defaultdict(list)
    for x in s["stock"]:
        if x["qty"] < x["min_qty"] and x["id"] not in on_order:
            by_sup[x["supplier_id"]].append({"stock_id": x["id"], "qty": x["reorder_qty"]})
    created = []
    with cd.LOCK:
        for sup_id, lines in by_sup.items():
            created.append(_new_po(sup_id, lines, "Auto-generated from low stock"))
    return {"status": "success", "created": created}


@router.patch("/procurement/orders/{po_id}")
def update_po(po_id: str, body: dict):
    po = find("purchase_orders", po_id)
    new_status = body.get("status")
    if new_status not in ("Draft", "Ordered", "Received", "Cancelled"):
        raise HTTPException(status_code=400, detail="Invalid status")
    with cd.LOCK:
        if new_status == "Received" and po["status"] != "Received":
            stock = {x["id"]: x for x in store()["stock"]}
            for l in po["lines"]:
                if l["stock_id"] in stock:
                    stock[l["stock_id"]]["qty"] += l["qty"]
            po["received_date"] = today().isoformat()
        if new_status == "Ordered":
            sup = next((sp for sp in store()["suppliers"] if sp["id"] == po["supplier_id"]), {"lead_days": 3})
            po["expected"] = (today() + timedelta(days=sup["lead_days"])).isoformat()
        po["status"] = new_status
    return {"status": "success", "order": po}


# ==========================================
# EQUIPMENT SCHEDULER
# ==========================================
@router.get("/equipment")
def list_equipment():
    s = store()
    t = today()
    since = (t - timedelta(days=30)).isoformat()
    slot = int(cd.get_settings().get("slot_minutes", 30))
    recent = [a for a in s["appointments"] if since <= a["date"] <= t.isoformat() and a["status"] != "Cancelled"]
    capacity_days = sum(1 for i in range(30) if cd.day_hours(t - timedelta(days=i)) is not None)
    slots_per_day = len(cd.day_slots(t)) or 18
    rows = []
    for e in s["equipment"]:
        if e["chair"]:
            used = sum(max(1, -(-a["duration"] // slot)) for a in recent if a["chair"] == e["chair"])
            util = used / (capacity_days * slots_per_day) * 100 if capacity_days else 0
        elif e["uses"]:
            used = sum(1 for a in recent if a["treatment"] in e["uses"])
            util = min(100, used / (capacity_days * slots_per_day) * 100 * 2) if capacity_days else 0
        else:
            util = None
        days_to_service = (date.fromisoformat(e["next_service"]) - t).days
        rows.append({**e, "utilization": round(util, 1) if util is not None else None, "days_to_service": days_to_service,
                     "service_state": "Overdue" if days_to_service < 0 else "Due soon" if days_to_service <= 14 else "OK"})
    return {
        "equipment": rows,
        "summary": {
            "total": len(rows), "operational": sum(r["status"] == "Operational" for r in rows),
            "maintenance": sum(r["status"] in ("Maintenance", "Out of service") for r in rows),
            "service_due": sum(r["service_state"] != "OK" for r in rows),
            "avg_chair_util": round(sum(r["utilization"] for r in rows if r["chair"]) / max(1, sum(1 for r in rows if r["chair"])), 1),
        },
    }


@router.get("/equipment/schedule")
def equipment_schedule(date_: Optional[str] = Query(None, alias="date")):
    d = date.fromisoformat(date_) if date_ else today()
    names, tmap, team = patient_names(), cd.treatment_map(), cd.team_map()
    s = store()
    apps = [a for a in s["appointments"] if a["date"] == d.isoformat() and a["status"] != "Cancelled"]
    rows = []
    for e in s["equipment"]:
        if e["chair"]:
            booked = [a for a in apps if a["chair"] == e["chair"]]
        elif e["uses"]:
            booked = [a for a in apps if a["treatment"] in e["uses"]]
        else:
            continue
        rows.append({"id": e["id"], "name": e["name"], "type": e["type"], "status": e["status"],
                     "bookings": [{"id": a["id"], "time": a["time"], "duration": a["duration"], "status": a["status"],
                                   "patient": names.get(a["patient_id"]), "treatment": tmap.get(a["treatment"], {}).get("name", a["treatment"]),
                                   "provider": team.get(a["provider_id"], {}).get("name"), "color": team.get(a["provider_id"], {}).get("color", "#4F5BD5")}
                                  for a in sorted(booked, key=lambda a: a["time"])]})
    return {"date": d.isoformat(), "slots": cd.day_slots(d), "slot_minutes": int(cd.get_settings().get("slot_minutes", 30)),
            "closed": cd.day_hours(d) is None, "resources": rows}


@router.patch("/equipment/{equipment_id}")
def update_equipment(equipment_id: str, body: dict):
    e = find("equipment", equipment_id)
    with cd.LOCK:
        if body.get("status") in ("Operational", "Maintenance", "Out of service"):
            e["status"] = body["status"]
        if body.get("log_service"):
            t = today()
            e["last_service"] = t.isoformat()
            e["next_service"] = (t + timedelta(days=e["service_interval_days"])).isoformat()
            e["service_log"].insert(0, {"date": t.isoformat(), "note": body.get("note") or "Service completed"})
            e["status"] = "Operational"
    return {"status": "success", "equipment": e}


# ==========================================
# LAB ORDERS
# ==========================================
class LabIn(BaseModel):
    patient_id: str
    type: str
    lab: str
    provider_id: Optional[str] = "U2"
    tooth: Optional[int] = None
    shade: Optional[str] = None
    due_date: Optional[str] = None
    cost: Optional[float] = 0
    notes: Optional[str] = ""
    plan_id: Optional[str] = None


LAB_STATUSES = ["Draft", "Sent", "In production", "Shipped", "Received", "Fitted", "Remake"]


@router.get("/lab-orders")
def list_lab_orders(status: Optional[str] = None):
    s = store()
    names, team = patient_names(), cd.team_map()
    t = today().isoformat()
    rows = []
    for l in s["lab_orders"]:
        overdue = l["due_date"] < t and l["status"] in ("Draft", "Sent", "In production", "Shipped")
        rows.append({**l, "patient": names.get(l["patient_id"]), "provider": team.get(l["provider_id"], {}).get("name"), "overdue": overdue})
    if status:
        rows = [r for r in rows if r["status"] == status]
    mkey = today().strftime("%Y-%m")
    active = [r for r in rows if r["status"] in ("Draft", "Sent", "In production", "Shipped")]
    return {
        "orders": rows,
        "labs": cd.LABS, "types": [{"type": lt, "cost": c, "turnaround": d} for lt, c, d in cd.LAB_TYPES], "statuses": LAB_STATUSES,
        "summary": {
            "active": len(active), "overdue": sum(r["overdue"] for r in rows),
            "due_this_week": sum(1 for r in active if t <= r["due_date"] <= (today() + timedelta(days=7)).isoformat()),
            "cost_mtd": round(sum(l["cost"] for l in s["lab_orders"] if month_key(l["sent_date"]) == mkey), 2),
            "remakes": sum(r["status"] == "Remake" for r in rows),
        },
    }


@router.post("/lab-orders")
def create_lab_order(body: LabIn):
    s = store()
    find("patients", body.patient_id)
    turnaround = next((d for lt, c, d in cd.LAB_TYPES if lt == body.type), 7)
    with cd.LOCK:
        next_id = max(int(l["id"].split("-")[1]) for l in s["lab_orders"]) + 1
        order = {**body.model_dump(), "id": f"LAB-{next_id}", "sent_date": today().isoformat(),
                 "due_date": body.due_date or (today() + timedelta(days=turnaround)).isoformat(), "status": "Sent"}
        s["lab_orders"].insert(0, order)
    return {"status": "success", "order": order}


@router.patch("/lab-orders/{lab_id}")
def update_lab_order(lab_id: str, body: dict):
    l = find("lab_orders", lab_id)
    if body.get("status") not in LAB_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    with cd.LOCK:
        l["status"] = body["status"]
        if body["status"] == "Remake":
            l["due_date"] = (today() + timedelta(days=7)).isoformat()
    return {"status": "success", "order": l}


# ==========================================
# REPORTS
# ==========================================
@router.get("/reports")
def reports(months: int = Query(6, ge=3, le=12)):
    s = store()
    keys = last_months(months)
    start = keys[0] + "-01"
    tmap, team = cd.treatment_map(), cd.team_map()
    appts = [a for a in s["appointments"] if a["date"] >= start and a["status"] != "Upcoming"]

    monthly = []
    for k in keys:
        grp = [a for a in appts if month_key(a["date"]) == k]
        past = [a for a in grp if a["status"] in PAST_STATUSES]
        ns = sum(a["status"] == "No-show" for a in past)
        new_p = sum(1 for p in s["patients"] if month_key(p["registered"]) == k)
        rated = [a["rating"] for a in grp if a.get("rating")]
        monthly.append({"month": month_label(k), "appointments": len(grp), "completed": sum(a["status"] == "Completed" for a in grp),
                        "noshows": ns, "cancelled": sum(a["status"] == "Cancelled" for a in grp),
                        "noshow_rate": round(ns / len(past) * 100, 1) if past else 0, "new_patients": new_p,
                        "avg_rating": round(sum(rated) / len(rated), 2) if rated else None})

    past = [a for a in appts if a["status"] in PAST_STATUSES]

    def rate(group):
        return round(sum(a["status"] == "No-show" for a in group) / len(group) * 100, 1) if group else 0

    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_weekday = [{"day": weekdays[i], "rate": rate([a for a in past if date.fromisoformat(a["date"]).weekday() == i]),
                   "count": sum(1 for a in past if date.fromisoformat(a["date"]).weekday() == i)} for i in range(7)]
    by_weekday = [r for r in by_weekday if r["count"]]
    buckets = [("Same day", 0, 0), ("1–7 d", 1, 7), ("8–30 d", 8, 30), ("31+ d", 31, 999)]
    by_lead = [{"bucket": b, "rate": rate([a for a in past if lo <= a["lead_days"] <= hi]), "count": sum(1 for a in past if lo <= a["lead_days"] <= hi)} for b, lo, hi in buckets]
    advance = [a for a in past if a["lead_days"] >= 2]
    reminders = [{"group": "SMS reminder sent", "rate": rate([a for a in advance if a["sms_reminder"]])},
                 {"group": "No reminder", "rate": rate([a for a in advance if not a["sms_reminder"]])}]
    mix = Counter(tmap.get(a["treatment"], {}).get("name", a["treatment"]) for a in appts if a["status"] == "Completed")
    providers = []
    for uid, u in team.items():
        grp = [a for a in appts if a["provider_id"] == uid]
        if not grp:
            continue
        done = [a for a in grp if a["status"] == "Completed"]
        rated = [a["rating"] for a in done if a.get("rating")]
        providers.append({"name": u["name"], "role": u["role"], "visits": len(done), "revenue": round(sum(a["fee"] for a in done)),
                          "noshow_rate": rate([a for a in grp if a["status"] in PAST_STATUSES]),
                          "avg_rating": round(sum(rated) / len(rated), 2) if rated else None})
    providers.sort(key=lambda r: -r["revenue"])
    ratings = [a["rating"] for a in appts if a.get("rating")]
    promoters = sum(r == 5 for r in ratings)
    detractors = sum(r <= 3 for r in ratings)
    return {
        "months": months,
        "monthly": monthly,
        "kpis": {
            "appointments": len(appts), "completed": sum(a["status"] == "Completed" for a in appts),
            "noshow_rate": rate(past), "cancel_rate": round(sum(a["status"] == "Cancelled" for a in appts) / len(appts) * 100, 1) if appts else 0,
            "new_patients": sum(m["new_patients"] for m in monthly),
            "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "nps": round((promoters - detractors) / len(ratings) * 100) if ratings else None, "ratings": len(ratings),
        },
        "by_weekday": by_weekday,
        "by_lead": by_lead,
        "reminders": reminders,
        "procedure_mix": [{"name": k, "count": v} for k, v in mix.most_common(10)],
        "providers": providers,
    }


# ==========================================
# GLOBAL SEARCH & NOTIFICATIONS (top bar)
# ==========================================
@router.get("/search")
def global_search(q: str = Query(..., min_length=2)):
    s = store()
    ql = q.strip().lower()
    names = patient_names()
    digits = "".join(ch for ch in ql if ch.isdigit())
    results = []
    for p in s["patients"]:
        if ql in p["name"].lower() or ql in p["id"].lower() or (len(digits) >= 4 and digits in p["phone"]) or ql in (p.get("email") or "").lower():
            results.append({"type": "Patient", "id": p["id"], "title": p["name"], "subtitle": f"{p['id']} · {p['phone']} · {p['city']}", "page": "patients", "params": {"open": p["id"]}})
        if len([r for r in results if r["type"] == "Patient"]) >= 6:
            break
    for i in sorted(s["invoices"], key=lambda i: i["date"], reverse=True):
        if ql in i["id"].lower() or ql in names.get(i["patient_id"], "").lower():
            results.append({"type": "Invoice", "id": i["id"], "title": i["id"], "subtitle": f"{names.get(i['patient_id'])} · €{i['amount']:.0f} · {i['status']}", "page": "billing", "params": {"q": i["id"]}})
        if len([r for r in results if r["type"] == "Invoice"]) >= 4:
            break
    for pl in s["plans"]:
        if ql in pl["id"].lower() or ql in pl["title"].lower() or ql in names.get(pl["patient_id"], "").lower():
            results.append({"type": "Treatment plan", "id": pl["id"], "title": pl["title"], "subtitle": f"{names.get(pl['patient_id'])} · {pl['status']}", "page": "treatment", "params": {"open": pl["id"]}})
        if len([r for r in results if r["type"] == "Treatment plan"]) >= 3:
            break
    for l in s["lab_orders"]:
        if ql in l["id"].lower() or ql in l["type"].lower() or ql in names.get(l["patient_id"], "").lower():
            results.append({"type": "Lab order", "id": l["id"], "title": f"{l['type']} · {l['id']}", "subtitle": f"{names.get(l['patient_id'])} · {l['status']}", "page": "lab", "params": {"open": l["id"]}})
        if len([r for r in results if r["type"] == "Lab order"]) >= 3:
            break
    for x in s["stock"]:
        if ql in x["name"].lower():
            results.append({"type": "Stock item", "id": x["id"], "title": x["name"], "subtitle": f"{x['qty']} {x['unit']} in stock (min {x['min_qty']})", "page": "stock", "params": {"q": x["name"]}})
    return {"query": q, "results": results[:20]}


@router.get("/notifications")
def notifications():
    """Live alerts for the bell icon, built from the current clinic data."""
    s = store()
    t = today()
    t_iso = t.isoformat()
    names = patient_names()
    settings = cd.get_settings()
    items = []

    soon = (t + timedelta(days=1)).isoformat()
    risky = [a for a in s["appointments"] if a["status"] == "Upcoming" and t_iso <= a["date"] <= soon
             and a["prob"] is not None and cd.risk_level(a["prob"]) in ("high", "critical") and not a.get("reminded_at")]
    if risky:
        items.append({"id": f"risk-{t_iso}-{len(risky)}", "kind": "risk", "icon": "🚨", "page": "predictions",
                      "title": f"{len(risky)} high-risk appointment(s) today or tomorrow",
                      "detail": f"€{round(sum(a['fee'] for a in risky))} at risk — send reminders now"})
    if settings["notifications"].get("low_stock_alert", True):
        low = [x for x in s["stock"] if x["qty"] < x["min_qty"]]
        if low:
            items.append({"id": "stock-" + "-".join(sorted(x["id"] for x in low)), "kind": "stock", "icon": "📦", "page": "stock",
                          "title": f"{len(low)} stock item(s) below minimum",
                          "detail": ", ".join(x["name"].split(" (")[0] for x in low[:3]) + ("…" if len(low) > 3 else "")})
    if settings["notifications"].get("overdue_lab_alert", True):
        late = [l for l in s["lab_orders"] if l["due_date"] < t_iso and l["status"] in ("Draft", "Sent", "In production", "Shipped")]
        for l in late[:3]:
            items.append({"id": f"lab-{l['id']}-{l['status']}", "kind": "lab", "icon": "🧪", "page": "lab", "params": {"open": l["id"]},
                          "title": f"Lab order {l['id']} is overdue", "detail": f"{l['type']} for {names.get(l['patient_id'])} · due {l['due_date']}"})
    overdue_days = int(settings["notifications"].get("overdue_invoice_days", 30))
    cutoff = (t - timedelta(days=overdue_days)).isoformat()
    old_unpaid = [i for i in s["invoices"] if i["status"] != "Paid" and i["date"] <= cutoff]
    if old_unpaid:
        items.append({"id": f"inv-{len(old_unpaid)}-{round(sum(i['patient_amount'] for i in old_unpaid))}", "kind": "billing", "icon": "💳", "page": "billing",
                      "params": {"status": "Overdue"}, "title": f"{len(old_unpaid)} invoice(s) unpaid for {overdue_days}+ days",
                      "detail": f"€{round(sum(i['patient_amount'] for i in old_unpaid))} outstanding"})
    for e in s["equipment"]:
        days = (date.fromisoformat(e["next_service"]) - t).days
        if days <= 14:
            items.append({"id": f"eq-{e['id']}-{e['next_service']}", "kind": "equipment", "icon": "🛠️", "page": "equipment",
                          "title": f"{e['name']}: service {'overdue' if days < 0 else 'due soon'}",
                          "detail": f"{'Overdue by ' + str(-days) if days < 0 else 'Due in ' + str(days)} day(s) · {e['next_service']}"})
    late_po = [po for po in s["purchase_orders"] if po["status"] == "Ordered" and po["expected"] < t_iso]
    if late_po:
        items.append({"id": "po-" + "-".join(po["id"] for po in late_po), "kind": "procurement", "icon": "🛒", "page": "procurement",
                      "title": f"{len(late_po)} supplier delivery(ies) late", "detail": ", ".join(po["id"] for po in late_po[:3])})
    proposed = [p for p in s["plans"] if p["status"] == "Proposed"]
    if proposed:
        items.append({"id": f"plans-{len(proposed)}", "kind": "plans", "icon": "📋", "page": "treatment",
                      "title": f"{len(proposed)} treatment plans awaiting patient decision",
                      "detail": f"€{round(sum(p['total'] for p in proposed)):,} in proposals"})
    return {"generated_at": cd.clinic_now().isoformat(timespec="minutes"), "items": items}


@router.get("/reminders/pending")
def pending_reminders(days: int = Query(2, ge=1, le=7)):
    """Upcoming moderate+ risk appointments that have not been reminded yet (top-bar messages)."""
    pmap = {p["id"]: p for p in store()["patients"]}
    names, tmap, team = patient_names(), cd.treatment_map(), cd.team_map()
    rows = []
    for a in upcoming_appointments(days):
        if a["prob"] is None or a.get("reminded_at") or cd.risk_level(a["prob"]) == "low":
            continue
        e = enrich_appointment(a, names, tmap, team)
        e["phone"] = pmap[a["patient_id"]]["phone"]
        rows.append(e)
    rows.sort(key=lambda r: -r["prob"])
    return {"items": rows[:15], "total": len(rows)}
