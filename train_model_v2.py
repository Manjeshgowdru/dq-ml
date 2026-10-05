"""
DentalIQ no-show model v2.

Changes vs dq_model.py (v1):
- Lead time is computed on calendar dates, so same-day bookings are kept (v1 dropped ~35% of rows).
- Adds patient history (previous appointments / no-shows, strictly before the current one).
- No scale_pos_weight, so predicted probabilities match real no-show rates.
- Validated on a time-based split (train on earlier dates, test on later), then refit on all data.

Usage:  python train_model_v2.py   (needs KaggleV2-May-2016.csv next to this file)
Output: no_show_model_v2.pkl + no_show_model_v2.json (metrics + feature list)
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "KaggleV2-May-2016.csv")
MODEL_PATH = os.path.join(HERE, "no_show_model_v2.pkl")
META_PATH = os.path.join(HERE, "no_show_model_v2.json")

# Only features the clinic actually knows at booking time
FEATURES = [
    "age", "lead_days", "same_day", "day_of_week", "booked_hour", "male", "sms_reminder",
    "hypertension", "diabetes", "alcoholism", "handicap",
    "prev_appts", "prev_noshows", "prev_noshow_rate",
]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    scheduled = pd.to_datetime(raw["ScheduledDay"])
    appointment = pd.to_datetime(raw["AppointmentDay"])
    df["patient_id"] = raw["PatientId"]
    df["appointment_id"] = raw["AppointmentID"]
    df["sched_date"] = scheduled.dt.normalize()
    df["appt_date"] = appointment.dt.normalize()
    df["y"] = (raw["No-show"] == "Yes").astype(int)
    df["age"] = raw["Age"]
    df["lead_days"] = (df["appt_date"] - df["sched_date"]).dt.days
    df["same_day"] = (df["lead_days"] == 0).astype(int)
    df["day_of_week"] = df["appt_date"].dt.dayofweek
    df["booked_hour"] = scheduled.dt.hour
    df["male"] = (raw["Gender"] == "M").astype(int)
    df["sms_reminder"] = raw["SMS_received"]
    df["hypertension"] = raw["Hipertension"]
    df["diabetes"] = raw["Diabetes"]
    df["alcoholism"] = raw["Alcoholism"]
    df["handicap"] = (raw["Handcap"] > 0).astype(int)

    df = df[(df["lead_days"] >= 0) & (df["age"] >= 0)]

    # Patient history: only appointments strictly before this one (no leakage)
    df = df.sort_values(["appt_date", "sched_date", "appointment_id"]).reset_index(drop=True)
    grouped = df.groupby("patient_id")
    df["prev_appts"] = grouped.cumcount()
    df["prev_noshows"] = grouped["y"].cumsum() - df["y"]
    df["prev_noshow_rate"] = np.where(df["prev_appts"] > 0, df["prev_noshows"] / df["prev_appts"].clip(lower=1), -1.0)
    return df


def new_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )


def main():
    df = build_features(pd.read_csv(CSV_PATH))

    cutoff = df["appt_date"].quantile(0.8)
    train, test = df[df["appt_date"] < cutoff], df[df["appt_date"] >= cutoff]
    model = new_model().fit(train[FEATURES], train["y"])
    pred = model.predict_proba(test[FEATURES])[:, 1]
    advance = (test["same_day"] == 0).values

    metrics = {
        "roc_auc": round(float(roc_auc_score(test["y"], pred)), 4),
        "roc_auc_advance_bookings": round(float(roc_auc_score(test["y"][advance], pred[advance])), 4),
        "pr_auc": round(float(average_precision_score(test["y"], pred)), 4),
        "brier": round(float(brier_score_loss(test["y"], pred)), 4),
        "mean_predicted": round(float(pred.mean()), 4),
        "actual_noshow_rate": round(float(test["y"].mean()), 4),
        "validation": f"time split: train < {cutoff.date()} ({len(train)} rows), test >= {cutoff.date()} ({len(test)} rows)",
    }
    print(json.dumps(metrics, indent=2))

    final = new_model().fit(df[FEATURES], df["y"])
    joblib.dump(final, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "version": "2.0.0",
            "algorithm": "XGBoost (unweighted, calibrated probabilities)",
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_rows": int(len(df)),
            "dataset": "KaggleV2-May-2016 (public medical appointment no-shows)",
            "features": FEATURES,
            "metrics": metrics,
        }, f, indent=2)
    print(f"Saved {MODEL_PATH} and {META_PATH}")


if __name__ == "__main__":
    main()
