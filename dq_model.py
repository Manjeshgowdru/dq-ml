import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# Load data
df = pd.read_csv("KaggleV2-May-2016.csv")

# Convert target Yes/No to 1/0
df["No-show"] = df["No-show"].map({"Yes": 1, "No": 0})

# Engineer features
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
df["lead_time_days"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
df["day_of_week"] = df["AppointmentDay"].dt.dayofweek

# Clean bad data
df = df[df["lead_time_days"] >= 0]

# Create age groups
df["Age"] = df["Age"].fillna(0)
df["age_group"] = pd.cut(df["Age"],
                   bins=[0, 18, 25, 35, 50, 100],
                   labels=[0, 1, 2, 3, 4])
df["age_group"] = df["age_group"].cat.codes

# Select features
features = ["Age", "age_group", "lead_time_days", "day_of_week",
            "SMS_received", "Scholarship",
            "Hipertension", "Diabetes",
            "Alcoholism", "Handcap"]

X = df[features]
y = df["No-show"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model with Random Forest
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(
#     n_estimators=100,
#     class_weight="balanced",
#     max_depth=5,
#     min_samples_leaf=50,
#     random_state=42
# )
# model.fit(X_train, y_train)

# from xgboost import XGBClassifier

# model = XGBClassifier(
#     n_estimators=100,
#     max_depth=4,
#     learning_rate=0.1,
#     scale_pos_weight=4,
#     random_state=42
# )
# model.fit(X_train, y_train)

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=4,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Get real AUC
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"Real AUC: {auc:.4f}")

# Save model
joblib.dump(model, "dental_model.pkl")
print("Model saved!")