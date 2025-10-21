import json
import pandas as pd
from datetime import datetime, timedelta

# ---------- 1. LOAD FHIR DATA ----------
OBS_PATH = "data/observations.json"
MED_PATH = "data/medication_administrations.json"
PAT_PATH = "data/patients.json"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

try:
    observations = load_json(OBS_PATH)
    medications = load_json(MED_PATH)
    patients = load_json(PAT_PATH)
except Exception as e:
    print("⚠️ Could not load files:", e)
    exit()

# ---------- 2. FLATTEN OBSERVATIONS ----------
def flatten_observation(obs):
    return {
        "patient_id": obs["subject"]["reference"].split("/")[-1],
        "code": obs["code"]["coding"][0]["code"],
        "display": obs["code"]["coding"][0]["display"],
        "value": obs.get("valueQuantity", {}).get("value"),
        "unit": obs.get("valueQuantity", {}).get("unit"),
        "effective_datetime": obs["effectiveDateTime"]
    }

obs_df = pd.DataFrame([flatten_observation(o["resource"]) for o in observations["entry"]])
obs_df["effective_datetime"] = pd.to_datetime(obs_df["effective_datetime"])

# ---------- 3. DEFINE SOFA FUNCTIONS ----------
def sofa_respiratory(pao2=None, spo2=None, fio2=0.21, resp_support=False):
    if pao2:
        ratio = pao2 / fio2
        if ratio < 100 and resp_support: return 4
        elif ratio < 200 and resp_support: return 3
        elif ratio < 300: return 2
        elif ratio < 400: return 1
        else: return 0
    elif spo2:
        ratio = spo2 / fio2
        if ratio < 67 and resp_support: return 4
        elif ratio < 142 and resp_support: return 3
        elif ratio < 221: return 2
        elif ratio < 302: return 1
        else: return 0
    return None

def sofa_liver(bilirubin):
    if bilirubin is None: return None
    if bilirubin >= 12: return 4
    elif bilirubin >= 6: return 3
    elif bilirubin >= 2: return 2
    elif bilirubin >= 1.2: return 1
    else: return 0

def sofa_coagulation(platelet):
    if platelet is None: return None
    if platelet < 20: return 4
    elif platelet < 50: return 3
    elif platelet < 100: return 2
    elif platelet < 150: return 1
    else: return 0

def sofa_kidney(creatinine):
    if creatinine is None: return None
    if creatinine >= 5: return 4
    elif creatinine >= 3.5: return 3
    elif creatinine >= 2: return 2
    elif creatinine >= 1.2: return 1
    else: return 0

def sofa_cns(gcs):
    if gcs is None: return None
    if gcs < 6: return 4
    elif gcs < 9: return 3
    elif gcs < 12: return 2
    elif gcs < 15: return 1
    else: return 0

def sofa_cardio(map_val, pressor=False):
    if map_val is None: return None
    if map_val < 70 and pressor: return 4
    elif map_val < 70: return 2
    else: return 0

# ---------- 4. CALCULATE SCORES ----------
rows = []
six_hours = timedelta(hours=6)

for pid, patient_data in obs_df.groupby("patient_id"):
    times = sorted(patient_data["effective_datetime"].unique())
    for t in times:
        window = patient_data[
            (patient_data["effective_datetime"] <= t) &
            (patient_data["effective_datetime"] >= t - six_hours)
        ]

        def get_val(keyword):
            row = window[window["display"].str.contains(keyword, case=False, na=False)]
            if not row.empty:
                return row["value"].iloc[-1]
            return None

        # Extract values
        pao2 = get_val("PaO2")
        spo2 = get_val("SpO2")
        bilirubin = get_val("bilirubin")
        platelet = get_val("platelet")
        creatinine = get_val("creatinine")
        gcs = get_val("Glasgow")
        map_val = get_val("Mean arterial")

        # Component scores
        resp = sofa_respiratory(pao2, spo2)
        liver = sofa_liver(bilirubin)
        coag = sofa_coagulation(platelet)
        kidney = sofa_kidney(creatinine)
        cns = sofa_cns(gcs)
        cardio = sofa_cardio(map_val)

        if all(v is not None for v in [resp, liver, coag, kidney, cns, cardio]):
            total = sum([resp, liver, coag, kidney, cns, cardio])
            rows.append({"patient_id": pid, "sofa_score_datetime": t, "sofa_score": total})

######5. EXPORT
df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)
print("submission.csv created with", len(df), "rows.")
