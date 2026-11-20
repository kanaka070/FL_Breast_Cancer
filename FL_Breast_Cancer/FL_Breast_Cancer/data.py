import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_schema import FEATURES
from workflow_utils import partition_hospital_indices

# DAY 1 - LOAD THE DATASET

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "wdbc.data"

df = pd.read_csv(DATASET_PATH, header=None)

print("First 5 rows:")
print(df.head())

print("Dataset shape:", df.shape)


# GIVE NAMES TO ALL 32 COLUMNS

df.columns = ["id", "diagnosis", *FEATURES]

print("\nDataset with column names:")
print(df.head())


# CHECK DATASET INFORMATION

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDiagnosis counts:")
print(df["diagnosis"].value_counts())

print("\nDiagnosis percentages:")
print(df["diagnosis"].value_counts(normalize=True) * 100)


# REMOVE PATIENT ID

df = df.drop(columns=["id"])


# CONVERT DIAGNOSIS INTO NUMBERS
# Malignant = 1
# Benign = 0

df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

print("\nDataset after preprocessing:")
print(df.head())


# DAY 2 - SEPARATE FEATURES AND TARGET

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# SPLIT INTO TRAINING AND TESTING DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# FEATURE SCALING

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training data:", X_train_scaled.shape)
print("Scaled testing data:", X_test_scaled.shape)

print("\nDAY 1 AND DAY 2 COMPLETED SUCCESSFULLY!")

# ============================================================
# DAY 5 - CREATE THREE SIMULATED HOSPITALS
# ============================================================

import numpy as np

# STEP 1: Separate benign and malignant training patients

hospital_indices = partition_hospital_indices(y_train, seed=42)

# STEP 6: Give our simulated hospitals fictional names

hospital_names = [
    "NovaCare Hospital",
    "Asterion Medical Center",
    "Lumina Health Institute"
]

# STEP 7: Create the three hospital datasets

hospitals = {}

for name, indices in zip(hospital_names, hospital_indices):

    hospitals[name] = {
        "X": X_train.loc[indices].copy(),
        "y": y_train.loc[indices].copy()
    }

# STEP 8: Display each hospital's information

print("\n" + "=" * 55)
print("DAY 5 - THREE SIMULATED HOSPITALS")
print("=" * 55)

total_patients = 0

for name, hospital in hospitals.items():

    labels = hospital["y"]

    total = len(labels)
    benign = int((labels == 0).sum())
    malignant = int((labels == 1).sum())

    malignant_percentage = (malignant / total) * 100

    print(f"\nHospital: {name}")
    print("Total patients:", total)
    print("Benign patients:", benign)
    print("Malignant patients:", malignant)
    print(f"Malignant percentage: {malignant_percentage:.2f}%")

    total_patients += total

# STEP 9: Verify the patient distribution

print("\n" + "=" * 55)
print("DATA VERIFICATION")
print("=" * 55)

all_indices = np.concatenate(hospital_indices)

# All 455 training patients must be included
assert len(all_indices) == len(X_train)

# No training patient should appear in two hospitals
assert len(np.unique(all_indices)) == len(X_train)

# None of the 114 test patients should appear in a hospital
assert set(all_indices).isdisjoint(set(X_test.index))

# All three hospitals together must contain the full training set
assert total_patients == len(X_train)

print("Total patients across hospitals:", total_patients)
print("No duplicate patients across hospitals.")
print("No test patients included in hospital training data.")
print("All verification checks passed!")

print("\nDAY 5 COMPLETED SUCCESSFULLY!")

# ============================================================
# DAY 5 - ADVANCED HOSPITAL HETEROGENEITY ANALYSIS
# ============================================================

from scipy.spatial.distance import jensenshannon
from pathlib import Path

# Create metrics folder
metrics_folder = Path(__file__).resolve().parent / "metrics"
metrics_folder.mkdir(exist_ok=True)

hospital_summary = []

# Collect hospital statistics
for name, hospital in hospitals.items():

    X_hospital = hospital["X"]
    y_hospital = hospital["y"]

    benign_count = int((y_hospital == 0).sum())
    malignant_count = int((y_hospital == 1).sum())
    total = len(y_hospital)

    hospital_summary.append({
        "hospital": name,
        "total_patients": total,
        "benign_patients": benign_count,
        "malignant_patients": malignant_count,
        "malignant_percentage": round(
            malignant_count / total * 100, 2
        )
    })

    # Data integrity checks
    assert X_hospital.shape[1] == 30
    assert not X_hospital.isnull().any().any()
    assert not y_hospital.isnull().any()

# Save hospital summary
summary_df = pd.DataFrame(hospital_summary)

summary_df.to_csv(
    metrics_folder / "hospital_distribution.csv",
    index=False
)

print("\nHOSPITAL DISTRIBUTION SUMMARY")
print(summary_df.to_string(index=False))

# Calculate pairwise Jensen-Shannon divergence
print("\nHOSPITAL HETEROGENEITY ANALYSIS")

jsd_results = []

for i in range(len(hospital_names)):
    for j in range(i + 1, len(hospital_names)):

        hospital_a = hospital_names[i]
        hospital_b = hospital_names[j]

        y_a = hospitals[hospital_a]["y"]
        y_b = hospitals[hospital_b]["y"]

        # Probability distributions: [benign, malignant]
        p = np.array([
            (y_a == 0).mean(),
            (y_a == 1).mean()
        ])

        q = np.array([
            (y_b == 0).mean(),
            (y_b == 1).mean()
        ])

        # scipy returns Jensen-Shannon distance
        js_distance = jensenshannon(p, q, base=2)

        # Squaring gives Jensen-Shannon divergence
        js_divergence = js_distance ** 2

        jsd_results.append({
            "hospital_1": hospital_a,
            "hospital_2": hospital_b,
            "js_divergence": round(
                float(js_divergence), 6
            )
        })

        print(
            f"{hospital_a} vs {hospital_b}: "
            f"JSD = {js_divergence:.6f}"
        )

# Save divergence results
jsd_df = pd.DataFrame(jsd_results)

jsd_df.to_csv(
    metrics_folder / "hospital_heterogeneity.csv",
    index=False
)

print("\nHospital distribution and heterogeneity reports saved!")
print("DAY 5 ADVANCED ANALYSIS COMPLETED!")