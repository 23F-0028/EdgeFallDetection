
import json
import math
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Configuration
# -----------------------------
DATA_FILE = Path("activity_dataset_consistent.csv")
OUTPUT_DIR = Path("processed_data")

WINDOW_SIZE = 50
STRIDE = 25
GAP_THRESHOLD_SECONDS = 1.0
RANDOM_SEED = 42

FEATURES = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z"
]

LABEL_MAP = {
    "sitting": 0,
    "standing": 1,
    "walking": 2,
    "fall": 3
}


# -----------------------------
# Load dataset
# -----------------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE.resolve()}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = ["timestamp", "activity"] + FEATURES

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df["activity"] = df["activity"].str.strip().str.lower()

unknown_labels = set(df["activity"]) - set(LABEL_MAP)

if unknown_labels:
    raise ValueError(f"Unknown labels: {unknown_labels}")

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

for feature in FEATURES:
    df[feature] = pd.to_numeric(
        df[feature], errors="coerce"
    )

df = df.dropna(
    subset=["timestamp", "activity"] + FEATURES
).reset_index(drop=True)

print("\nDATASET LOADED")
print("------------------------------")
print("Total valid samples:", len(df))
print("\nActivity counts:")
print(df["activity"].value_counts())


# -----------------------------
# Detect recording sessions
# -----------------------------
time_gap = df["timestamp"].diff().dt.total_seconds()

new_session = (
    time_gap.isna()
    | (time_gap > GAP_THRESHOLD_SECONDS)
    | (df["activity"] != df["activity"].shift())
)

df["session_id"] = new_session.cumsum() - 1

print("\nRecording sessions:", df["session_id"].nunique())


# -----------------------------
# Create sensor windows
# -----------------------------
X = []
y = []
groups = []

for session_id, session in df.groupby("session_id", sort=False):

    values = session[FEATURES].to_numpy(dtype=np.float32)

    label_name = session["activity"].iloc[0]
    label_id = LABEL_MAP[label_name]

    if len(values) < WINDOW_SIZE:
        continue

    for start in range(
        0, len(values) - WINDOW_SIZE + 1, STRIDE
    ):
        window = values[start:start + WINDOW_SIZE]

        X.append(window)
        y.append(label_id)
        groups.append(int(session_id))


if not X:
    raise ValueError("No windows created. Check your dataset.")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)
groups = np.asarray(groups, dtype=np.int64)

print("\nWINDOWING RESULTS")
print("------------------------------")
print("Window size:", WINDOW_SIZE)
print("Stride:", STRIDE)
print("X shape:", X.shape)
print("y shape:", y.shape)


# -----------------------------
# Split by recording sessions
# -----------------------------
rng = np.random.default_rng(RANDOM_SEED)

train_indices = []
test_indices = []

for class_id in sorted(LABEL_MAP.values()):

    class_groups = np.unique(groups[y == class_id])

    if len(class_groups) < 2:
        raise ValueError(
            f"Class {class_id} has fewer than 2 sessions. "
            "Collect more recordings."
        )

    shuffled_groups = rng.permutation(class_groups)

    test_count = max(
        1, math.ceil(len(shuffled_groups) * 0.20)
    )

    test_groups = shuffled_groups[:test_count]
    train_groups = shuffled_groups[test_count:]

    train_indices.extend(
        np.where(
            np.isin(groups, train_groups) & (y == class_id)
        )[0]
    )

    test_indices.extend(
        np.where(
            np.isin(groups, test_groups) & (y == class_id)
        )[0]
    )


train_indices = np.asarray(train_indices)
test_indices = np.asarray(test_indices)

rng.shuffle(train_indices)
rng.shuffle(test_indices)

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

groups_train = groups[train_indices]
groups_test = groups[test_indices]

# Check for data leakage
if set(groups_train).intersection(set(groups_test)):
    raise RuntimeError(
        "Data leakage detected between training and testing!"
    )


# -----------------------------
# Normalize sensor features
# -----------------------------
scaler = StandardScaler()

scaler.fit(
    X_train.reshape(-1, len(FEATURES))
)

X_train = scaler.transform(
    X_train.reshape(-1, len(FEATURES))
).reshape(X_train.shape).astype(np.float32)

X_test = scaler.transform(
    X_test.reshape(-1, len(FEATURES))
).reshape(X_test.shape).astype(np.float32)


# -----------------------------
# Save processed dataset
# -----------------------------
OUTPUT_DIR.mkdir(exist_ok=True)

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "X_test.npy", X_test)

np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

# Save recording session IDs
np.save(OUTPUT_DIR / "groups_train.npy", groups_train)
np.save(OUTPUT_DIR / "groups_test.npy", groups_test)

# Save normalization scaler
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

# Save activity label mapping
with open(OUTPUT_DIR / "label_map.json", "w") as file:
    json.dump(LABEL_MAP, file, indent=4)


# -----------------------------
# Final report
# -----------------------------
print("\nPREPROCESSING COMPLETED")
print("------------------------------")

print("Training windows:", len(X_train))
print("Testing windows:", len(X_test))

print("\nTraining class distribution:")

for name, class_id in LABEL_MAP.items():
    print(f"{name}: {np.sum(y_train == class_id)}")

print("\nTesting class distribution:")

for name, class_id in LABEL_MAP.items():
    print(f"{name}: {np.sum(y_test == class_id)}")

print("\nFinal array shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nSession group files:")
print("groups_train:", groups_train.shape)
print("groups_test:", groups_test.shape)

print("\nData leakage check: PASSED")

print("\nSaved files in:", OUTPUT_DIR.resolve())
