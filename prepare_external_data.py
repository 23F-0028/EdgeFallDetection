
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Configuration
WINDOW_SIZE = 50
STRIDE = 25

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

LABEL_MAP = {
    "Falling": 0,
    "Lying": 1,
    "Running": 2,
    "Sitting": 3,
    "Standing": 4,
    "Walking": 5
}

OUTPUT_DIR = Path("external_processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_windows(file_path):
    df = pd.read_csv(file_path)

    X = []
    y = []

    df["segment"] = (
        df["class"] != df["class"].shift()
    ).cumsum()

    for _, segment in df.groupby("segment"):
        activity = segment["class"].iloc[0]

        if activity not in LABEL_MAP:
            raise ValueError(f"Unknown activity: {activity}")

        values = segment[FEATURES].to_numpy(
            dtype=np.float32
        )

        label = LABEL_MAP[activity]

        for start in range(
            0,
            len(values) - WINDOW_SIZE + 1,
            STRIDE
        ):
            window = values[
                start:start + WINDOW_SIZE
            ]

            X.append(window)
            y.append(label)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64)
    )


print("Creating windows...")

X_train, y_train = create_windows("train.csv")
X_val, y_val = create_windows("validation.csv")
X_test, y_test = create_windows("test.csv")

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# Fit scaler ONLY on training data
scaler = StandardScaler()
scaler.fit(
    pd.read_csv("train.csv")[FEATURES].to_numpy(
        dtype=np.float32
    )
)

def scale_windows(X):
    original_shape = X.shape

    X = scaler.transform(
        X.reshape(-1, len(FEATURES))
    )

    return X.reshape(original_shape).astype(np.float32)


X_train = scale_windows(X_train)
X_val = scale_windows(X_val)
X_test = scale_windows(X_test)

# Save arrays
for name, data in {
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "X_test": X_test,
    "y_test": y_test
}.items():
    np.save(OUTPUT_DIR / f"{name}.npy", data)

joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

with open(OUTPUT_DIR / "label_map.json", "w") as f:
    json.dump(LABEL_MAP, f, indent=4)

print("\nCLASS WINDOW COUNTS")
for label, index in LABEL_MAP.items():
    print(
        label,
        "Train:", int(np.sum(y_train == index)),
        "Validation:", int(np.sum(y_val == index)),
        "Test:", int(np.sum(y_test == index))
    )

print("\nPREPROCESSING COMPLETED SUCCESSFULLY")
