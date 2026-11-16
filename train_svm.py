
import json
import joblib
import numpy as np

from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_DIR = Path("processed_data")
OUTPUT_DIR = Path("svm_output")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["sitting", "standing", "walking", "fall"]


def extract_features(X):
    features = []

    for window in X:
        row = []

        for axis in range(window.shape[1]):
            signal = window[:, axis]

            row.extend([
                np.mean(signal),
                np.std(signal),
                np.min(signal),
                np.max(signal),
                np.median(signal),
                np.ptp(signal),
                np.sqrt(np.mean(signal ** 2)),
                np.mean(np.abs(signal)),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                np.mean(np.abs(np.diff(signal))),
                np.std(np.diff(signal))
            ])

        features.append(row)

    return np.asarray(features, dtype=np.float32)


print("\nLOADING DATA")

X_train = np.load(DATA_DIR / "X_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

groups_train = np.load(DATA_DIR / "groups_train.npy")
groups_test = np.load(DATA_DIR / "groups_test.npy")

if set(groups_train).intersection(set(groups_test)):
    raise RuntimeError("Session leakage detected!")

print("Session leakage check: PASSED")

print("\nEXTRACTING FEATURES")

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

print("Training features:", X_train_features.shape)
print("Testing features:", X_test_features.shape)

print("\nBUILDING SVM MODEL")

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced"
)

print("\nSTARTING TRAINING")

model.fit(X_train_features, y_train)

print("\nEVALUATING MODEL")

predictions = model.predict(X_test_features)

accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2, 3]
)

print("\nTEST ACCURACY:", round(accuracy * 100, 2), "%")
print("\nCLASSIFICATION REPORT")
print(report)

print("CONFUSION MATRIX")
print(matrix)

joblib.dump(model, OUTPUT_DIR / "svm_model.pkl")

(OUTPUT_DIR / "classification_report.txt").write_text(report)

np.savetxt(
    OUTPUT_DIR / "confusion_matrix.csv",
    matrix,
    delimiter=",",
    fmt="%d"
)

with open(OUTPUT_DIR / "evaluation_metrics.json", "w") as file:
    json.dump({"accuracy": float(accuracy)}, file, indent=4)

print("\nSVM TRAINING COMPLETED")
print("Results saved in:", OUTPUT_DIR.resolve())
