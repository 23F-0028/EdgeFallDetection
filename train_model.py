
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("processed_data")
OUTPUT_DIR = Path("model_output")

RANDOM_SEED = 42
EPOCHS = 40
BATCH_SIZE = 16

LABEL_MAP = {
    "sitting": 0,
    "standing": 1,
    "walking": 2,
    "fall": 3
}

CLASS_NAMES = list(LABEL_MAP.keys())

np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

# -----------------------------
# Load processed dataset
# -----------------------------
print("\nLOADING PROCESSED DATA")
print("------------------------------")

X_train = np.load(DATA_DIR / "X_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")

y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

groups_train = np.load(DATA_DIR / "groups_train.npy")
groups_test = np.load(DATA_DIR / "groups_test.npy")

# Confirm no session leakage
if set(groups_train).intersection(set(groups_test)):
    raise ValueError("Data leakage detected!")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nSession leakage check: PASSED")

# -----------------------------
# Calculate class weights
# -----------------------------
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("\nCLASS WEIGHTS")
print("------------------------------")

for class_id, weight in class_weights.items():
    print(f"{CLASS_NAMES[class_id]}: {weight:.4f}")

# -----------------------------
# Build 1D-CNN model
# -----------------------------
print("\nBUILDING 1D-CNN MODEL")
print("------------------------------")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(50, 6)),

    tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train model
# -----------------------------
print("\nSTARTING MODEL TRAINING")
print("------------------------------")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    shuffle=True,
    verbose=1
)

# -----------------------------
# Save trained model
# -----------------------------
OUTPUT_DIR.mkdir(exist_ok=True)

model.save(OUTPUT_DIR / "activity_fall_model.keras")

print("\nMODEL SAVED")
print("------------------------------")
print(OUTPUT_DIR / "activity_fall_model.keras")

# -----------------------------
# Evaluate on test dataset
# -----------------------------
print("\nMODEL EVALUATION")
print("------------------------------")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

y_probabilities = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(y_probabilities, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Accuracy (%): {accuracy * 100:.2f}%")

# -----------------------------
# Classification report
# -----------------------------
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print("------------------------------")
print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as file:
    file.write(report)

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3]
)

print("\nCONFUSION MATRIX")
print("------------------------------")
print(cm)

plt.figure(figsize=(8, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("1D-CNN Confusion Matrix")
plt.colorbar()

positions = np.arange(len(CLASS_NAMES))

plt.xticks(positions, CLASS_NAMES, rotation=30)
plt.yticks(positions, CLASS_NAMES)

plt.xlabel("Predicted Activity")
plt.ylabel("Actual Activity")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=150
)
plt.close()

# -----------------------------
# Training history plot
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.title("1D-CNN Training History")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "training_history.png",
    dpi=150
)
plt.close()

# -----------------------------
# Save evaluation metrics
# -----------------------------
metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(accuracy),
    "test_accuracy_percent": float(accuracy * 100),
    "epochs_completed": len(history.history["loss"]),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "class_names": CLASS_NAMES
}

with open(
    OUTPUT_DIR / "evaluation_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

# -----------------------------
# Final results
# -----------------------------
print("\nTRAINING COMPLETED")
print("------------------------------")
print("Model:", OUTPUT_DIR / "activity_fall_model.keras")
print("Confusion matrix saved.")
print("Training history saved.")
print("Classification report saved.")
print("Evaluation metrics saved.")
print("\nOutput folder:", OUTPUT_DIR.resolve())
