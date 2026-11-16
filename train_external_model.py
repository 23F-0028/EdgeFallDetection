
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

# Configuration
DATA_DIR = Path("external_processed_data")
OUTPUT_DIR = Path("external_model_output")

RANDOM_SEED = 42
EPOCHS = 40
BATCH_SIZE = 16

LABEL_MAP = {
    "Falling": 0,
    "Lying": 1,
    "Running": 2,
    "Sitting": 3,
    "Standing": 4,
    "Walking": 5
}

CLASS_NAMES = list(LABEL_MAP.keys())

np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

OUTPUT_DIR.mkdir(exist_ok=True)

# Load datasets
print("\nLOADING EXTERNAL DATASET")

X_train = np.load(DATA_DIR / "X_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")

y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# Class weights
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

for class_id, weight in class_weights.items():
    print(f"{CLASS_NAMES[class_id]}: {weight:.4f}")

# Build the existing 1D-CNN architecture
print("\nBUILDING 1D-CNN MODEL")

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

    tf.keras.layers.Dense(6, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

# Train model
print("\nSTARTING MODEL TRAINING")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1
)

# Save model
model.save(
    OUTPUT_DIR / "external_activity_model.keras"
)

print("\nMODEL SAVED")

# Final evaluation on untouched test data
print("\nFINAL TEST EVALUATION")

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

# Classification report
report = classification_report(
    y_test,
    y_pred,
    labels=list(range(6)),
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as file:
    file.write(report)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(6))
)

print("\nCONFUSION MATRIX")
print(cm)

plt.figure(figsize=(9, 7))
plt.imshow(cm, interpolation="nearest")
plt.title("External Dataset - 1D-CNN Confusion Matrix")
plt.colorbar()

positions = np.arange(6)

plt.xticks(positions, CLASS_NAMES, rotation=30)
plt.yticks(positions, CLASS_NAMES)

plt.xlabel("Predicted Activity")
plt.ylabel("Actual Activity")

for i in range(6):
    for j in range(6):
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

# Training history
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history["accuracy"], label="Training")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["loss"], label="Training")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "training_history.png",
    dpi=150
)
plt.close()

# Save metrics
metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(accuracy),
    "test_accuracy_percent": float(accuracy * 100),
    "epochs_completed": len(history.history["loss"]),
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "testing_samples": int(len(X_test)),
    "class_names": CLASS_NAMES
}

with open(
    OUTPUT_DIR / "evaluation_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\nTRAINING COMPLETED")
print("Model:", OUTPUT_DIR / "external_activity_model.keras")
print("Output folder:", OUTPUT_DIR.resolve())
