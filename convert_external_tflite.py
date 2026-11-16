
import json
import numpy as np
import tensorflow as tf
import joblib

from pathlib import Path

MODEL_PATH = Path(
    "external_model_output/external_activity_model.keras"
)

DATA_DIR = Path("external_processed_data")
OUTPUT_DIR = Path("external_model_output")

print("Loading trained CNN...")
model = tf.keras.models.load_model(MODEL_PATH)

X_train = np.load(DATA_DIR / "X_train.npy")

# Save normalization values for ESP32
scaler = joblib.load(DATA_DIR / "scaler.pkl")

normalization = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "class_names": [
        "Falling", "Lying", "Running",
        "Sitting", "Standing", "Walking"
    ]
}

with open(OUTPUT_DIR / "normalization.json", "w") as f:
    json.dump(normalization, f, indent=4)

print("Normalization saved.")

# Representative dataset for INT8 quantization
def representative_dataset():
    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(X_train),
        size=min(500, len(X_train)),
        replace=False
    )

    for index in indices:
        yield [X_train[index:index + 1].astype(np.float32)]

print("Converting model to INT8 TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

tflite_path = OUTPUT_DIR / "external_activity_model_int8.tflite"

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("\nTFLITE MODEL CREATED")
print("File:", tflite_path)
print("Size:", len(tflite_model), "bytes")

# Verify input and output details
interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

print("\nINPUT DETAILS:")
print(interpreter.get_input_details())

print("\nOUTPUT DETAILS:")
print(interpreter.get_output_details())

print("\nCONVERSION COMPLETED")
