
import json
import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import accuracy_score

# Paths
MODEL_PATH = Path("model_output/activity_fall_model.keras")
DATA_DIR = Path("processed_data")
OUTPUT_DIR = Path("model_output")

OUTPUT_DIR.mkdir(exist_ok=True)

# Load trained model and processed data
print("\nLOADING TRAINED MODEL")
model = tf.keras.models.load_model(MODEL_PATH)

X_train = np.load(DATA_DIR / "X_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

# Load original scaler
scaler = joblib.load(DATA_DIR / "scaler.pkl")

# Representative dataset for INT8 quantization
def representative_dataset():
    for i in range(min(200, len(X_train))):
        sample = X_train[i:i + 1].astype(np.float32)
        yield [sample]

# Convert model
print("\nCONVERTING MODEL TO INT8 TFLITE")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save TFLite model
tflite_path = OUTPUT_DIR / "activity_fall_model_int8.tflite"

with open(tflite_path, "wb") as file:
    file.write(tflite_model)

print("TFLite model saved:", tflite_path)

# Test converted model
print("\nTESTING QUANTIZED MODEL")

interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

if input_scale == 0 or output_scale == 0:
    raise ValueError("Invalid quantization parameters.")

predictions = []

for sample in X_test:

    sample = sample[np.newaxis, ...].astype(np.float32)

    quantized_input = np.round(
        sample / input_scale + input_zero_point
    )

    quantized_input = np.clip(
        quantized_input, -128, 127
    ).astype(np.int8)

    interpreter.set_tensor(
        input_details[0]["index"],
        quantized_input
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    predictions.append(int(np.argmax(output[0])))

predictions = np.asarray(predictions)

accuracy = accuracy_score(y_test, predictions)

print("\nQUANTIZED MODEL RESULTS")
print("------------------------------")
print("Test accuracy:", round(accuracy * 100, 2), "%")
print("Input shape:", input_details[0]["shape"])
print("Input type:", input_details[0]["dtype"])
print("Output type:", output_details[0]["dtype"])
print("Input scale:", input_scale)
print("Input zero point:", input_zero_point)
print("Output scale:", output_scale)
print("Output zero point:", output_zero_point)

# Save normalization information for ESP32
normalization = {
    "features": [
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z"
    ],
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "input_quantization_scale": float(input_scale),
    "input_zero_point": int(input_zero_point),
    "output_quantization_scale": float(output_scale),
    "output_zero_point": int(output_zero_point),
    "labels": {
        "0": "sitting",
        "1": "standing",
        "2": "walking",
        "3": "fall"
    }
}

with open(OUTPUT_DIR / "normalization.json", "w") as file:
    json.dump(normalization, file, indent=4)

# File size
size_kb = tflite_path.stat().st_size / 1024

print("\nCONVERSION COMPLETED")
print("------------------------------")
print(f"TFLite model size: {size_kb:.2f} KB")
print("Normalization information saved.")
print("Output folder:", OUTPUT_DIR.resolve())
