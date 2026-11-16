
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = "external_processed_data/"
MODEL_PATH = "external_model_output/external_activity_model_int8.tflite"

X_test = np.load(DATA_DIR + "X_test.npy")
y_test = np.load(DATA_DIR + "y_test.npy")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero = input_details["quantization"]
output_scale, output_zero = output_details["quantization"]

predictions = []

for sample in X_test:
    sample = sample[np.newaxis, :, :]

    quantized_input = np.round(
        sample / input_scale + input_zero
    )

    quantized_input = np.clip(
        quantized_input, -128, 127
    ).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        quantized_input
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details["index"]
    )

    probabilities = (
        output.astype(np.float32) - output_zero
    ) * output_scale

    predictions.append(np.argmax(probabilities))

predictions = np.array(predictions)

accuracy = accuracy_score(y_test, predictions)

print("\nINT8 TFLITE TEST RESULTS")
print("------------------------------")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    target_names=[
        "Falling", "Lying", "Running",
        "Sitting", "Standing", "Walking"
    ],
    zero_division=0
))
