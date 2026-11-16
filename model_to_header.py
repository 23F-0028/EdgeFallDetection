
from pathlib import Path

MODEL_PATH = Path("model_output/activity_fall_model_int8.tflite")
HEADER_PATH = Path("model_output/model_data.h")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model_bytes = MODEL_PATH.read_bytes()

with open(HEADER_PATH, "w") as file:
    file.write("#ifndef MODEL_DATA_H\n")
    file.write("#define MODEL_DATA_H\n\n")
    file.write("#include <Arduino.h>\n\n")

    file.write(
        "const unsigned char activity_fall_model_int8[] PROGMEM = {\n"
    )

    for i in range(0, len(model_bytes), 12):
        chunk = model_bytes[i:i + 12]
        line = ", ".join(f"0x{byte:02x}" for byte in chunk)
        file.write("    " + line + ",\n")

    file.write("};\n\n")

    file.write(
        f"const unsigned int activity_fall_model_int8_len = "
        f"{len(model_bytes)};\n\n"
    )

    file.write("#endif\n")

print("MODEL HEADER CREATED SUCCESSFULLY")
print("Header:", HEADER_PATH.resolve())
print("Model bytes:", len(model_bytes))
print("Header size:", f"{HEADER_PATH.stat().st_size / 1024:.2f} KB")
