
import serial
import csv
import os
from datetime import datetime

SERIAL_PORT = "COM7"
BAUD_RATE = 115200
CSV_FILE = "activity_dataset.csv"

SAMPLES_PER_TRIAL = 100
TOTAL_TRIALS = 10

file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)

    print("Fall-like Movement Data Collection")
    print("----------------------------------")
    print("Trials:", TOTAL_TRIALS)
    print("Samples per trial:", SAMPLES_PER_TRIAL)

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp", "sample_id",
                "accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z",
                "activity"
            ])

        sample_id = 1

        for trial in range(1, TOTAL_TRIALS + 1):
            input(f"\nTrial {trial}/{TOTAL_TRIALS}: Press ENTER when ready...")

            ser.reset_input_buffer()

            print("RECORDING NOW - perform your safe controlled movement.")

            collected = 0

            while collected < SAMPLES_PER_TRIAL:
                line = ser.readline().decode(
                    "utf-8", errors="ignore"
                ).strip()

                values = line.split(",")

                if len(values) != 6:
                    continue

                try:
                    sensor_values = [int(value) for value in values]
                except ValueError:
                    continue

                writer.writerow([
                    datetime.now().isoformat(),
                    sample_id,
                    *sensor_values,
                    "fall"
                ])

                sample_id += 1
                collected += 1

            file.flush()
            print("Trial completed:", collected, "samples")

    ser.close()

    print("\nFall-like data collection completed!")
    print("Total samples:", sample_id - 1)
    print("Saved in:", CSV_FILE)

except serial.SerialException as error:
    print("Serial connection error:", error)
    print("Check ESP32 connection and COM port.")
