
import serial
import csv
import os
import time
from datetime import datetime

# ESP32 serial configuration
SERIAL_PORT = "COM7"
BAUD_RATE = 115200

# CSV dataset file
CSV_FILE = "activity_dataset_consistent.csv"

# Ask activity label
print("Activity Data Collection System")
print("--------------------------------")
print("Available labels: sitting, standing, walking, fall")

label = input("Enter activity label: ").strip().lower()

if label not in ["sitting", "standing", "walking", "fall"]:
    print("Invalid label. Please run the program again.")
    exit()

duration = int(input("Enter collection duration in seconds (e.g. 30): "))

file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    print("\nConnecting to ESP32...")
    time.sleep(2)
    ser.reset_input_buffer()

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "sample_id",
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "activity"
            ])

        start_time = time.time()
        sample_id = 1

        print(f"\nCollecting {label} data for {duration} seconds...")

        while time.time() - start_time < duration:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

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
                label
            ])

            sample_id += 1

    ser.close()

    print("\nData collection completed!")
    print("Activity:", label)
    print("Samples collected:", sample_id - 1)
    print("CSV file:", CSV_FILE)

except serial.SerialException as error:
    print("Serial connection error:", error)
    print("Check COM6 and close Arduino Serial Monitor.")
