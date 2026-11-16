
import csv
from collections import Counter

file_name = "activity_dataset.csv"

with open(file_name, "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

print("DATASET VERIFICATION")
print("--------------------")

print("Total samples:", len(rows))
print("\nActivity distribution:")

counts = Counter(row["activity"] for row in rows)

for activity, count in counts.items():
    print(f"{activity}: {count}")

print("\nCSV columns:")
print(reader.fieldnames)

print("\nMissing values:", sum(
    1 for row in rows
    for value in row.values()
    if value is None or value.strip() == ""
))

print("\nDataset verification completed.")
