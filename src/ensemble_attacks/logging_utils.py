import csv
import os
from datetime import datetime


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_csv_row(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def now_timestamp():
    return datetime.now().isoformat(timespec="seconds")
