import os


def check_csv_fin(file_path="/data/features_financials.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[dep_check] Missing: {file_path}")
    print(f"[dep_check] OK: {file_path}")


if __name__ == "__main__":
    check_csv_fin()
