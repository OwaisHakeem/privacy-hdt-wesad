from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
RAW_DIR = Path("data/raw/WESAD")
OUTPUT_DIR = Path("data/processed")
CLIENT_DIR = Path("data/processed/clients")
TABLE_DIR = Path("results/tables")
LOG_DIR = Path("results/logs")
SUMMARY_PATH = TABLE_DIR / "full_dataset_subject_summary.csv"
FEATURE_INFO_PATH = LOG_DIR / "full_dataset_feature_info.json"
OUTPUT_ALL = OUTPUT_DIR / "wesad_windows.csv"

CHEST_SAMPLING_RATE = 700
WINDOW_SECONDS = 10
CHEST_WINDOW_SIZE = CHEST_SAMPLING_RATE * WINDOW_SECONDS

# WESAD valid protocol labels
# 0 = undefined/transient, 5/6/7 = ignored
VALID_LABELS = {
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

TASKS = {
    "binary_baseline_stress": {
        1: 0,
        2: 1,
    },
    "multiclass_3_baseline_stress_amusement": {
        1: 0,
        2: 1,
        3: 2,
    },
    "multiclass_4_valid_conditions": {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
    },
}

TASK_LABEL_NAMES = {
    "binary_baseline_stress": {
        0: "baseline",
        1: "stress",
    },
    "multiclass_3_baseline_stress_amusement": {
        0: "baseline",
        1: "stress",
        2: "amusement",
    },
    "multiclass_4_valid_conditions": {
        0: "baseline",
        1: "stress",
        2: "amusement",
        3: "meditation",
    },
}

# Sampling rates from WESAD README for wrist E4 in the synchronised pkl
WRIST_SAMPLING_RATES = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
}


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as file:
        return pickle.load(file, encoding="latin1")


def flatten_signal(signal_array: np.ndarray) -> np.ndarray:
    array = np.asarray(signal_array)

    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]

    if array.ndim == 1:
        return array

    raise ValueError(f"Unexpected signal shape: {array.shape}")


def safe_stats(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_var": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_rms": np.nan,
        }

    q75, q25 = np.percentile(values, [75, 25])

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_var": float(np.var(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_rms": float(np.sqrt(np.mean(values ** 2))),
    }


def get_subject_files() -> list[tuple[str, Path]]:
    subject_files = []

    for subject_dir in sorted(RAW_DIR.glob("S*")):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        pkl_path = subject_dir / f"{subject_id}.pkl"

        if pkl_path.exists():
            subject_files.append((subject_id, pkl_path))

    return subject_files


def slice_by_seconds(signal: np.ndarray, sampling_rate: int, start_sec: float, end_sec: float) -> np.ndarray:
    start_index = int(round(start_sec * sampling_rate))
    end_index = int(round(end_sec * sampling_rate))

    start_index = max(start_index, 0)
    end_index = min(end_index, len(signal))

    if end_index <= start_index:
        return np.asarray([])

    return signal[start_index:end_index]


def extract_chest_signals(chest: dict, min_length: int) -> dict:
    ecg = flatten_signal(chest["ECG"])[:min_length]
    eda = flatten_signal(chest["EDA"])[:min_length]
    emg = flatten_signal(chest["EMG"])[:min_length]
    resp = flatten_signal(chest["Resp"])[:min_length]
    temp = flatten_signal(chest["Temp"])[:min_length]

    acc = np.asarray(chest["ACC"], dtype=np.float64)[:min_length]

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"Unexpected chest ACC shape: {acc.shape}")

    return {
        "chest_ecg": ecg,
        "chest_eda": eda,
        "chest_emg": emg,
        "chest_resp": resp,
        "chest_temp": temp,
        "chest_acc_x": acc[:, 0],
        "chest_acc_y": acc[:, 1],
        "chest_acc_z": acc[:, 2],
        "chest_acc_mag": np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2 + acc[:, 2] ** 2),
    }


def extract_wrist_signals(wrist: dict) -> dict:
    wrist_acc = np.asarray(wrist["ACC"], dtype=np.float64)

    if wrist_acc.ndim != 2 or wrist_acc.shape[1] != 3:
        raise ValueError(f"Unexpected wrist ACC shape: {wrist_acc.shape}")

    wrist_bvp = flatten_signal(wrist["BVP"])
    wrist_eda = flatten_signal(wrist["EDA"])
    wrist_temp = flatten_signal(wrist["TEMP"])

    return {
        "wrist_acc_x": wrist_acc[:, 0],
        "wrist_acc_y": wrist_acc[:, 1],
        "wrist_acc_z": wrist_acc[:, 2],
        "wrist_acc_mag": np.sqrt(wrist_acc[:, 0] ** 2 + wrist_acc[:, 1] ** 2 + wrist_acc[:, 2] ** 2),
        "wrist_bvp": wrist_bvp,
        "wrist_eda": wrist_eda,
        "wrist_temp": wrist_temp,
    }


def extract_subject(subject_id: str, pkl_path: Path) -> pd.DataFrame:
    data = load_pickle(pkl_path)

    labels = np.asarray(data["label"])
    chest = data["signal"]["chest"]
    wrist = data["signal"]["wrist"]

    min_length = min(
        len(labels),
        len(flatten_signal(chest["ECG"])),
        len(flatten_signal(chest["EDA"])),
        len(flatten_signal(chest["EMG"])),
        len(flatten_signal(chest["Resp"])),
        len(flatten_signal(chest["Temp"])),
        np.asarray(chest["ACC"]).shape[0],
    )

    labels = labels[:min_length]
    chest_signals = extract_chest_signals(chest, min_length)
    wrist_signals = extract_wrist_signals(wrist)

    rows = []
    window_id = 0

    for start in range(0, min_length - CHEST_WINDOW_SIZE + 1, CHEST_WINDOW_SIZE):
        end = start + CHEST_WINDOW_SIZE
        window_labels = labels[start:end]
        unique_labels = set(np.unique(window_labels).tolist())

        # Keep only pure-condition windows
        if len(unique_labels) != 1:
            continue

        original_label = int(next(iter(unique_labels)))

        if original_label not in VALID_LABELS:
            continue

        start_sec = start / CHEST_SAMPLING_RATE
        end_sec = end / CHEST_SAMPLING_RATE

        row = {
            "subject_id": subject_id,
            "client_id": f"client_{subject_id}",
            "window_id": window_id,
            "start_sample_chest": start,
            "end_sample_chest": end,
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "window_seconds": WINDOW_SECONDS,
            "original_wesad_label": original_label,
            "condition_name": VALID_LABELS[original_label],
        }

        # Chest features
        for signal_name, signal_values in chest_signals.items():
            row.update(safe_stats(signal_values[start:end], signal_name))

        # Wrist features aligned by seconds
        for signal_name, signal_values in wrist_signals.items():
            if signal_name.startswith("wrist_acc"):
                rate = WRIST_SAMPLING_RATES["ACC"]
            elif signal_name == "wrist_bvp":
                rate = WRIST_SAMPLING_RATES["BVP"]
            elif signal_name == "wrist_eda":
                rate = WRIST_SAMPLING_RATES["EDA"]
            elif signal_name == "wrist_temp":
                rate = WRIST_SAMPLING_RATES["TEMP"]
            else:
                raise ValueError(f"Unknown wrist signal: {signal_name}")

            window_values = slice_by_seconds(signal_values, rate, start_sec, end_sec)
            row.update(safe_stats(window_values, signal_name))

        rows.append(row)
        window_id += 1

    return pd.DataFrame(rows)


def add_task_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    for task_name, mapping in TASKS.items():
        output[f"{task_name}_label"] = output["original_wesad_label"].map(mapping)

        output[f"{task_name}_label_name"] = output[f"{task_name}_label"].map(
            TASK_LABEL_NAMES[task_name]
        )

        output[f"{task_name}_included"] = output[f"{task_name}_label"].notna()

    return output


def get_feature_columns(df: pd.DataFrame) -> dict:
    meta_columns = {
        "subject_id",
        "client_id",
        "window_id",
        "start_sample_chest",
        "end_sample_chest",
        "start_sec",
        "end_sec",
        "window_seconds",
        "original_wesad_label",
        "condition_name",
    }

    task_columns = {
        column for column in df.columns
        if column.endswith("_label")
        or column.endswith("_label_name")
        or column.endswith("_included")
    }

    feature_columns = [
        column for column in df.columns
        if column not in meta_columns
        and column not in task_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    chest_features = [column for column in feature_columns if column.startswith("chest_")]
    wrist_features = [column for column in feature_columns if column.startswith("wrist_")]

    return {
        "all_features": feature_columns,
        "chest_features": chest_features,
        "wrist_features": wrist_features,
        "num_all_features": len(feature_columns),
        "num_chest_features": len(chest_features),
        "num_wrist_features": len(wrist_features),
        "meta_columns": sorted(meta_columns),
        "task_definitions": TASKS,
    }


def save_task_specific_files(df: pd.DataFrame):
    for task_name in TASKS:
        task_df = df[df[f"{task_name}_included"]].copy()
        task_df[f"{task_name}_label"] = task_df[f"{task_name}_label"].astype(int)

        output_path = OUTPUT_DIR / f"{task_name}.csv"
        task_df.to_csv(output_path, index=False)

        for client_id, client_df in task_df.groupby("client_id"):
            task_client_dir = CLIENT_DIR / task_name
            task_client_dir.mkdir(parents=True, exist_ok=True)
            client_df.to_csv(task_client_dir / f"{client_id}.csv", index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIENT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    subject_files = get_subject_files()

    if not subject_files:
        raise FileNotFoundError(f"No subject files found in {RAW_DIR}")

    print("Building full WESAD feature dataset...")
    print(f"Subjects found: {[subject_id for subject_id, _ in subject_files]}")
    print(f"Window size: {WINDOW_SECONDS}s")
    print("Modalities: chest + wrist")
    print("Tasks: binary, 3-class, 4-class")

    subject_frames = []
    summary_rows = []

    for subject_id, pkl_path in subject_files:
        print(f"\nProcessing {subject_id}: {pkl_path}")
        subject_df = extract_subject(subject_id, pkl_path)

        if subject_df.empty:
            print(f"WARNING: no valid windows for {subject_id}")
            continue

        subject_frames.append(subject_df)

        counts = subject_df["condition_name"].value_counts().to_dict()

        summary_rows.append(
            {
                "subject_id": subject_id,
                "client_id": f"client_{subject_id}",
                "total_valid_windows": int(len(subject_df)),
                "baseline_windows": int(counts.get("baseline", 0)),
                "stress_windows": int(counts.get("stress", 0)),
                "amusement_windows": int(counts.get("amusement", 0)),
                "meditation_windows": int(counts.get("meditation", 0)),
            }
        )

        print(f"Windows: {len(subject_df)} | Counts: {counts}")

    if not subject_frames:
        raise RuntimeError("No subject data extracted.")

    full_df = pd.concat(subject_frames, ignore_index=True)
    full_df = add_task_columns(full_df)

    full_df.to_csv(OUTPUT_ALL, index=False)

    save_task_specific_files(full_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    feature_info = get_feature_columns(full_df)

    with open(FEATURE_INFO_PATH, "w", encoding="utf-8") as file:
        json.dump(feature_info, file, indent=4)

    print("\nFull WESAD dataset build completed.")
    print(f"Full feature file: {OUTPUT_ALL}")
    print(f"Subject summary: {SUMMARY_PATH}")
    print(f"Feature info: {FEATURE_INFO_PATH}")

    print("\nOverall condition distribution:")
    print(full_df["condition_name"].value_counts())

    print("\nTask-specific rows:")
    for task_name in TASKS:
        count = int(full_df[f"{task_name}_included"].sum())
        print(f"{task_name}: {count}")

    print("\nFeature counts:")
    print(f"All features: {feature_info['num_all_features']}")
    print(f"Chest features: {feature_info['num_chest_features']}")
    print(f"Wrist features: {feature_info['num_wrist_features']}")


if __name__ == "__main__":
    main()
