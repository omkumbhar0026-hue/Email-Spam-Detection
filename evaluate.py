import os
import pandas as pd
from datetime import datetime
import subprocess


def get_git_revision_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def main():
    # 1. Load ground truth (must contain 'target')
    gt_path = "data/test.csv"
    if not os.path.exists(gt_path):
        raise Exception("data/test.csv not found")

    gt = pd.read_csv(gt_path)

    if "target" not in gt.columns:
        raise Exception("test.csv must contain 'target' column")

    # 2. Get latest submission file
    submission_folder = "submissions"
    if not os.path.exists(submission_folder):
        raise Exception("submissions folder not found")

    files = [f for f in os.listdir(submission_folder) if f.endswith(".csv")]

    if not files:
        raise Exception("No submission file found in submissions/")

    # ✅ FIX: pick most recently modified file (not alphabetical)
    latest_file = max(
        files,
        key=lambda x: os.path.getmtime(os.path.join(submission_folder, x))
    )

    sub_path = os.path.join(submission_folder, latest_file)

    print("Files found:", files)
    print("Selected file:", latest_file)

    sub = pd.read_csv(sub_path)

    # 3. Validate submission format
    if "prediction" not in sub.columns:
        raise Exception("Submission must contain 'prediction' column")

    if len(sub) != len(gt):
        raise Exception("Row count mismatch between submission and test data")

    # 4. Calculate accuracy
    accuracy = (
        sub["prediction"].astype(int) == gt["target"].astype(int)
    ).mean()

    # 5. Metadata
    username = os.getenv("GITHUB_ACTOR") or "unknown"
    commit_hash = get_git_revision_hash()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # 6. Update leaderboard
    leaderboard_file = "leaderboard.csv"

    new_entry = pd.DataFrame([{
        "username": username,
        "accuracy": round(accuracy, 4),
        "commit_hash": commit_hash,
        "timestamp": timestamp
    }])

    if os.path.exists(leaderboard_file):
        leaderboard = pd.read_csv(leaderboard_file)
        leaderboard = pd.concat([leaderboard, new_entry], ignore_index=True)
    else:
        leaderboard = new_entry

    # Sort by accuracy (highest first)
    leaderboard = leaderboard.sort_values(by="accuracy", ascending=False)

    leaderboard.to_csv(leaderboard_file, index=False)

    print(f"{username} | Accuracy: {accuracy:.4f}")
    print("Leaderboard updated successfully.")


if __name__ == "__main__":
    main()
