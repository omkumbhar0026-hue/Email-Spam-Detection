import os
import pandas as pd
from datetime import datetime
import subprocess
import glob


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
    # 1. Load ground truth
    gt_path = "data/test.csv"
    if not os.path.exists(gt_path):
        raise Exception("data/test.csv not found")

    gt = pd.read_csv(gt_path)

    if "target" not in gt.columns:
        raise Exception("test.csv must contain 'target' column")

    # 2. Get submission file (IGNORE sample file)
    files = glob.glob("submissions/*.csv")

    files = [f for f in files if "sample_submission" not in f.lower()]

    if not files:
        raise Exception("No valid submission file found")

    # Pick most recent file
    latest_file = max(files, key=os.path.getmtime)

    print("Available files:", files)
    print("Selected file:", latest_file)

    sub = pd.read_csv(latest_file)

    # 3. Validate
    if "prediction" not in sub.columns:
        raise Exception("Submission must contain 'prediction' column")

    if len(sub) != len(gt):
        raise Exception(
            f"Row mismatch: submission={len(sub)}, test={len(gt)}"
        )

    # 4. Accuracy
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

    leaderboard = leaderboard.sort_values(by="accuracy", ascending=False)
    leaderboard.to_csv(leaderboard_file, index=False)

    print(f"{username} | Accuracy: {accuracy:.4f}")
    print("Leaderboard updated successfully.")


if __name__ == "__main__":
    main()
