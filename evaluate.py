import os
import subprocess
from datetime import datetime

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "num_words",
    "num_characters",
    "num_exclamation_marks",
    "num_links",
    "has_suspicious_link",
    "num_attachments",
    "has_attachment",
    "sender_reputation_score",
    "email_hour",
    "email_day_of_week",
    "is_weekend",
    "num_recipients",
    "contains_money_terms",
    "contains_urgency_terms",
]
TARGET = "target"


def get_git_revision_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def main():
    # 1. Load datasets
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    print(f"Loading training data from '{train_path}'...")
    df_train = pd.read_csv(train_path)

    print(f"Loading test data from '{test_path}'...")
    df_test = pd.read_csv(test_path)

    # Validate that target column has actual labels
    if df_train[TARGET].isnull().all() or df_test[TARGET].isnull().all():
        raise ValueError(
            "The 'target' column is entirely NaN in your CSV files.\n"
            "Please re-run the notebook's save cell using the fixed version:\n\n"
            "  train_df = X_train.copy().reset_index(drop=True)\n"
            "  train_df['target'] = y_train.reset_index(drop=True)\n"
            "  test_df = X_test.copy().reset_index(drop=True)\n"
            "  test_df['target'] = y_test.reset_index(drop=True)\n"
            "  train_df.to_csv('data/train.csv', index=False)\n"
            "  test_df.to_csv('data/test.csv', index=False)\n"
        )

    # 2. Prepare features and labels
    df_train = df_train.reset_index(drop=True).dropna(subset=[TARGET])
    df_test = df_test.reset_index(drop=True).dropna(subset=[TARGET])

    X_train = df_train[FEATURES]
    y_train = df_train[TARGET].astype(int)

    X_test = df_test[FEATURES]
    y_test = df_test[TARGET].astype(int)

    print(f"Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # 3. Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 4. Train model
    print("Training Gradient Boosting model...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_sc, y_train)

    # 5. Evaluate
    predictions = model.predict(X_test_sc)
    accuracy = accuracy_score(y_test, predictions)

    # 6. Update leaderboard
    leaderboard_file = "leaderboard.csv"
    commit_hash = get_git_revision_hash()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_entry = pd.DataFrame(
        [{"timestamp": timestamp, "commit_hash": commit_hash, "accuracy": accuracy}]
    )

    if os.path.exists(leaderboard_file):
        leaderboard = pd.read_csv(leaderboard_file)
        leaderboard = pd.concat([leaderboard, new_entry], ignore_index=True)
    else:
        leaderboard = new_entry

    leaderboard = leaderboard.sort_values(by="accuracy", ascending=False)
    leaderboard.to_csv(leaderboard_file, index=False)

    print(
        f"Evaluated commit {commit_hash}. Accuracy: {accuracy:.4f}. Leaderboard updated."
    )


if __name__ == "__main__":
    main()
