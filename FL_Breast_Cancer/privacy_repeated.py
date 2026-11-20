
# ============================================================
# REPEATED PRIVACY-UTILITY EXPERIMENT
# FIVE RANDOM SEEDS
# ============================================================

from pathlib import Path

import pandas as pd

from privacy_experiments import (
    prepare_validation_data,
    train_model,
    evaluate_model
)


BASE_DIR = Path(__file__).resolve().parent

METRICS_DIR = BASE_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 1: DEFINE EXPERIMENT SETTINGS
# ============================================================

NOISE_LEVELS = [0.0, 1.0, 2.0, 4.0, 8.0]

SEEDS = [11, 22, 33, 44, 55]

ROUNDS = 30

LEARNING_RATE = 0.05

CLIPPING_NORM = 1.0

DELTA = 1e-5


# ============================================================
# STEP 2: REPEAT TRAINING
# ============================================================

def run_repeated_experiments():

    X_fit, y_fit, X_val, y_val = (
        prepare_validation_data()
    )

    all_results = []

    print("\nREPEATED PRIVACY-UTILITY EXPERIMENT")
    print("Training patients:", len(y_fit))
    print("Validation patients:", len(y_val))
    print("Random seeds:", SEEDS)
    print("Noise multipliers:", NOISE_LEVELS)

    for noise_multiplier in NOISE_LEVELS:

        for seed in SEEDS:

            print(
                f"\nNoise: {noise_multiplier}"
                f" | Seed: {seed}"
            )

            weights, bias, epsilon = train_model(
                X_fit,
                y_fit,
                noise_multiplier=noise_multiplier,
                rounds=ROUNDS,
                learning_rate=LEARNING_RATE,
                clipping_norm=CLIPPING_NORM,
                delta=DELTA,
                seed=seed
            )

            metrics = evaluate_model(
                X_val,
                y_val,
                weights,
                bias
            )

            all_results.append({
                "noise_multiplier": noise_multiplier,
                "seed": seed,
                "epsilon_upper_bound": epsilon,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "malignant_recall": metrics["malignant_recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "false_negatives": metrics["false_negatives"],
                "false_positives": metrics["false_positives"]
            })

    # ========================================================
    # STEP 3: SAVE INDIVIDUAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        METRICS_DIR / "privacy_repeated_runs.csv",
        index=False
    )

    # ========================================================
    # STEP 4: CALCULATE MEAN AND STANDARD DEVIATION
    # ========================================================

    summary_df = results_df.groupby(
        "noise_multiplier",
        as_index=False
    ).agg(
        epsilon=("epsilon_upper_bound", "first"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_malignant_recall=(
            "malignant_recall", "mean"
        ),
        std_malignant_recall=(
            "malignant_recall", "std"
        ),
        mean_f1=("f1_score", "mean"),
        std_f1=("f1_score", "std"),
        mean_roc_auc=("roc_auc", "mean"),
        std_roc_auc=("roc_auc", "std"),
        mean_false_negatives=(
            "false_negatives", "mean"
        )
    )

    summary_df.to_csv(
        METRICS_DIR / "privacy_repeated_summary.csv",
        index=False
    )

    # ========================================================
    # STEP 5: DISPLAY SUMMARY
    # ========================================================

    print("\nREPEATED EXPERIMENT SUMMARY")

    print(
        summary_df.round(6).to_string(index=False)
    )

    print("\nSaved files:")
    print("metrics/privacy_repeated_runs.csv")
    print("metrics/privacy_repeated_summary.csv")

    print(
        "\nREPEATED PRIVACY EXPERIMENT COMPLETED!"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_repeated_experiments()
