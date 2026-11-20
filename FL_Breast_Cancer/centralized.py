
# CENTRALIZED LOGISTIC REGRESSION

import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)



from data import X_train, X_test, y_train, y_test

print("\nDATA IMPORTED SUCCESSFULLY!")

print("Training patients:", len(X_train))
print("Testing patients:", len(X_test))


# STEP 2 - CREATE THE LOGISTIC REGRESSION PIPELINE

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs"
    ))
])

print("\nPipeline created successfully!")


# STEP 3 - STRATIFIED 5-FOLD CROSS-VALIDATION

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="recall"
)

print("\nCross-validation recall scores:", cv_scores)
print("Average cross-validation recall:", cv_scores.mean())


# STEP 4 - HYPERPARAMETER TUNING

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best CV recall:", grid_search.best_score_)


# STEP 5 - FINAL MODEL EVALUATION

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

print("\nFINAL TEST RESULTS")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC-AUC:", auc)

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred))


# STEP 6 - SAVE THE TRAINED MODEL AND RESULTS

BASE_DIR = Path(__file__).resolve().parent

(BASE_DIR / "saved_models").mkdir(exist_ok=True)
(BASE_DIR / "metrics").mkdir(exist_ok=True)

joblib.dump(
    best_model,
    BASE_DIR / "saved_models" / "centralized_logistic_regression.pkl"
)

results = pd.DataFrame([{
    "model": "Centralized Logistic Regression",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": auc,
    "false_positives": int(cm[0, 1]),
    "false_negatives": int(cm[1, 0])
}])

results.to_csv(
    BASE_DIR / "metrics" / "centralized_results.csv",
    index=False
)

print("\nMODEL AND RESULTS SAVED SUCCESSFULLY!")
