# Breast Cancer Federated Learning — Project Workflow

## Objective
Build a research application for breast cancer classification using the WDBC dataset and three simulated hospitals.

## Approaches
1. Centralized Logistic Regression baseline.
2. Federated Averaging (FedAvg).
3. Differentially Private FedAvg (DP-FedAvg).

These are three separate approaches to train and evaluate, not one combined model.

## Initial training
- Split data correctly into training and evaluation sets.
- Simulate three hospitals using separate portions of the training data.
- Train and evaluate all three approaches.
- Compare accuracy, precision, recall, specificity, F1-score, ROC-AUC, training time and confusion matrices.
- Verify DP implementation and privacy accounting before reporting privacy guarantees.
- Select a model for deployment based on evaluation results and admin approval.

## New patient prediction
- Doctor enters the patient's 30 numerical features.
- Validate the input and apply the matching saved preprocessing.
- Run the currently deployed model locally at the hospital.
- Display the prediction and model probability.
- Do not retrain for every new patient.

## Periodic retraining
- Provide an admin-configurable threshold for new confirmed labelled records.
- Use 100 as an initial configurable example, not a validated requirement.
- Predictions alone are not confirmed labels.
- Trigger training and evaluation when the threshold is reached.
- Compare the candidate model against the current deployed model.
- Replace the current model only after evaluation and admin approval.

## Application
- Three simulated hospital views.
- Patient input and local prediction.
- Central coordinator for federated model updates.
- Dashboard comparing the three approaches.
- Training history and metrics.
- Admin controls for retraining and model deployment.

## Privacy and correctness
- Keep hospital training records separate during federated training.
- Do not send raw patient records to the federated coordinator.
- Prevent train/test leakage.
- Do not invent metrics or claim unverified privacy guarantees.

## Development
Audit the existing code first. Reuse working components, fix issues incrementally, and run tests after significant changes. Do not rebuild from scratch without a technical reason.
