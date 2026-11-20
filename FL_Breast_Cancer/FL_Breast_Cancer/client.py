
# ============================================================
# LOCAL HOSPITAL TRAINING
# ============================================================

import numpy as np
from sklearn.preprocessing import StandardScaler

from model import LogisticRegressionModel
from data import hospitals, X_train


class HospitalClient:

    def __init__(self, name, X, y, learning_rate=0.05):

        self.name = name

        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)

        self.num_patients = len(self.y)

        self.model = LogisticRegressionModel(
            n_features=self.X.shape[1],
            learning_rate=learning_rate
        )

    # Receive global parameters and train locally
    def train_local(self, global_parameters, epochs=5):

        self.model.set_parameters(global_parameters)

        loss = self.model.train(
            self.X,
            self.y,
            epochs=epochs
        )

        return {
            "hospital": self.name,
            "parameters": self.model.get_parameters(),
            "num_patients": self.num_patients,
            "loss": loss
        }


# ============================================================
# TEST LOCAL TRAINING
# ============================================================

if __name__ == "__main__":

    print("\nLOCAL HOSPITAL TRAINING TEST")

    # Use one common feature scale for the simulation
    scaler = StandardScaler()
    scaler.fit(X_train)

    clients = []

    for name, hospital in hospitals.items():

        X_scaled = scaler.transform(hospital["X"])

        client = HospitalClient(
            name=name,
            X=X_scaled,
            y=hospital["y"],
            learning_rate=0.05
        )

        clients.append(client)

    # Create initial global model
    global_model = LogisticRegressionModel(
        n_features=30,
        learning_rate=0.05
    )

    global_parameters = global_model.get_parameters()

    total_patients = 0

    for client in clients:

        result = client.train_local(
            global_parameters,
            epochs=5
        )

        print("\nHospital:", result["hospital"])
        print("Patients:", result["num_patients"])
        print("Local training loss:", round(result["loss"], 6))
        print(
            "Updated weights shape:",
            result["parameters"]["weights"].shape
        )
        print(
            "Updated bias:",
            round(result["parameters"]["bias"], 6)
        )

        total_patients += result["num_patients"]

    assert len(clients) == 3
    assert total_patients == 455

    print("\nTotal training patients:", total_patients)
    print("All three hospitals trained successfully!")
    print("\nLOCAL HOSPITAL TRAINING COMPLETED SUCCESSFULLY!")
