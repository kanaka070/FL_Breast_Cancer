


# LOGISTIC REGRESSION MODEL FOR FEDERATED LEARNING


import numpy as np


class LogisticRegressionModel:

    def __init__(self, n_features=30, learning_rate=0.05):

        # Start with 30 weights, all equal to zero
        self.weights = np.zeros(n_features, dtype=float)

        # Start with bias equal to zero
        self.bias = 0.0

        # Controls how large each training update is
        self.learning_rate = learning_rate


    # STEP 1: Calculate the sigmoid function

    def sigmoid(self, z):

        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))


    # STEP 2: Predict the probability of malignancy

    def predict_proba(self, X):

        X = np.asarray(X, dtype=float)

        z = np.dot(X, self.weights) + self.bias

        return self.sigmoid(z)


    # STEP 3: Predict benign (0) or malignant (1)

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)


    # STEP 4: Calculate binary cross-entropy loss

    def calculate_loss(self, X, y):

        y = np.asarray(y, dtype=float)

        probabilities = self.predict_proba(X)

        probabilities = np.clip(
            probabilities, 1e-15, 1 - 1e-15
        )

        loss = -np.mean(
            y * np.log(probabilities)
            + (1 - y) * np.log(1 - probabilities)
        )

        return float(loss)


    # STEP 5: Train the model using gradient descent

    def train(self, X, y, epochs=1):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        for epoch in range(epochs):

            # Calculate current predictions
            probabilities = self.predict_proba(X)

            # Difference between prediction and actual label
            errors = probabilities - y

            # Calculate gradients
            weight_gradient = np.dot(X.T, errors) / len(y)

            bias_gradient = np.mean(errors)

            # Update weights and bias
            self.weights -= self.learning_rate * weight_gradient

            self.bias -= self.learning_rate * bias_gradient

        return self.calculate_loss(X, y)


    # STEP 6: Get model parameters for FedAvg

    def get_parameters(self):

        return {
            "weights": self.weights.copy(),
            "bias": float(self.bias)
        }


    # STEP 7: Receive updated parameters from FedAvg server

    def set_parameters(self, parameters):

        self.weights = np.asarray(
            parameters["weights"], dtype=float
        ).copy()

        self.bias = float(parameters["bias"])


# ============================================================
# TEST THE MODEL
# ============================================================

if __name__ == "__main__":

    print("\n LOGISTIC REGRESSION MODEL TEST")

    model = LogisticRegressionModel(
        n_features=30,
        learning_rate=0.05
    )

    print("\nNumber of weights:", len(model.weights))
    print("Initial weights:", model.weights)
    print("Initial bias:", model.bias)

    # Small artificial dataset for checking calculations
    X_sample = np.array([
        [0.2] * 30,
        [0.8] * 30,
        [-0.5] * 30,
        [1.0] * 30
    ])

    y_sample = np.array([0, 1, 0, 1])

    print("\nInitial loss:",
          model.calculate_loss(X_sample, y_sample))

    model.train(X_sample, y_sample, epochs=100)

    print("Loss after training:",
          model.calculate_loss(X_sample, y_sample))

    print("\nPredicted probabilities:")
    print(model.predict_proba(X_sample))

    print("\nPredicted classes:")
    print(model.predict(X_sample))

    parameters = model.get_parameters()

    print("\nParameters ready for FedAvg:")
    print("Weights shape:", parameters["weights"].shape)
    print("Bias:", parameters["bias"])

  
