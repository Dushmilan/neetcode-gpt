import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # Derivative of MSE: -2/N * sum((y - y_hat) * x_j)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.matmul(X, weights)

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        
        weights = initial_weights.copy()
        N = len(Y)

        for _ in range(num_iterations):
            # 1. Get current predictions with current weights
            predictions = self.get_model_prediction(X, weights)
            
            # 2. Calculate derivatives and update each weight
            # We use a temporary array to store updates to ensure we're 
            # using the same 'predictions' for all partial derivatives in this step
            new_weights = np.zeros_like(weights)
            for j in range(len(weights)):
                derivative = self.get_derivative(predictions, Y, N, X, j)
                new_weights[j] = weights[j] - (self.learning_rate * derivative)
            
            weights = new_weights

        return np.round(weights, 5)