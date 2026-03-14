import numpy as np

class BiasedSGD:
    """
    Biased Matrix Factorization optimized via SGD for explicit feedback.
    """
    def __init__(
        self, 
        n_users: int, 
        n_items: int, 
        global_mean: float,
        factors: int = 50, 
        lr: float = 0.01, 
        reg: float = 0.1, 
        seed: int = 42
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.global_mean = global_mean
        self.factors = factors
        self.lr = lr
        self.reg = reg
        
        rng = np.random.RandomState(seed)
        
        # Biases
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        
        # Latent Factors
        self.P = rng.normal(scale=0.1, size=(n_users, factors))
        self.Q = rng.normal(scale=0.1, size=(n_items, factors))

    def fit_epoch(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        """
        Runs one epoch of SGD. Data should be shuffled externally.
        """
        # We iterate over interactions
        # Numba could accelerate this, but avoiding heavy/complex dependencies per prompt.
        for u, i, r in zip(users, items, ratings):
            pred = self.global_mean + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :])
            err = r - pred
            
            # Update biases
            self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
            self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
            
            # Update factors
            # Copy factor P_u since it gets updated before Q_i update
            P_u = self.P[u, :].copy()
            self.P[u, :] += self.lr * (err * self.Q[i, :] - self.reg * self.P[u, :])
            self.Q[i, :] += self.lr * (err * P_u - self.reg * self.Q[i, :])

    def predict(self, u_indices: np.ndarray, i_indices: np.ndarray) -> np.ndarray:
        """Vectorized prediction for pairs of users and items."""
        pred = self.global_mean + self.b_u[u_indices] + self.b_i[i_indices] + np.sum(self.P[u_indices, :] * self.Q[i_indices, :], axis=1)
        return pred

    def predict_for_user(self, u_index: int, i_indices: np.ndarray) -> np.ndarray:
        """Predicts scores for one user over multiple items."""
        pred = self.global_mean + self.b_u[u_index] + self.b_i[i_indices] + self.Q[i_indices, :] @ self.P[u_index, :]
        return pred

    def calculate_objective(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray) -> float:
        """Calculates regularized squared error."""
        preds = self.predict(users, items)
        loss = np.sum((ratings - preds) ** 2)
        
        reg_loss = self.reg * (
            np.sum(self.b_u ** 2) + 
            np.sum(self.b_i ** 2) + 
            np.sum(self.P ** 2) + 
            np.sum(self.Q ** 2)
        )
        return loss + reg_loss
