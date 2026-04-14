import numpy as np

class ALS:
    """
    Alternating Least Squares for explicit feedback matrix factorization.
    """
    def __init__(self, n_users: int, n_items: int, factors: int = 50, reg: float = 0.1, seed: int = 42):
        self.n_users = n_users
        self.n_items = n_items
        self.factors = factors
        self.reg = reg
        
        # Initialize latent factors
        rng = np.random.RandomState(seed)
        self.P = rng.normal(scale=0.1, size=(n_users, factors))
        self.Q = rng.normal(scale=0.1, size=(n_items, factors))

    def fit_step(self, user_items_dict: dict, item_users_dict: dict):
        """
        Performs one full ALS update (update all P, then update all Q).
        user_items_dict: dict of user_idx -> list of (item_idx, rating)
        item_users_dict: dict of item_idx -> list of (user_idx, rating)
        """
        # Update user factors P
        reg_mat = self.reg * np.eye(self.factors)
        for u in range(self.n_users):
            if u not in user_items_dict: continue
            items_ratings = user_items_dict[u]
            if not items_ratings: continue
            
            i_indices = [idx for idx, r in items_ratings]
            r_vals = np.array([r for idx, r in items_ratings])
            
            Q_u = self.Q[i_indices, :]  # shape: (n_items_u, factors)
            A = Q_u.T @ Q_u + reg_mat
            V = Q_u.T @ r_vals
            
            self.P[u, :] = np.linalg.solve(A, V)

        # update item factors Q
        for i in range(self.n_items):
            if i not in item_users_dict: continue
            users_ratings = item_users_dict[i]
            if not users_ratings: continue
            
            u_indices = [idx for idx, r in users_ratings]
            r_vals = np.array([r for idx, r in users_ratings])
            
            P_i = self.P[u_indices, :]
            A = P_i.T @ P_i + reg_mat
            V = P_i.T @ r_vals
            
            self.Q[i, :] = np.linalg.solve(A, V)

    def predict(self, u_indices: np.ndarray, i_indices: np.ndarray) -> np.ndarray:
        """Vectorized prediction for pairs of users and items."""
        return np.sum(self.P[u_indices, :] * self.Q[i_indices, :], axis=1)

    def predict_for_user(self, u_index: int, i_indices: np.ndarray) -> np.ndarray:
        """Predicts scores for one user over multiple items."""
        return self.Q[i_indices, :] @ self.P[u_index, :]

    def calculate_objective(self, user_items_dict: dict) -> float:
        """Calculates regularized squared error over training data."""
        loss = 0.0
        for u, items_ratings in user_items_dict.items():
            if not items_ratings: continue
            i_indices = [idx for idx, r in items_ratings]
            r_vals = np.array([r for idx, r in items_ratings])
            
            preds = self.predict_for_user(u, i_indices)
            loss += np.sum((r_vals - preds) ** 2)
            
        # regularization loss
        loss += self.reg * (np.sum(self.P ** 2) + np.sum(self.Q ** 2))
        return loss
