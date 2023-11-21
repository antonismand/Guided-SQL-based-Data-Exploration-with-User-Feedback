import numpy as np

class MAB:
    def __init__(self, n_arms: int, feedback: int = 0):
        self.feedback = feedback
        self.n_arms = n_arms

    def choose_arms(self, pool, top_k, t, user):
        return None

    def update(self, rewards, true):
        return None


class ReturnAll:
    def __init__(self):
        self.algorithm = "Best Bin"

    def choose_arms(self, pool: list, top_k: int, user: str | int = None, update=True):
        return pool

    def update(self, ids: list[int], rewards: list[int], user: str | int = None):
        return None




class PopularRegion:
    def __init__(self, n_arms: int):
        self.algorithm = "Most popular"
        self.q = np.zeros(n_arms)
        self.feedback = 1

    def choose_arms(self, pool: list, top_k: int, user: str | int = None, update=True):
        top_idx = (-self.q[pool]).argsort()[:top_k]
        return pool[top_idx]

    def update(self, ids: list[int], rewards: list[int], user: str | int = None):
        self.q += rewards

