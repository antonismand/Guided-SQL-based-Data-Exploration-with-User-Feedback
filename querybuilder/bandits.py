import time
from typing import Dict

import numpy as np


class Ucb:
    """
    UCB algorithm implementation
    """

    def __init__(
        self, alpha: float, init: bool, n_arms: int = None, bias: np.ndarray = None
    ):
        """
        Parameters
        ----------
        alpha : exploration parameter
        init : whether to initialize the algorithm's weights
        n_arms : number of arms
        bias : the initial payoffs
        """

        self.alpha = round(alpha, 1)
        self.algorithm = "UCB"
        if init:
            self.n_arms = n_arms
            self.global_payoff = bias
            self.global_n = np.ones(n_arms)
            self.global_t = 1

    def toDict(self):
        return {
            "global_payoff": self.global_payoff.tolist(),
            "global_n": self.global_n.tolist(),
            "global_t": self.global_t,
            "n_arms": self.n_arms,
        }

    def fromDict(self, dict: dict):
        self.global_payoff = np.asarray(dict["global_payoff"])
        self.global_n = np.asarray(dict["global_n"])
        self.global_t = dict["global_t"]
        self.n_arms = dict["n_arms"]

    def _calculate_UB(self, payoff: np.ndarray, n: np.ndarray, t: int) -> np.ndarray:
        q = payoff / n
        ucbs = q + np.sqrt(self.alpha * np.log(t) / n)
        return ucbs

    def choose_arms(
        self, pool: np.ndarray, top_k: int, user: str | int = None, update=True
    ):
        """
        Returns top-K recommendations from a pool
        Parameters
        ----------
        pool : indexes of available items
        top_k : how many items to recommend
        update : whether to update the algorithm's weights
        """

        ucbs = self._calculate_UB(
            self.global_payoff[pool], self.global_n[pool], self.global_t
        )
        if top_k == -1:
            top_idxs = (-ucbs).argsort()
        else:
            top_idxs = (-ucbs).argsort()[:top_k]

        recs = pool[top_idxs]

        if update:
            self.global_n[recs] += 1
            self.global_t += 1

        return recs

    def update(self, ids: list[int], rewards: list[int], user: str | int = None):
        """
        Updates algorithm's parameters
        Parameters
        ----------
        ids : indexes to update
        rewards : the reward for each index
        """

        self.global_payoff[ids] += rewards


class pUcb(Ucb):
    """
    Personalized UCB algorithm
    """

    def __init__(
        self,
        alpha: float,
        personal_ratio: float,
        init: bool,
        n_arms: int = None,
        bias: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        alpha : exploration parameter
        init : whether to initialize the algorithm's weights
        n_arms : number of arms
        bias : the initial payoffs
        """
        Ucb.__init__(self, alpha=alpha, init=init, n_arms=n_arms, bias=bias)
        self.algorithm = "pUCB"
        self.p = round(personal_ratio, 1)

        self.users_payoff: Dict[str, np.ndarray] = {}
        self.users_n: Dict[str, np.ndarray] = {}
        self.users_t: Dict[str, int] = {}

    def fromDict(
        self,
        global_vars: dict,
        user_vars: dict | None = None,
        userId: str | None = None,
    ):
        super().fromDict(global_vars)

        if userId and user_vars:
            self.users_payoff[userId] = np.asarray(user_vars["users_payoff"])
            self.users_n[userId] = np.asarray(user_vars["users_n"])
            self.users_t[userId] = user_vars["users_t"]

    def toDict(self, userId: str):
        global_vars = super().toDict()
        user_vars = {
            "users_payoff": self.users_payoff[userId].tolist(),
            "users_n": self.users_n[userId].tolist(),
            "users_t": self.users_t[userId],
            "last_update": time.time(),
        }

        return global_vars, user_vars

    def choose_arms(
        self, pool: np.ndarray, top_k: int, user: str | int | None, update=True
    ):
        """
        Returns top-K recommendations from a pool
        Parameters
        ----------
        pool : indexes of available items
        top_k : how many items to recommend
        user : the user identifier
        update : whether to update the algorithm's weights
        """

        if not user:
            return super().choose_arms(pool, top_k, user, update)

        if user not in self.users_payoff:
            self.users_payoff[user] = np.zeros(self.n_arms)
            self.users_n[user] = np.ones(self.n_arms)
            self.users_t[user] = 1

        global_ucbs = super()._calculate_UB(
            self.global_payoff[pool], self.global_n[pool], self.global_t
        )

        user_ucbs = super()._calculate_UB(
            self.users_payoff[user][pool], self.users_n[user][pool], self.users_t[user]
        )

        ucbs = self.p * user_ucbs + (1 - self.p) * global_ucbs
        if top_k == -1:
            top_idxs = (-ucbs).argsort()
        else:
            top_idxs = (-ucbs).argsort()[:top_k]
        recs = pool[top_idxs]

        if update:
            self.global_n[recs] += 1
            self.global_t += 1

            self.users_n[user][recs] += 1
            self.users_t[user] += 1

        return recs

    def update(self, ids: list[int], rewards: list[int], user: str | int):
        """
        Updates algorithm's parameters
        Parameters
        ----------
        ids : indexes to update
        rewards : the reward for each index
        user : the user identifier
        """

        self.global_payoff[ids] += rewards
        self.users_payoff[user][ids] += rewards


class LinUCB:
    """
    LinUCB algorithm implementation
    """

    def __init__(
        self, alpha: float, n_arms: int = None, n_features: int = None, init=False
    ):
        """
        Parameters
        ----------
        alpha : LinUCB exploration parameter
        n_arms :  number of arms
        n_features : number of user features
        init : whether to initialize the algorithm's weights
        -----------
        Get significant speed up on Mac M1 by installing numpy with BLAS interface specified as vecLib
        See more at https://stackoverflow.com/questions/70240506/why-python-native-on-m1-max-is-greatly-slower-than-python-on-old-intel-i5
        """

        self.n_features = n_features
        self.algorithm = "LinUCB"
        self.alpha = round(alpha, 1)

        if init:
            self.A = np.array([np.identity(n_features)] * n_arms)
            self.A_inv = np.array([np.identity(n_features)] * n_arms)
            self.b = np.zeros((n_arms, n_features, 1))

    def choose_arms(self, pool: np.ndarray, top_k: int, user: list, update=True):
        """
        Returns top-K recommendations from a pool
        Parameters
        ----------
        pool : indexes of available items
        top_k : how many items to recommend
        user : user features
        """

        A_inv = self.A_inv[pool]  # (23, 12, 6)
        b = self.b[pool]  # (23, 12, 1)

        n_pool = len(pool)
        x = np.array([user] * n_pool)  # (23, 6)

        x = x.reshape((n_pool, self.n_features, 1))  # (23, 12, 1)

        theta = A_inv @ b  # (23, 12, 1)

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A_inv @ x
        )
        p = p.reshape((n_pool))
        top_idxs = (-p).argsort()[:top_k]
        return pool[top_idxs]

    def update(self, ids: list[int], rewards: list[int], user: list = None):
        """
        Updates algorithm's parameters
        Parameters
        ----------
        ids : indexes to update
        rewards : the reward for each index
        user : user features
        """

        user = np.array(user)
        x = user.reshape((self.n_features, 1))

        for i, r in enumerate(ids):
            self.A[r] += x @ x.T
            self.b[r] += rewards[i] * x
            self.A_inv[r] = np.linalg.inv(self.A[r])


class Egreedy:
    """
    Epsilon greedy algorithm implementation
    """

    def __init__(
        self, epsilon: float, init: bool, n_arms: int = None, bias: np.ndarray = None
    ):
        """
        Parameters
        ----------
        epsilon : Egreedy parameter
        init : whether to initialize the algorithm's weights
        n_arms : number of arms
        bias : the initial payoffs
        """

        self.e = round(epsilon, 1)
        self.algorithm = "Egreedy"

        if init:
            self.global_payoff = bias
            self.global_n = np.ones(n_arms)

    def choose_arms(
        self, pool: np.ndarray, top_k: int, user: str | int = None, update=True
    ):
        """
        Returns top-K recommendations from a pool
        Parameters
        ----------
        pool : indexes of available items
        top_k : how many items to recommend
        """

        if np.random.rand() > self.e:
            top_idx = (-self.global_payoff[pool] / self.global_n[pool]).argsort()[
                :top_k
            ]
            recs = pool[top_idx]
        else:
            recs = np.random.choice(pool, top_k)

        if update:
            self.global_n[recs] += 1

        return recs

    def update(self, ids: list[int], rewards: list[int], user: str | int = None):
        """
        Updates algorithm's parameters
        Parameters
        ----------
        ids : indexes to update
        rewards : the reward for each index
        """

        self.global_payoff[ids] += rewards


class ThompsonSampling:
    """
    Thompson sampling algorithm implementation
    """

    def __init__(self, init: bool, n_arms: int = None):
        """
        Parameters
        ----------
        init : whether to initialize the algorithm's weights
        n_arms : number of arms

        """
        self.n_arms = n_arms
        self.algorithm = "Thompson Sampling"

        if init:
            self.alpha = np.ones(n_arms)
            self.beta = np.ones(n_arms)

    def choose_arms(
        self, pool: np.ndarray, top_k: int, user: str | int = None, update=True
    ):
        """
        Returns top-K recommendations from pool
        Parameters
        ----------
        pool : indexes of available items
        top_k : how many items to recommend
        """

        theta = np.random.beta(self.alpha[pool], self.beta[pool])
        top_idxs = (-theta).argsort()[:top_k]
        return pool[top_idxs]

    def update(self, ids: list[int], rewards: list[int], user: str | int = None):
        """
        Updates algorithm's parameters(matrices) : a,b
        Parameters
        ----------
        ids : indexes to update
        rewards : the reward for each index
        """

        self.alpha[ids] += rewards
        self.beta[ids] = self.beta[ids] + 1 - rewards


class Popular:
    """
    Most-Popular algorithm
    """

    def __init__(self, bias: np.ndarray, init=False):
        """
        Parameters
        ----------
        bias : the initial payoffs
        init : whether to initialize the algorithm's weights
        """

        self.algorithm = "Most popular"

        if init:
            self.global_payoff = bias

    def choose_arms(
        self, pool: np.ndarray, top_k: int, user: str | int = None, update=True
    ):
        """
        Returns top-K recommendations from a pool
        Parameters
        ----------
        pool : indexes of available items
        top_k : how many items to recommend
        update : whether to update the algorithm's weights
        """

        top_idx = (-self.global_payoff[pool]).argsort()[:top_k]
        return pool[top_idx]

    def update(self, ids: list[int], rewards: list[int] = None, user: str | int = None):
        """
        Updates algorithm's parameters
        Parameters
        ----------
        ids : indexes to update
        """

        self.global_payoff[ids] += 1
