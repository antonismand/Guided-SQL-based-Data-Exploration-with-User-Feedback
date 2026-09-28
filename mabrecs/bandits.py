from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from mabrecs.storage import Storage


class Bandit(ABC):  # storage compatible classes
    @abstractmethod
    def toDict(self):
        pass

    @abstractmethod
    def fromDict(self, dct: dict):
        pass

    @abstractmethod
    def choose_arms(
        self, pool: np.ndarray, top_k: int, user: str | int = None, update=True
    ):
        pass

    @abstractmethod
    def recommend(
        self, pool: np.ndarray, top_k: int, user: str | int = None, update=True
    ):
        pass

    @abstractmethod
    def update(self, ids: list[int], rewards: list[int], user: str | int = None):
        pass


class Ucb(Bandit):
    """
    UCB algorithm implementation
    """

    def __init__(
        self,
        alpha: float,
        init: bool,
        n_arms: int = None,
        bias: np.ndarray = None,
        storage: Storage = None,
        arms: list = None,
    ):
        """
        Parameters
        ----------
        alpha : exploration parameter
        init : whether to initialize the algorithm's weights or read from storage
        n_arms : number of arms
        bias : the initial payoffs
        """

        self.alpha = round(alpha, 1)
        self.algorithm = "UCB"
        self.storage = storage

        if init:
            self.n_arms = n_arms
            self.payoff = bias
            self.n = np.ones(n_arms)
            self.t = 1

            if storage is not None:
                storage.set("arms", arms)
                self.save()
        else:
            self.load()

    def load(self):
        self.fromDict(self.storage.get("ucb"))

    def save(self):
        if self.storage is not None:
            self.storage.set("ucb", self.toDict())

    def toDict(self):
        return {
            "payoff": self.payoff.tolist(),
            "n": self.n.tolist(),
            "t": self.t,
            "n_arms": self.n_arms,
        }

    def fromDict(self, dct: dict):
        self.payoff = np.asarray(dct["payoff"])
        self.n = np.asarray(dct["n"])
        self.t = dct["t"]
        self.n_arms = dct["n_arms"]

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

        ucbs = self._calculate_UB(self.payoff[pool], self.n[pool], self.t)
        if top_k == -1:
            top_idxs = (-ucbs).argsort()
        else:
            top_idxs = (-ucbs).argsort()[:top_k]

        recs = pool[top_idxs]

        if update:
            self.n[recs] += 1
            self.t += 1

        return recs

    def recommend(
        self, pool: np.ndarray, top_k: int, user: str | int = None, update=True
    ):
        rec_indexes = self.choose_arms(pool, top_k, user, update)

        if update:
            self.save()

        return [self.storage.get_array_index("arms", rec) for rec in rec_indexes]

    def update(self, ids: list[int], rewards: list[int], user: str | int = None):
        """
        Updates algorithm's parameters
        Parameters
        ----------
        ids : indexes to update
        rewards : the reward for each index
        """

        self.payoff[ids] += rewards
        self.save()


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
        storage: Storage = None,
        arms: list = None,
    ):
        """
        Parameters
        ----------
        alpha : exploration parameter
        init : whether to initialize the algorithm's weights
        n_arms : number of arms
        bias : the initial payoffs
        """
        super().__init__(
            alpha=alpha,
            init=init,
            n_arms=n_arms,
            bias=bias,
            storage=storage,
            arms=arms,
        )
        self.algorithm = "pUCB"
        self.p = round(personal_ratio, 1)

        self.user = None

        if init and self.storage is not None:
            self.storage.set("users", {})

        self.users_payoff: Dict[str, np.ndarray] = {}
        self.users_n: Dict[str, np.ndarray] = {}
        self.users_t: Dict[str, int] = {}

    def load_user(self):
        if self.storage is not None and self.user not in self.users_payoff:
            user = self.storage.get_nested_key("users", self.user)
            if user is not None:
                self.users_payoff[self.user] = np.array(user["payoff"])
                self.users_n[self.user] = np.array(user["n"])
                self.users_t[self.user] = user["t"]

    def save_user(self):
        if self.storage is not None:
            self.storage.set(
                "users",
                {
                    self.user: {
                        "payoff": self.users_payoff[self.user].tolist(),
                        "n": self.users_n[self.user].tolist(),
                        "t": self.users_t[self.user],
                    }
                },
            )

    def toDict(self):
        return super().toDict()

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

        self.user = user
        if user is not None:
            self.load_user()

        if user not in self.users_payoff:
            self.users_payoff[user] = np.zeros(self.n_arms)
            self.users_n[user] = np.ones(self.n_arms)
            self.users_t[user] = 1

        global_ucbs = super()._calculate_UB(self.payoff[pool], self.n[pool], self.t)

        user_ucbs = super()._calculate_UB(
            self.users_payoff[user][pool],
            self.users_n[user][pool],
            self.users_t[user],
        )

        ucbs = self.p * user_ucbs + (1 - self.p) * global_ucbs
        if top_k == -1:
            top_idxs = (-ucbs).argsort()
        else:
            top_idxs = (-ucbs).argsort()[:top_k]
        recs = pool[top_idxs]

        if update:
            self.n[recs] += 1
            self.t += 1

            self.users_n[user][recs] += 1
            self.users_t[user] += 1

        return recs

    def recommend(
        self, pool: np.ndarray, top_k: int, user: str | int = None, update=True
    ):
        if user is None:
            return super().recommend(pool, top_k, user, update)

        rec_indexes = self.choose_arms(pool, top_k, user, update)

        if update:
            super().save()
            self.save_user()

        return [self.storage.get_array_index("arms", rec) for rec in rec_indexes]

    def update(self, ids: list[int], rewards: list[int], user: str | int):
        """
        Updates algorithm's parameters
        Parameters
        ----------
        ids : indexes to update
        rewards : the reward for each index
        """
        self.payoff[ids] += rewards
        super().save()

        if user is not None:
            self.user = user
            self.load_user()
            self.users_payoff[user][ids] += rewards
            self.save_user()


####################### NOT COMPATIBLE WITH STORAGE ############################

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
        rewards : reward of each id (e.g. region overlap); 1 for each id if None
        """

        self.global_payoff[ids] += 1 if rewards is None else rewards
