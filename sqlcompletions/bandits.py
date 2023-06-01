import numpy as np

# from sklearn.neighbors import NearestNeighbors


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


class pUCBC:
    def __init__(
        self,
        alpha,
        personal_ratio,
        cluster_ratio,
        global_ratio,
        n_arms,
        feedback=0,
        similar_users=5,
    ):

        self.alpha = round(alpha, 1)
        # self.algorithm = (
        #     "Hybrid UCB (α=" + str(self.alpha) + ", p=" + str(personal_ratio) + ")"
        # )
        # assert personal_ratio + cluster_ratio + global_ratio == 1

        self.algorithm = (
            f"pUCBC p={personal_ratio}, c={cluster_ratio}, g={global_ratio}"
        )
        self.pr = personal_ratio
        self.cr = cluster_ratio
        self.gr = global_ratio

        self.feedback = feedback
        self.similar_users = similar_users

        self.q = {}
        self.n = {}
        self.t = {}
        self.global_q = np.zeros(n_arms)
        self.global_n = np.ones(n_arms)
        self.global_t = 0

        self.neigh = None

        self.n_arms = n_arms
        self.users = []

    def choose_arms(self, pool, top_k, t, user):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        q = np.zeros(len(pool))
        n = np.zeros(len(pool))
        t = 0
        if len(self.users) > self.similar_users and user in self.q:
            if t % 100 == 0 or self.neigh is None:
                self.neigh = NearestNeighbors(n_neighbors=self.similar_users)
                self.neigh.fit(list(self.q.values()))

            neighbors = self.neigh.kneighbors([self.q[user]], return_distance=False)[0]
            for neighbor in neighbors:
                usr = self.users[neighbor]
                q += self.q[usr][pool]
                n += self.n[usr][pool]
                t += self.t[usr]

            q /= self.similar_users
            n /= self.similar_users
            t /= self.similar_users
        else:
            q = self.global_q[pool]
            n = self.global_n[pool]
            t = self.global_t

        if user not in self.q:
            self.q[user] = np.zeros(self.n_arms)
            self.n[user] = np.ones(self.n_arms)
            self.t[user] = 0
            self.users.append(user)

        mixed_q = (
            self.pr * self.q[user][pool] + self.cr * q + self.gr * self.global_q[pool]
        )

        mixed_b = (
            self.pr
            * np.sqrt(self.alpha * np.log(self.t[user] + 1) / self.n[user][pool])
            + self.cr * np.sqrt(self.alpha * np.log(t + 1) / n)
            + self.gr
            * np.sqrt(self.alpha * np.log(self.global_t + 1) / self.global_n[pool])
        )

        ucbs = mixed_q + mixed_b

        top_idxs = (-ucbs).argsort()[:top_k]
        return pool[top_idxs]

    def update(self, recs, rewards, user, true=None):
        self.n[user][recs] += 1
        self.q[user][recs] += (rewards - self.q[user][recs]) / self.n[user][recs]

        self.global_n[recs] += 1
        self.global_q[recs] += (rewards - self.global_q[recs]) / self.global_n[recs]
        self.global_t += 1

        if self.feedback == 1:
            rest = list(set(true) - set(recs))
            rewards = np.array([1] * len(rest))

            self.q[user][rest] += (rewards - self.q[user][rest]) / self.n[user][rest]
            self.global_q[rest] += (rewards - self.global_q[rest]) / self.global_n[rest]


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


class PopularPerUser(MAB):
    def __init__(self, n_arms: int):
        MAB.__init__(self, n_arms, feedback=1)
        self.algorithm = "Per User Popular"
        self.q = {}

    def choose_arms(self, pool: list, top_k: int, t=None, user: int = None):
        self.user = user
        if user not in self.q:
            self.q[user] = np.zeros(self.n_arms)

        top_idx = (-self.q[user][pool]).argsort()[:top_k]
        self.recs = pool[top_idx]
        return self.recs

    def update(self, rewards=None, true: list = []):
        # if self.feedback == 0:
        #     self.q[self.user][self.recs] += rewards
        # else:
        self.q[self.user][true] += 1


class Hedge(MAB):
    def __init__(self, n_arms: int, learning_rate: float):
        MAB.__init__(self, n_arms, feedback=1)
        self.w = np.ones(n_arms)
        self.h = learning_rate
        self.algorithm = "Hedge"
        self.all = set(np.arange(n_arms))

    def choose_arms(self, pool: list, top_k: int, t=None, user=None):
        p = self.w[pool] / np.sum(self.w[pool])
        return pool[np.random.choice(len(pool), size=top_k, p=p)]

    def update(self, rewards=None, true: list = []):
        not_selected = list(self.all - set(true))
        self.w[not_selected] *= 1 - self.h


class Exp3(MAB):
    def __init__(self, n_arms: int, gamma: float, feedback: int = 0):

        MAB.__init__(self, n_arms, feedback)
        self.algorithm = "Exp3"
        self.gamma = gamma

        self.weights = np.ones(n_arms)

    def choose_arms(self, pool: list, top_k: int, t=None, user=None):
        sum_weights_pool = np.sum(self.weights[pool])
        sum_weights = np.sum(self.weights)

        self.p = (1 - self.gamma) * self.weights / sum_weights + (
            self.gamma / self.n_arms
        )

        p = (1 - self.gamma) * self.weights[pool] / sum_weights_pool + (
            self.gamma / len(pool)
        )
        self.recs = pool[np.random.choice(len(pool), size=top_k, p=p)]
        return self.recs

    def update(self, rewards: list = None, true: list = []):
        recs = self.recs
        est = rewards / self.p[recs]
        self.weights[recs] *= np.exp(est * self.gamma / self.n_arms)

        if self.feedback == 1:
            rest = list(set(true) - set(recs))
            rewards = np.array([1] * len(rest))

            est = rewards / self.p[rest]
            self.weights[rest] *= np.exp(est * self.gamma / self.n_arms)
