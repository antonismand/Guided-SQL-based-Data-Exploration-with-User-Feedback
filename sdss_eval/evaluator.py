import json
import multiprocessing as mp
import re
import time
from bisect import bisect_left

import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib import pyplot as plt
from sdss_eval.sqlparser import schema
from sdss_eval.paths import PROCESSED_DIR
from tqdm import tqdm

from mabrecs.bandits import Egreedy, Popular, ThompsonSampling, Ucb, pUcb


class ReturnAll:
    """Recommends the whole pool; used to score a binning on its own."""

    def __init__(self):
        self.algorithm = "Best Bin"

    def choose_arms(self, pool: list, top_k: int, user: str | int = None, update=True):
        return pool

    def update(self, ids: list[int], rewards: list[int], user: str | int = None):
        return None


def BinarySearch(lst, x):
    i = bisect_left(lst, x)
    return i != len(lst) and lst[i] == x


def evaluate(
    clause: int,
    bandit,
    top_k: int,
    db,
    log_file="log_th50.json",
    pbar=False,
    max_iterations=None,
    skip_lines=None,
):
    total_payoff = 0
    ctr = []

    log_file = PROCESSED_DIR / log_file
    log = open(log_file)
    fixed_pool = False
    pool = []

    total_logs = sum(1 for line in open(log_file))

    if skip_lines:
        if skip_lines > 0 and skip_lines < 1:
            skip_lines = round(skip_lines * total_logs)
        total_logs -= skip_lines

        for _ in range(skip_lines):
            next(log)

    if max_iterations:
        if max_iterations > 0 and max_iterations < 1:
            max_iterations *= total_logs
        total_logs = max_iterations

    if clause == 1:
        pool = np.arange(db.n_tables)
        fixed_pool = True
    # elif clause == 0:
    #     pool = np.arange(db.n_columns)
    #     fixed_pool = True

    if pbar:
        log = tqdm(log, desc=bandit.algorithm, total=total_logs)

    # selected = Counter()
    i = 0
    for line in log:
        try:
            row = json.loads(line)
            user = row[-1]
            target = row[clause]
            n_target = len(target)
        except Exception:
            n_target = 0
            continue
        # user = row[-1][user_feature]

        if n_target > 0:
            i += 1

            if not fixed_pool:
                pool = []
                for table in row[1]:  # create pool from tables in from clause
                    pool.extend(db.schema_idx[table])
                pool = np.array(pool)

            recs = bandit.choose_arms(pool, top_k, user)

            rewards = np.zeros(len(recs))
            hits = 0
            ap = 0

            for idx, rec in enumerate(recs):
                # selected[rec] += 1
                if BinarySearch(target, rec):
                    hits += 1
                    ap += hits / (idx + 1)
                    rewards[idx] = 1

            if hits > 0:
                max_hits = min(n_target, top_k)
                total_payoff += ap / max_hits

            # if i % 10 == 0:
            if bandit.algorithm == "Most popular":  # give full feedback to popular
                bandit.update(target, None, user)
            else:
                bandit.update(recs, rewards, user)

            if i % 1000 == 0:
                ctr.append(total_payoff / i)

            if max_iterations and i > max_iterations:
                break

    return ctr
    # return ctr, selected


def measure_overlap(a, b):
    # check if there is any overlap at all
    if a[1] < b[0] or b[1] < a[0]:
        # print("Overlap of ", a, "with", b, "= 0")
        return 0

    overlap = abs(min(a[1], b[1]) - max(a[0], b[0]))
    full = abs(max(a[1], b[1]) - min(a[0], b[0]))

    # print("Overlap of ", a, "with", b, "=", overlap / full)
    return overlap / full


def evaluate_conditions(
    bandit,
    db,
    top_k=5,
    log_file="where_log_th500.json",
    pbar=False,
    max_lines=None,
    skip_lines=None,
):
    total_payoff = 0
    ctr = []

    log_file = PROCESSED_DIR / log_file
    log = open(log_file)

    total_logs = sum(1 for line in open(log_file))

    if skip_lines:
        if skip_lines > 0 and skip_lines < 1:
            skip_lines = round(skip_lines * total_logs)
        total_logs -= skip_lines

        for _ in range(skip_lines):
            next(log)

    if max_lines:
        if max_lines > 0 and max_lines < 1:
            max_lines *= total_logs
        total_logs = max_lines

    if pbar:
        log = tqdm(log, desc=bandit.algorithm, total=total_logs)

    i = 0
    for line in log:
        row = json.loads(line)
        user = row[-1]

        for column_id, conditions in row[0].items():
            if int(column_id) in db.bins_idx:
                pool = np.array(db.bins_idx[int(column_id)])
                for condition in conditions:
                    i += 1

                    recs = bandit.choose_arms(pool, top_k, user)
                    bins = db.bins[recs]

                    user_selected = (condition[1], condition[2])
                    rewards = np.array(
                        [measure_overlap(x, user_selected) for x in bins]
                    )

                    all = db.bins[pool]
                    all_rewards = np.array(
                        [measure_overlap(x, user_selected) for x in all]
                    )
                    best_reward = np.max(all_rewards)
                    if best_reward != 0:
                        total_payoff += np.max(rewards) / best_reward

                    # total_payoff += np.max(rewards)

                    if (
                        bandit.algorithm == "Most popular"
                    ):  # give full feedback to popular
                        rewards = np.array(
                            [measure_overlap(x, user_selected) for x in db.bins]
                        )
                        bandit.update(np.arange(len(db.bins)), rewards, user)
                    else:
                        bandit.update(recs, rewards, user)

                    if i % 1000 == 0:
                        ctr.append(total_payoff / i)

                if max_lines and i > max_lines:
                    break

    return ctr


class run_tuning:
    def __init__(
        self,
        tests: list,
    ):
        self.results = []
        cpus = mp.cpu_count()
        pool = mp.Pool(cpus)

        for test in tests:
            pool.apply_async(
                self.run_test,
                args=(test,),
                callback=self.update_results,
            )

        pool.close()
        pool.join()

        _, ax = plt.subplots()
        scores = np.array([])
        params = np.array([])
        for test in self.results:
            scores = np.append(scores, test[5])
            params = np.append(params, test[7])

        idxs = np.argsort(params)
        ax.plot(params[idxs], scores[idxs])

        title = (
            self.results[0][4]
            + ", K="
            + str(self.results[0][2])
            + ", Clause="
            + self.results[0][1]
            + ", Dataset="
            + self.results[0][0]
        )
        ax.set(xlabel=self.results[0][6], xticks=params, ylabel="MAP@K", title=title)
        # ax.legend([self.results[0][4]])

        plot_name = (
            f"tune_{self.results[0][4]}_{self.results[0][6]}_{self.results[0][1]}"
        )

        tikzplotlib.clean_figure()
        tikzplotlib.save(
            f"plots/{plot_name}.tex",
            extra_axis_parameters=["scaled x ticks=false", "scaled y ticks=false"],
            axis_height="\\figH",
            axis_width="\\figW",
        )

        best_idx = np.argmax(scores)
        print(
            "Best parameter for:",
            title,
            ", ",
            self.results[0][6],
            "=",
            round(params[best_idx], 1),
        )

        self.results = pd.DataFrame(
            self.results,
            columns=[
                "Dataset",  # 0
                "Clause",  # 1
                "K",  # 2
                "Algorithm",  # 3
                "Comment",  # 4
                "MAP@K",  # 5
                "Parameter Name",  # 6
                "Parameter value",  # 7
                "Execution Time",  # 8
            ],
        ).sort_values(
            by=["Dataset", "Clause", "Parameter value", "MAP@K"], ascending=True
        )

    def run_test(self, test):
        comment = "" if "comment" not in test else test.pop("comment")

        parameter_name = test.pop("parameter_name")
        parameter_value = test.pop("parameter_value")

        alg = test["bandit"].algorithm
        clauses = ["Select", "From", "Where"]
        clause = clauses[test["clause"]]
        k = test["top_k"]

        dataset = "TH" + re.search("th(\d*).json", test["log_file"]).group(1)

        start = time.time()
        score = round(evaluate(**test)[-1], 2)
        end = round((time.time() - start) / 60, 2)

        return [
            dataset,
            clause,
            k,
            alg,
            comment,
            score,
            parameter_name,
            parameter_value,
            end,
        ]

    def update_results(self, result):
        self.results.append(result)


class run_plot_parameter:
    def __init__(self, tests, parameter_name, param_map):
        self.results = []
        cpus = mp.cpu_count()
        pool = mp.Pool(cpus)

        for test in tests:
            pool.apply_async(
                self.run_test,
                args=(test,),
                callback=self.update_results,
            )

        pool.close()
        pool.join()

        fig2, ax = plt.subplots()

        alg_results = {}

        for test in self.results:
            alg = test["algorithm"]
            if alg not in alg_results:
                alg_results[alg] = {"scores": [], "params": [], "color": test["color"]}

            alg_results[alg]["scores"] = np.append(
                alg_results[alg]["scores"], test["score"]
            )

            alg_results[alg]["params"] = np.append(
                alg_results[alg]["params"], test["parameter_value"]
            )

        results = []
        for alg in alg_results:
            idxs = np.argsort(alg_results[alg]["params"])
            ax.plot(
                alg_results[alg]["params"][idxs],
                alg_results[alg]["scores"][idxs],
                label=alg,
                color=alg_results[alg]["color"],
            )
            results.append([alg] + list(alg_results[alg]["scores"][idxs]))

        ax.set(
            xlabel=parameter_name,
            xticks=list(param_map.keys()),
            xticklabels=list(param_map.values()),
            ylabel="MAP@K",
            title=self.results[0]["clause"],
        )
        leg = ax.legend()

        tikzplotlib_fix_ncols(leg)

        plot_name = parameter_name + "_" + self.results[0]["clause"]
        tikzplotlib.save(
            "plots/" + plot_name + ".tex",
            extra_axis_parameters=["scaled x ticks=false", "scaled y ticks=false"],
            axis_height="\\figH",
            axis_width="\\figW",
        )

        self.results = pd.DataFrame(
            results,
            columns=["Algorithm"] + list(param_map.values()),
        )
        # .sort_values(by=["Algorithm"], ascending=False)

    def run_test(self, test):
        parameter_value = test.pop("parameter_value")
        color = test.pop("color")

        clauses = ["Select", "From", "Where", "Predicates"]
        clause = clauses[test["clause"]]
        start = time.time()
        if test["clause"] == 3:
            test.pop("clause")
            score = evaluate_conditions(**test)[-1]
        else:
            score = evaluate(**test)[-1]

        end = (time.time() - start) / 60

        return dict(
            dataset="TH" + re.search("th(.*).json", test["log_file"]).group(1),
            algorithm=test["bandit"].algorithm,
            clause=clause,
            k=test["top_k"],
            score=round(score, 2),
            time=round(end, 2),
            parameter_value=parameter_value,
            color=color,
        )

    def update_results(self, result):
        self.results.append(result)


class run_n_tests:
    def __init__(self, tests: list, plot_name="plot", ylabel="MAP@K"):
        self.results = []
        cpus = mp.cpu_count()
        pool = mp.Pool(cpus)

        for test in tests:
            pool.apply_async(
                self.run_test,
                args=(test,),
                callback=self.update_results,
            )

        pool.close()
        pool.join()

        _, ax = plt.subplots()
        for test in self.results:
            ax.plot(test["scores"], label=test["name"], color=test["color"])

        ax.set(xlabel="t", ylabel=ylabel, title=plot_name)
        leg = ax.legend()

        tikzplotlib_fix_ncols(leg)

        tikzplotlib.clean_figure()
        tikzplotlib.save(
            "plots/" + plot_name + ".tex",
            extra_axis_parameters=["scaled x ticks=false", "scaled y ticks=false"],
            axis_height="\\figH",
            axis_width="\\figW",
        )
        # pl.dump(fig2, open("plots/" + plot_name + ".pickle", "wb"))

        for i, r in enumerate(self.results):
            self.results[i] = [r["name"], round(r["scores"][-1], 2), r["time"]]

        self.results = pd.DataFrame(
            self.results, columns=["Test", "MAP@K", "Execution Time"]
        ).sort_values(by=["MAP@K"], ascending=False)

    def run_test(self, test):
        start = time.time()
        test_name = test["bandit"].algorithm if "name" not in test else test.pop("name")
        color = test.pop("color")

        if test["clause"] == 3:
            test.pop("clause")
            evaluator = evaluate_conditions(**test)
        else:
            evaluator = evaluate(**test)

        return dict(
            scores=evaluator,
            name=test_name,
            time=round((time.time() - start) / 60, 2),
            color=color,
        )

    def update_results(self, result):
        self.results.append(result)


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def create_tests(
    algorithms: list,
    clause: int,
    top_k: int,
    db=schema(),
    skip_lines=0.2,
    th=50,
):
    tests = []
    if clause == 3:  # predicate suggestion
        n = len(db.bins)
    else:
        n = db.n_tables if clause == 1 else db.n_columns

    if "egreedy" in algorithms:
        tests.append(
            dict(
                bandit=Egreedy(epsilon=0.1, init=True, n_arms=n, bias=np.zeros(n)),
                color="y",
            )
        )
    if "popular" in algorithms:
        tests.append(dict(bandit=Popular(np.zeros(n), init=True), color="k"))
    if "popularRegion" in algorithms:  # same baseline, rewarded by region overlap
        tests.append(dict(bandit=Popular(np.zeros(n), init=True), color="k"))
    if "ucb" in algorithms:
        tests.append(
            dict(
                bandit=Ucb(alpha=0.1, init=True, n_arms=n, bias=np.zeros(n)), color="g"
            )
        )

    if "all" in algorithms:
        tests.append(dict(bandit=ReturnAll(), color="m"))

    if "pucb" in algorithms:
        tests.append(
            dict(
                bandit=pUcb(
                    alpha=0.1,
                    personal_ratio=0.5,
                    init=True,
                    n_arms=n,
                    bias=np.zeros(n),
                ),
                color="b",
            )
        )
    if "thompson" in algorithms:
        tests.append(dict(bandit=ThompsonSampling(init=True, n_arms=n), color="c"))

    if clause == 3:
        for test in tests:
            test.update(log_file=f"where_log_th{th}.json")

    else:
        for test in tests:
            test.update(log_file=f"log_th{th}.json", skip_lines=skip_lines)

    for test in tests:  # all
        test.update(
            clause=clause,
            top_k=top_k,
            db=db,
        )

    return tests
