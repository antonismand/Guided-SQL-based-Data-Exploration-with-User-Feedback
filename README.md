<p align="center">
<a href="https://doi.org/10.1109/ICDE60146.2024.00372"><img alt="Paper: ICDE 2024" src="https://img.shields.io/badge/paper-ICDE%202024-informational"></a>
<a href="#"><img alt="Python version" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?logo=python"></a>
<a href="#"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# Guided SQL-Based Data Exploration with User Feedback

Code, experiments and user study for the ICDE 2024 paper
[**Guided SQL-Based Data Exploration with User Feedback**](https://doi.org/10.1109/ICDE60146.2024.00372)
by Antonis Mandamadiotis, Georgia Koutrika and Sihem Amer-Yahia.

## Abstract

The exploration of large, real-world databases poses major challenges to users due to their volume and complexity. SQL is the preferred language for data exploration. However, the process of iteratively refining SQL queries is tedious and time consuming. We formulate the automation of personalized SQL-based data exploration as the problem of suggesting the most relevant query and accounting for user feedback at each step. We develop an end-to-end solution and a system to assist users in exploring different components of a complex database. We instantiate our solution using Multi-Armed Bandits, a category of algorithms that are suitable for interactive online learning by balancing exploration with exploitation. We design a lightweight algorithm to personalize stepwise SQL recommendations that efficiently discovers the current user preferences in coordination with that user's feedback and what other users prefer. We run extensive experiments that demonstrate the utility of our approach for large-scale data exploration.

## Overview

Instead of recommending full SQL queries, the user builds a query step by step, top-down, and a recommender helps at every step:

```
FROM (tables) ──► SELECT (columns) ──► WHERE (columns) ──► WHERE (predicates)
```

1. **Table recommender** suggests tables for the `FROM` clause.
2. **Column recommenders** suggest columns of the selected tables for the `SELECT` and `WHERE` clauses.
3. **Predicate recommenders** suggest filters on a chosen column:
   - _categorical_ columns get `column = value` predicates;
   - _numerical_ columns get `column BETWEEN x AND y` regions, built with equal-width, equal-height or K-means binning.

Breaking the problem up this way means each recommender picks from a small, finite pool of items (for example, only the columns of the tables already selected), rather than from every possible query.

Each recommender is a **multi-armed bandit** that returns the top-K items and learns online from which ones the user picks (any number, in any position). Feedback is implicit: the system reads it from the queries the user builds, so the user never rates anything. A selected item gets a reward of 1. For numerical regions, the reward is how much the recommended bin overlaps the region the user chose.

### pUCB: personalized UCB

pUCB is a lightweight extension of UCB. It needs no user or item features, so it works from a cold start. For each arm it keeps two upper confidence bounds, one over all users and one for the current user, and ranks arms by their weighted sum:

```math
\begin{aligned}
UB_a    &= q_a     + \sqrt{\frac{\alpha \ln t}{n_a}}             && \text{(all users)} \\
UB_a(u) &= q_{a,u} + \sqrt{\frac{\alpha \ln t_u}{n_{a,u}}}       && \text{(current user } u\text{)} \\[4pt]
a_t     &= \underset{a \in P_t}{\arg\max} \; p \cdot UB_a(u) + (1 - p) \cdot UB_a
\end{aligned}
```

Here $q_a$ is the average reward of item $a$, $n_a$ is how many times it has been recommended, $t$ is the number of trials, $\alpha$ controls exploration and $P_t$ is the pool of items available at step $t$. The personalization factor $p$ controls the balance: $p = 0$ is plain UCB and $p = 1$ uses only the current user's history. Choosing the top-K items costs $O(N \log N)$ for a pool of $N$ items.

## Key results

**Offline evaluation** replays real query logs from the [Sloan Digital Sky Survey](https://classic.sdss.org/) (SDSS DR7: 115 tables, 52 views, 9,903 columns). We compare pUCB with UCB, Thompson Sampling, ε-greedy and a _Popular_ baseline.

- **Tables and columns:** pUCB scores highest on every dataset and clause. On TH50 (MAP@K):

  | Clause   | Popular | ε-greedy | TS   | UCB  | **pUCB** |
  | -------- | ------- | -------- | ---- | ---- | -------- |
  | `SELECT` | 0.60    | 0.57     | 0.61 | 0.64 | **0.75** |
  | `FROM`   | 0.45    | 0.40     | 0.44 | 0.45 | **0.64** |
  | `WHERE`  | 0.56    | 0.50     | 0.53 | 0.58 | **0.69** |

  The gap is largest for `FROM` and grows as users have longer histories (e.g. `FROM` 0.74 vs. 0.43 for UCB on TH10_500).

- **Numerical regions:** all bandits score about the same here. Personalization does not help, because users rarely look at the same region twice. Popular gets worse over time, which shows that exploration is needed.
- **Parameters:** $\alpha = 0.1$ worked best. Any $0.1 \le p \le 0.9$ beat both extremes, so combining the user's own preferences with everyone else's is better than using either alone.
- **Scalability:** pUCB adds almost no overhead over UCB. With the pool fixed to all 9,903 columns, pUCB still reaches MAP@K 0.44, against 0.19 for UCB and 0.22 for Popular.

**User study.** 17 participants used _Query Builder (QB)_, a system built on this framework, on the [CORDIS](https://cordis.europa.eu/) database of European research projects. Each did NL-to-SQL and data-navigation tasks, some with QB and some writing SQL by hand. They fully completed **90% of tasks with QB, compared with 70% by hand**, and finished much faster. Every participant, including the SQL experts, said they preferred QB to writing SQL manually.

## Repository structure

| Path                                                                                           | Contents                                                                                                                    |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| [mabrecs/](mabrecs/)                                                                           | Bandit algorithms (`Ucb`, `pUcb`, `Egreedy`, `ThompsonSampling`, `Popular`) and pluggable state storage (JSON file / Redis) |
| [sdss_eval/](sdss_eval/)                                                                       | SDSS log parsing and processing, binning of numerical columns into regions and the offline evaluation harness               |
| [notebooks/sdss_log_eda/](notebooks/sdss_log_eda/)                                             | Exploratory analysis of the SDSS query logs                                                                                 |
| [notebooks/offline_evaluation/](notebooks/offline_evaluation/)                                 | Notebooks for the offline experiments                                                                                       |
| [notebooks/user_study/](notebooks/user_study/)                                                 | Notebooks with the user study logs and survey                                                                               |

## Installation

The project is managed with [Poetry](https://python-poetry.org/) and requires Python 3.11 or 3.12.

```bash
poetry install                       # core dependencies
poetry install --with evaluation     # also install the notebook/evaluation dependencies
```

`poetry install` also installs `mabrecs` and `sdss_eval` as packages, so run the notebooks with this environment as their Jupyter kernel.

## Data

[data/](data/) contains the processed SDSS data needed to run the offline evaluation:

```
data/
├── raw/
│   └── schema.csv                          # SDSS DR7 tables and columns
└── processed/
    ├── log_th{20,50,100,500,10_500}.json   # query log of each dataset
    ├── where_log_th500.json                # WHERE-clause conditions of TH500, for the region recommenders
    └── bins/
        ├── {table}-{column}.json           # min/max of each numerical column
        ├── {uniform,quantile,kmeans}/      # bins of each column per binning method (equal-width, equal-height, K-means)
        └── samples/{photoobj,specobj}.csv  # table samples the bins are computed from
```

Each line of a query log is one query: `[SELECT column ids, FROM table ids, WHERE column ids, user id]`, with ids referring to [schema.csv](data/raw/schema.csv). Each line of the WHERE log maps column ids to their conditions, e.g. `[{"3503": [["between", 0, 19.6]]}, 2]`.

The raw SDSS query log (`data/raw/log.csv`) is not included. [notebooks/sdss_log_eda/queries.ipynb](notebooks/sdss_log_eda/queries.ipynb) documents how it was extracted from the SkyServer SQL log and processed into the files above.

Paths are resolved in [sdss_eval/paths.py](sdss_eval/paths.py), so the code finds `data/` regardless of the working directory.

## Usage

A minimal pUCB example, run without persistent storage:

```python
import numpy as np
from mabrecs.bandits import pUcb

n_items = 10
bandit = pUcb(alpha=0.1, personal_ratio=0.5, init=True,
              n_arms=n_items, bias=np.zeros(n_items))

pool = np.array([0, 2, 3, 5, 7])                  # currently valid items, e.g. columns of the selected tables
recs = bandit.choose_arms(pool, top_k=3, user="alice")

# the user picked the first recommendation
bandit.update(ids=[recs[0]], rewards=[1], user="alice")
```

To persist bandit state across sessions, pass `storage=JsonFile(...)` or `storage=RedisStorage(...)` from [mabrecs/storage.py](mabrecs/storage.py) together with `arms=[...]`, then call `recommend(...)`, which returns item names instead of indexes.

## Experiments

### Offline evaluation

Evaluation on the SDSS logs for the `SELECT` / `FROM` / `WHERE` clauses and `BETWEEN` predicates. The datasets TH20, TH50, TH100 and TH500 drop users who issued more than 20, 50, 100 or 500 queries in the raw log (to filter out bots and noise). TH10_500 keeps users with between 10 and 500 queries.

| Dataset  | Users  | Queries | Distinct queries |
| -------- | ------ | ------- | ---------------- |
| TH50     | 18,793 | 99,144  | 59,132 (60%)     |
| TH100    | 19,622 | 140,903 | 83,660 (59%)     |
| TH500    | 20,261 | 236,992 | 133,165 (56%)    |
| TH10_500 | 5,678  | 204,006 | 118,735 (58%)    |

#### Table/column recommendations

- [Algorithm evaluation](notebooks/offline_evaluation/evaluation.ipynb): all algorithms on TH50, for the `SELECT`, `FROM` and `WHERE` clauses.
- [Different K values](notebooks/offline_evaluation/k.ipynb): varying the number of recommended items.
- [Different datasets](notebooks/offline_evaluation/thresholds.ipynb): TH20, TH50, TH100, TH500, TH10_500.

#### Predicate recommendations

- [Evaluation on TH500](notebooks/offline_evaluation/predicates/evaluation.ipynb): numerical region recommenders for each binning method.
- [Different K values](notebooks/offline_evaluation/predicates/k.ipynb)
- [Number of bins](notebooks/offline_evaluation/predicates/n_bins.ipynb): tuning the number of bins per column.

### User study

- [Logs](notebooks/user_study/logs.ipynb): logged results from the user study (task scores and completion times).
- [Survey](notebooks/user_study/survey.ipynb): participants' responses to the survey.

## Citation

If you use this code or build on this work, please cite:

> A. Mandamadiotis, G. Koutrika and S. Amer-Yahia, "Guided SQL-Based Data Exploration with User Feedback," _2024 IEEE 40th International Conference on Data Engineering (ICDE)_, Utrecht, Netherlands, 2024, pp. 4884-4896, doi: [10.1109/ICDE60146.2024.00372](https://doi.org/10.1109/ICDE60146.2024.00372).

```bibtex
@inproceedings{mandamadiotis2024guided,
  author    = {Mandamadiotis, Antonis and Koutrika, Georgia and Amer-Yahia, Sihem},
  title     = {Guided {SQL}-Based Data Exploration with User Feedback},
  booktitle = {2024 IEEE 40th International Conference on Data Engineering (ICDE)},
  address   = {Utrecht, Netherlands},
  year      = {2024},
  pages     = {4884--4896},
  publisher = {IEEE},
  doi       = {10.1109/ICDE60146.2024.00372},
  keywords  = {Structured Query Language; Automation; Databases; Large language models; Refining; Data engineering; Real-time systems; active learning; machine learning; recommender systems; data exploration; SQL query recommendations}
}
```
