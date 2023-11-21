import json
import os
from io import StringIO

import numpy as np
import pandas as pd
import requests

from mabrecs.bandits import Ucb
from mabrecs.storage import JsonFile
from querybuilder.api.config import AbstractDatabase, settings


class Database:
    def __init__(self, database):
        if isinstance(database, str):
            self.config = self._get_database_from_name(database)
        else:
            self.config = database

    def _get_database_from_name(self, name) -> AbstractDatabase:
        for db in settings.databases:
            if name == db.id:
                return db
        raise Exception("Invalid database name")

    def read_sql(
        self,
        query: str,
        limit=500,
    ) -> pd.DataFrame:
        url = f"{settings.database_api_service}/sql/"
        options = {"database": self.config.id, "query": query, "limit": limit}
        try:
            resp = requests.post(url, json=options)
            return pd.read_json(StringIO(resp.text), orient="split")

        except Exception as e:  # pragma: no cover
            print("Unhandled Query Execution error", e, options)

    def read_schema(self) -> pd.DataFrame:
        url = f"{settings.database_api_service}/schema/"
        options = {
            "database": self.config.id,
            "blacklist_tables": self.config.blacklist_tables,
        }
        try:
            resp = requests.post(url, json=options)
            return resp.json()

        except Exception as e:  # pragma: no cover
            print("Unhandled Query Execution error", e, options)

    def get_categorical(self, table_name: str, sample: pd.DataFrame):
        counts = sample.nunique()
        cats = counts[
            (counts >= settings.categorical_lower_threshold)
            & (counts <= settings.categorical_upper_threshold)
        ]
        cats = cats[~cats.index.str.endswith("id")]

        if cats.shape[0] > 0:
            print(
                f"{cats.shape[0]} categorical columns for table {table_name}: {list(cats.index)}"
            )
        return cats

    def _parse_categorical(
        self,
        table_name: str,
        sample: pd.DataFrame,
        cat_json={"where": {"tables": {}, "arms": []}, "predicates": {}},
        idx=0,
    ):
        cats = self.get_categorical(table_name, sample)

        if cats.shape[0] > 0:
            cat_json["where"]["tables"][table_name] = []
            for col in cats.index:
                val_counts: pd.DataFrame = (
                    sample[col].value_counts(normalize=True).round(1)
                )
                val_counts = val_counts[~(val_counts.index == "")]
                id = f"{table_name}.{col}"

                bias = np.array(val_counts.values)
                n_arms = val_counts.shape[0]

                storage = JsonFile(
                    f"{settings.production_dir}{self.config.id}-{id}.json"
                )
                Ucb(
                    settings.ucb_alpha,
                    True,
                    n_arms,
                    bias,
                    storage=storage,
                    arms=list(val_counts.index),
                )

                # cat_json["predicates"][id] = {
                #     "arms": list(val_counts.index),
                #     "ucb": ucb.toDict(),
                # }

                cat_json["where"]["arms"].append(id)
                cat_json["where"]["tables"][table_name].append(idx)
                idx += 1

        return cat_json, idx

    def export_db_to_json(self, limit=100000):
        info = self.read_schema()

        cat_json = {"where": {"tables": {}, "arms": []}, "predicates": {}}
        idx = 0
        for table_name in info["tables"]:
            if table_name not in self.config.blacklist_predicate_tables:
                if len(self.config.whitelist_predicate_columns) == 0:
                    q = f"SELECT * FROM {table_name}"
                else:
                    white_cols = []
                    for white_col in self.config.whitelist_predicate_columns:
                        if white_col.startswith(table_name):
                            white_cols.append(white_col)
                    q = f"SELECT {','.join(white_cols)} FROM {table_name}"

                sample = self.read_sql(q, limit=limit)

                cat_json, idx = self._parse_categorical(
                    table_name, sample, cat_json, idx
                )

        info["predicates"] = cat_json["where"]["arms"]

        type = {"from": "tables", "select": "columns", "where": "predicates"}

        # init idRecs

        with open(settings.idrecs_file, "r", encoding="utf8") as f:
            idrecs = json.load(f)

        arms = {
            "select": len(info["columns"]),
            "from": len(info["tables"]),
            "where": len(info["predicates"]),
        }

        for clause, n_arms in arms.items():
            bias = np.zeros(n_arms)

            if self.config.id in idrecs:
                for name in idrecs[self.config.id][clause]:
                    try:
                        idx = info[type[clause]].index(name)
                        bias[idx] = idrecs[self.config.id][clause][name]
                    except ValueError:
                        continue

            storage = JsonFile(
                f"{settings.production_dir}{self.config.id}-{clause}.json"
            )

            Ucb(
                settings.ucb_alpha,
                True,
                n_arms,
                bias,
                storage=storage,
                arms=info[type[clause]],
            )

        db_json = {
            "select": {
                "tables": info["table"],
            },
            "from": {
                "joins": info["joins"],
            },
            "where": {
                "tables": cat_json["where"]["tables"],
            },
            # "predicates": cat_json["predicates"],
        }

        os.makedirs(f"{settings.production_dir}", exist_ok=True)
        with open(
            f"{settings.production_dir}{self.config.id}.json",
            "w",
            encoding="utf8",
        ) as f:
            json.dump(db_json, f, indent=4, ensure_ascii=False)
