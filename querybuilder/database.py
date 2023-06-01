import json

import numpy as np
import pandas as pd
import requests

from querybuilder.api.config import settings
from querybuilder.bandits import Ucb


class Database:
    def __init__(self, database):
        if isinstance(database, str):
            self.config = self._get_database_from_name(database)
        else:
            self.config = database

        self.schemas = ",".join(["'" + k + "'" for k in self.config.schemas])

    def _get_database_from_name(self, name):
        for db in settings.databases:
            if name in db.aliases:
                return db
        raise Exception("Invalid database name")

    def read_sql(
        self,
        query: str,
        limit=500,
    ) -> pd.DataFrame:

        url = f"{settings.database_api_service}/api/database/sql/"
        options = {"database": self.config.db_name, "query": query, "limit": limit}
        try:
            resp = requests.post(url, json=options)
            return pd.read_json(resp.text, orient="split")

        except Exception as e:  # pragma: no cover
            print("Unhandled Query Execution error", e, options)

    def _parse_tables_and_columns(self, results: pd.DataFrame):
        column_id = 0
        prev_table = ""
        parsed = {"tables": [], "columns": [], "table": {}}

        for _, row in results.iterrows():
            table, column = row

            if prev_table != table:
                parsed["tables"].append(table)
                parsed["table"][table] = []

            parsed["columns"].append(table + "." + column)
            parsed["table"][table].append(column_id)

            column_id += 1
            prev_table = table

        return parsed

    def get_tables_and_columns(self):
        blacklist = " AND ".join(
            ["table_name not like '" + k + "'" for k in self.config.blacklist_tables]
        )
        q = f"""
            SELECT table_name,column_name
            FROM information_schema.COLUMNS
            WHERE table_schema in ({self.schemas})
            AND {blacklist}
        """
        results = self.read_sql(q)
        return self._parse_tables_and_columns(results)

    def _parse_joins(self, results: pd.DataFrame):
        joins = {}
        for _, join in results.iterrows():
            for i in [0, 2]:
                thisTable = join[i]
                otherTable = join[0] if i == 2 else join[2]

                if thisTable not in joins:
                    joins[thisTable] = {}

                if otherTable not in joins[thisTable]:
                    joins[thisTable][otherTable] = []

                condition = join[0] + "." + join[1] + "=" + join[2] + "." + join[3]
                joins[thisTable][otherTable].append(condition)

        for tableA, valA in joins.items():
            for tableB, valB in valA.items():
                joins[tableA][tableB] = " AND ".join(valB)

        return joins

    def get_joins(self):
        if settings.db_type != "mysql":
            query = f"""
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM
                information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' and tc.table_schema in ({self.schemas})
            """
        else:
            query = f"""
            SELECT
                TABLE_NAME,
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM
                INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE REFERENCED_COLUMN_NAME is not null
            AND CONSTRAINT_SCHEMA in ({self.schemas})
            """
        results = self.read_sql(query)
        return self._parse_joins(results)

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
                ucb = Ucb(settings.ucb_alpha, True, n_arms, bias)

                cat_json["predicates"][id] = {
                    "arms": list(val_counts.index),
                    "ucb": ucb.toDict(),
                }

                cat_json["where"]["arms"].append(id)
                cat_json["where"]["tables"][table_name].append(idx)
                idx += 1

        return cat_json, idx

    def export_db_to_json(self):
        info = self.get_tables_and_columns()

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

                sample = self.read_sql(q, limit=10000)

                cat_json, idx = self._parse_categorical(
                    table_name, sample, cat_json, idx
                )

        info["predicates"] = cat_json["where"]["arms"]

        type = {"from": "tables", "select": "columns", "where": "predicates"}

        db_json = {
            "select": {
                "tables": info["table"],
            },
            "from": {
                "joins": self.get_joins(),
            },
            "where": cat_json["where"],
            "predicates": cat_json["predicates"],
        }

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

            for name in idrecs[self.config.id][clause]:
                try:
                    idx = info[type[clause]].index(name)
                    bias[idx] = idrecs[self.config.id][clause][name]
                except ValueError:
                    continue

            ucb = Ucb(settings.ucb_alpha, True, n_arms, bias)

            db_json[clause]["arms"] = info[type[clause]]
            db_json[clause]["ucb"] = ucb.toDict()

        db_json["users"] = {
            "select": {},
            "from": {},
            "where": {},
        }

        with open(
            f"{settings.production_dir}{self.config.id}.json", "w", encoding="utf8"
        ) as f:
            json.dump(db_json, f, indent=4, ensure_ascii=False)
