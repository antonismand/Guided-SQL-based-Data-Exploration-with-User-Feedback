import json
from os.path import exists

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from querybuilder.database import Database


class Binning:
    def __init__(self, attributes: list[str], dir="data/processed/bins"):
        self.attributes = attributes
        self.dir = dir

        table, _ = self.attributes[0].split(".")
        path = f"{dir}/samples/{table}.csv"
        if exists(path):
            # print(f"Using sample from {path}.")
            self.data = pd.read_csv(path)
        else:
            print(f"{path} does not exist.")
            print("Run get_sample() to retrieve a sample from the database.")

    def get_sample(self, db_name: str, sample=0.01):
        db = Database(db_name)
        columns = (",").join([col.split(".")[1] for col in self.attributes])

        table, _ = self.attributes[0].split(".")

        query = f"SELECT {columns} FROM {table}"
        path = f"{self.dir}/samples/{table}.csv"
        self.data = db.read_sql(query, sample)
        self.data.to_csv(path, index=False)

    def create_bins(self, n_bins: list[int], strategy="uniform"):
        for i, attr in enumerate(self.attributes):
            _, column = attr.split(".")

            X = self.data[column].to_numpy().reshape(-1, 1)
            est = KBinsDiscretizer(n_bins=n_bins[i], strategy=strategy, subsample=None)
            est.fit(X)

            self.bins = []
            for b, bin in enumerate(est.bin_edges_[0]):
                if b != n_bins[i]:
                    self.bins.append(
                        (round(bin, 6), round(est.bin_edges_[0][b + 1], 6))
                    )

            with open(
                f"{self.dir}/{strategy}/{attr.replace('.','-')}.json",
                "w+",
                encoding="utf-8",
            ) as f:
                json.dump(self.bins, f, ensure_ascii=False, indent=4)
