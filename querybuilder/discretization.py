import json
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from querybuilder.database import Database

PATH = "data/processed/bins"


class Binning:
    def __init__(self, db_name: str, attributes: list[str], sample=0.01):
        self.attributes = attributes

        table, _ = self.attributes[0].split(".")
        path = f"{PATH}/samples/{table}.csv"
        if exists(path):
            print(f"Using sample from {path}.")
            self.data = pd.read_csv(path)
        else:
            print(f"{path} does not exist. Retrieving sample for {table}")
            db = Database(db_name)
            columns = (",").join([col.split(".")[1] for col in self.attributes])

            query = f"SELECT {columns} FROM {table}"
            self.data = db.read_sql(query, sample)
            self.data.to_csv(path, index=False)


class EqualWidthBinning(Binning):
    def create_bins(self):
        for attr in self.attributes:
            _, column = attr.split(".")

            np_data = self.data[column].to_numpy()
            bin_width, _ = self.freedman_diaconis(np_data)

            bins = [
                (x, x + bin_width)
                for x in np.arange(
                    self.data[column].min(),
                    self.data[column].max(),
                    bin_width,
                )
            ]

            attr = attr.replace(".", "-")
            with open(
                f"{PATH}/equal-width/{attr.replace('.','-')}.json",
                "w+",
                encoding="utf-8",
            ) as f:
                json.dump(bins, f, ensure_ascii=False, indent=4)

    def freedman_diaconis(self, data: np.ndarray):
        n = data.size

        q3, q1 = np.percentile(data, [75, 25])
        iqr = q3 - q1

        bin_width = (2 * iqr) / (n ** (1 / 3))
        bin_count = int(np.ceil((data.max() - data.min()) / bin_width))

        bin_width = round(bin_width, 4)

        return bin_width, bin_count


class EqualHeightBinning(Binning):
    def create_bins(self):

        for attr in self.attributes:
            _, column = attr.split(".")

            df = self.data[column]
            bin_count = 2 * (df.size ** (2 / 5))
            bin_count = int(np.ceil(bin_count))

            bins = pd.qcut(df, q=bin_count).cat.categories
            bins = [(x.left, x.right) for x in bins]

            attr = attr.replace(".", "-")
            with open(
                f"{PATH}/equal-height/{attr.replace('.','-')}.json",
                "w+",
                encoding="utf-8",
            ) as f:
                json.dump(bins, f, ensure_ascii=False, indent=4)


class KMeansBinning(Binning):
    def analyze_bins(self, kmin=4, kmax=64, step=4):

        k_values_to_test = np.arange(kmin, kmax, step=step)
        self.best_ks = {}

        for attr in self.attributes:
            table, column = attr.split(".")

            sil = []
            x = self.data[column].to_numpy().reshape(-1, 1)

            for k in k_values_to_test:
                kmeans = KMeans(n_clusters=k, n_init="auto").fit(x)
                labels = kmeans.labels_
                sil.append(silhouette_score(x, labels, metric="euclidean"))

            plt.plot(k_values_to_test, sil)
            plt.title(table + "." + column)
            plt.show()
            self.best_ks[attr] = k_values_to_test[np.argmax(sil)]

        return self.best_ks

    def create_bins(self, n_clusters=None):
        for attr in self.attributes:
            if n_clusters is None:
                k = self.best_ks[attr]
            else:
                k = n_clusters
            _, column = attr.split(".")

            x = self.data[column].to_numpy().reshape(-1, 1)
            kmeans = KMeans(n_clusters=k, n_init="auto").fit(x)

            bins = np.array([[np.inf, np.NINF]] * k)
            for i, val in enumerate(kmeans.labels_):
                if x[i][0] > bins[val][1]:
                    bins[val][1] = np.round(x[i][0], 3)
                if x[i][0] < bins[val][0]:
                    bins[val][0] = np.round(x[i][0], 3)

            sorted_bins = bins[bins[:, 0].argsort()]

            attr = attr.replace(".", "-")
            with open(
                f"{PATH}/kmeans/{attr.replace('.','-')}.json",
                "w+",
                encoding="utf-8",
            ) as f:
                json.dump(sorted_bins.tolist(), f, ensure_ascii=False, indent=4)
