import csv
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sdss_eval import sqlparser
from sdss_eval.paths import PROCESSED_DIR, RAW_DIR
from tqdm import tqdm


class statistics:
    def __init__(
        self,
        log_file="log_th50.json",
        max_iterations=None,
        skip_lines=None,
        clauses=["Select", "From", "Where"],
    ):
        self.db = sqlparser.schema()
        self.clauses = clauses
        log_file = PROCESSED_DIR / log_file
        self.popular = self.popular_counters(log_file, max_iterations, skip_lines)

    def popular_counters(self, log_file, max_iterations, skip_lines):
        # Calculate most popular identifiers for each clause
        if max_iterations and max_iterations > 0 and max_iterations < 1:
            total_logs = sum(1 for line in open(log_file))
            max_iterations *= total_logs

        if skip_lines and skip_lines > 0 and skip_lines < 1:
            total_logs = sum(1 for line in open(log_file))
            skip_lines *= total_logs
            skip_lines = round(skip_lines)

        counter = [[] for i in self.clauses]

        counter[1] = [0] * self.db.n_tables  # from

        for i in [i for i, p in enumerate(self.clauses) if i != 1]:
            counter[i] = [0] * self.db.n_columns
        log_events = [0 for i in self.clauses]
        j = 0
        with open(log_file) as file:
            if skip_lines is not None:
                for _ in range(skip_lines):
                    next(file)

            for line in file:
                j += 1
                row = json.loads(line)
                for i in range(len(self.clauses)):
                    if len(row[i]) > 0:
                        log_events[i] += 1
                        for x in row[i]:
                            counter[i][x] += 1

                if max_iterations and j > max_iterations:
                    break

        for i, c in enumerate(log_events):
            print(self.clauses[i], ":", c, "events")
        return counter

    def get_table_name(self, id):
        for table, val in self.db.schema.items():
            if val["__id__"] == id:
                return table

    def get_column_name(self, id):
        for table, columns in self.db.schema.items():
            for column, val in columns.items():
                if column != "__id__" and val == id:
                    return table + "." + column

    def most_popular_identifiers(self, n):
        all = {}
        for c, clause in enumerate(self.clauses):
            clause = self.clauses[c]
            get_id_func = self.get_table_name if c == 1 else self.get_column_name
            counts = Counter({get_id_func(k): v for k, v in enumerate(self.popular[c])})
            # print(clause, "top", n, "identifiers", counts.most_common(n))
            all[clause] = counts.most_common(n)

        print(all)

    def used_identifiers(self):
        unused = []
        for c, clause in enumerate(self.clauses):
            get_id_func = self.get_table_name if c == 1 else self.get_column_name
            n = self.db.n_tables if c == 1 else self.db.n_columns

            unused.append(
                [get_id_func(k) for k, v in enumerate(self.popular[c]) if v == 0]
            )
            print(
                clause,
                "clause",
                "used identifiers:",
                n - len(unused[c]),
                "out of",
                n,
                100 - round(len(unused[c]) / n * 100),
                "%",
            )


class user_context:
    def __init__(self, clause, split=0.1, log_file="log_th50.json"):
        db = sqlparser.schema()
        self.clause = clause
        self.n = db.get_arms(clause)

        self.users = {}
        user_counters = {}

        self.log_file = log_file
        self.split = split
        self.total_logs = sum(1 for line in open(PROCESSED_DIR / log_file))
        max_iterations = split * self.total_logs

        i = 0
        for line in tqdm(open(PROCESSED_DIR / log_file)):
            i += 1
            row = json.loads(line)
            user = row[-1]
            if len(row[clause]) > 0:
                if user not in self.users:
                    self.users[user] = [0 for id in range(self.n)]
                    user_counters[user] = 0

            for identifier in row[clause]:
                self.users[user][identifier] += 1
                user_counters[user] += 1

            if max_iterations and i > max_iterations:
                break

        # convert to frequencies
        for user, counters in self.users.items():
            for id, count in enumerate(counters):
                self.users[user][id] = count / user_counters[user]

        self.users = list(self.users.values())
        self.clauses = ["Select", "From", "Where"]

    def PCA_plot_components(self):
        pca = PCA().fit(self.users)

        fig, ax = plt.subplots()
        ax.plot(pca.explained_variance_)

        ax.set(xlabel="Principal Components", ylabel="Explained Variances")
        ax.set_xlim(1, 25)
        ax.set_xticks([1, 5, 10, 15, 20, 25])

        tikzplotlib.clean_figure()
        tikzplotlib.save(
            "plots/pca_" + self.clauses[self.clause] + ".tex",
            extra_axis_parameters=["scaled x ticks=false", "scaled y ticks=false"],
            axis_height="\\figH",
            axis_width="\\figW",
        )

    def PCA_fit(self, n_components):
        self.PCA = PCA(n_components=n_components).fit(self.users)
        self.components = n_components

    def TSNE(self):
        data = self.PCA.fit_transform(self.users)
        X_embedded = TSNE(n_components=2).fit_transform(data)
        plt.plot(X_embedded[:, 0], X_embedded[:, 1], "bo", markersize=2)
        plt.title(self.clauses[self.clause] + " clause")
        plt.show()

    def transform(self, add_one=True):

        log = open(PROCESSED_DIR / self.log_file)
        skip_lines = round(self.split * self.total_logs)
        for _ in range(skip_lines):
            next(log)

        users = {}
        user_counters = {}

        out = self.log_file.replace(
            "log_", "log_" + str(self.clause) + "_pca" + str(self.components) + "_"
        )
        if add_one:
            out = out.replace("_th", "_1_th")

        outfile = open(
            PROCESSED_DIR / out,
            "w",
        )
        i = 0
        for line in tqdm(log, total=self.total_logs - skip_lines):
            row = json.loads(line)
            user = row[-1]

            if len(row[self.clause]) > 0:
                user_features = [0 for id in range(self.n)]
                if user not in users:
                    users[user] = user_features
                    user_counters[user] = 0
                else:
                    for id, count in enumerate(users[user]):
                        user_features[id] = count / user_counters[user]

                transf = self.PCA.transform([user_features])[0]

                if add_one:
                    row[-1] = list(np.append((transf + 1) / 2, [1]))
                else:
                    row[-1] = list((transf + 1) / 2)

                json.dump(row, outfile)
                outfile.write("\n")

                for identifier in row[self.clause]:
                    users[user][identifier] += 1
                    user_counters[user] += 1

            # i+=1
            # if i%50000==0: # refit PCA every 50.000 iterations
            #     pca = pca.fit(list(users.values()))


class create_query_log:
    def __init__(
        self,
        threshold=None,
        log=RAW_DIR / "log.csv",
        processed_dir=PROCESSED_DIR,
        lower_threshold=None,
        where=False,
    ):
        self.log = log
        self.parsed = {}
        self.errors = Counter()
        self.processed_dir = Path(processed_dir)
        self.where = where

        if threshold:
            self.th = threshold
            self.lower_threshold = lower_threshold

            self.create_blacklist()
            self.create_hashes()
            self.process_hashes()

    def users(self):
        self.ips = Counter()
        with open(self.log, "r", encoding="ISO-8859-1") as file:
            next(file)
            for line in file:
                ip = line.split(",")[0]
                self.ips[ip] += 1

        self.total_queries = sum(self.ips.values())
        print("Queries in the log:", self.total_queries)
        print("Users in the log:", len(self.ips))

    def create_blacklist(self):
        if not hasattr(self, "ips"):
            self.users()

        self.blacklist = {}
        for ip in self.ips:
            queries = self.ips[ip]
            if queries > self.th or (
                self.lower_threshold is not None and queries < self.lower_threshold
            ):
                self.blacklist[ip] = True

    def create_hashes(self):
        num_lines = sum(1 for line in open(self.log, "r", encoding="ISO-8859-1"))

        hashes = {}
        users = {}
        userid = 0

        tsv_hashes = csv.writer(
            open(self.processed_dir / "hashes.tsv", "wt"), delimiter="\t"
        )
        tsv_log = csv.writer(open(self.processed_dir / "log.tsv", "wt"), delimiter="\t")

        # tsv_hashes.writerow(["hash", "query"])
        tsv_log.writerow(["user", "hash"])

        total = 0
        duplicates = 0
        blacklisted = 0

        prev_line = ""
        invalid = 0
        with open(self.log, encoding="ISO-8859-1") as file:
            next(file)
            for line in tqdm(file, total=num_lines):

                split = line.split(",")

                query = ",".join(split[1:])
                if query.count("select") == 1 and query.count(" from ") == 1:
                    ip = split[0]
                    if ip in self.blacklist:
                        blacklisted += 1
                        continue

                    if prev_line == line:
                        duplicates += 1
                        continue
                    else:
                        prev_line = line

                    user = users.get(ip, None)
                    if user is None:
                        userid += 1
                        users[ip] = userid
                        user = userid

                    query = query.replace("\t", " ").rstrip("\n").strip('"')
                    if self.where:
                        query = "select * from " + query.split(" from ")[1]

                    hs = hash(query)
                    if hs not in hashes:
                        hashes[hs] = True
                        tsv_hashes.writerow([hs, query])

                    tsv_log.writerow([user, hs])
                    total += 1
                else:
                    invalid += 1
        os.system(
            "shuf -o "
            + str(self.processed_dir / "hashes.tsv")
            + " < "
            + str(self.processed_dir / "hashes.tsv")
        )  # shuffle the lines

        num_lines -= invalid
        print("-" * 10, "Hash Phase", "-" * 10)
        print("Queries after removing blacklisted users:", num_lines - blacklisted)
        print(
            "Queries after removing duplicates:", num_lines - blacklisted - duplicates
        )
        print(
            "Distinct queries:",
            len(hashes),
            "out of",
            total,
            "valid queries",
            round(len(hashes) / total * 100),
            "%",
        )
        print("Total users:", len(users))

    def chunks(self, n, size):
        chunk_size = size / n
        return [(int(i * chunk_size), int((i + 1) * chunk_size), i) for i in range(n)]

    def process_chunk(self, start, end, chunk_num, progress_bar=False):
        results = {}
        errors = Counter()
        if progress_bar:
            pbar = tqdm(
                total=end - start,
                position=chunk_num,
                desc="Chunk #" + str(chunk_num + 1),
            )
        db = sqlparser.schema(db_stats=(self.where == True))
        sqlparser.fix_sqlparse_keywords()
        with open(self.processed_dir / "hashes.tsv", encoding="ISO-8859-1") as file:
            # next(file)
            tsvreader = csv.reader(file, delimiter="\t")
            for i, line in enumerate(tsvreader):
                if i >= start and i < end:
                    if progress_bar:
                        pbar.update(1)
                    try:
                        if not self.where:
                            parse = sqlparser.Parse(line[1], db, clauses=[0, 1, 2])
                            results[line[0]] = parse.indexes
                        else:
                            results[line[0]] = sqlparser.ParseWhere(line[1], db).final

                    except Exception as error:
                        error = str(error)
                        errors[error] += 1

        if progress_bar:
            pbar.close()
        return (results, errors)

    def get_result(self, result):
        self.parsed.update(result[0])
        self.errors += result[1]

    def process_hashes(
        self, cpus=mp.cpu_count(), remove_dups=False
    ):  # remove_dups remove lines with same ids
        num_hashes = sum(
            1
            for line in open(
                self.processed_dir / "hashes.tsv", "r", encoding="ISO-8859-1"
            )
        )

        pool = mp.Pool(cpus)
        for chunk in self.chunks(cpus, num_hashes):
            pool.apply_async(self.process_chunk, args=chunk, callback=self.get_result)
        pool.close()
        pool.join()

        print("-" * 10, "Parse Phase", "-" * 10)
        print("Total unique errors:", len(self.errors))
        distinct_count = len(self.parsed)
        print(
            "Parsed distinct queries:",
            distinct_count,
            "out of",
            num_hashes,
            round(distinct_count / num_hashes * 100),
            "%",
        )

        json.dump(
            self.errors.most_common(),
            open(self.processed_dir / "errors.json", "w"),
            indent=4,
        )
        # pickle.dump(parsed, open(processed_dir + "parsed_idx.p", "wb"))

        output_file = "log_th" if not self.where else "where_log_th"

        if self.lower_threshold is not None:
            output_file += str(self.lower_threshold) + "_"
        output_file += str(self.th) + ".json"
        outfile = open(self.processed_dir / output_file, "w")
        outfile_not_parsed = open(self.processed_dir / "not_parsed.txt", "w")

        not_parsed = 0
        total = 0
        users = {}
        with open(self.processed_dir / "log.tsv") as file:
            next(file)
            tsvreader = csv.reader(file, delimiter="\t")
            prev_row = None
            dups = 0
            for line in tsvreader:
                try:
                    query = self.parsed[line[1]]
                    userid = int(line[0])
                    if self.where:
                        row = [query, userid]
                    else:
                        row = query + [userid]
                    if not remove_dups or prev_row != row:
                        json.dump(row, outfile)
                        outfile.write("\n")
                        total += 1
                        prev_row = row
                        if userid not in users:
                            users[userid] = True
                    else:
                        dups += 1

                except KeyError:
                    outfile_not_parsed.write(line[1])
                    outfile_not_parsed.write("\n")
                    not_parsed += 1
                    continue

        print("Failed to parse:", not_parsed)
        if remove_dups:
            print("Duplicates removed:", dups)
        print(
            "Final queries:",
            total,
            "Users:",
            len(users),
            "Distinct:",
            distinct_count,
            round(distinct_count / total * 100),
            "%",
        )
