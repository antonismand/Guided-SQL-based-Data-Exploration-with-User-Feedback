import json
import re

import numpy as np
import sqlparse
from mo_sql_parsing import parse
from sqlparse import keywords
from sqlparse.sql import (Comparison, Function, Identifier, IdentifierList,
                          Operation, Where)
from sqlparse.tokens import DML, Keyword, Name, Punctuation

from sdss_eval.paths import BINS_DIR, RAW_DIR

# schema = {}


class schema:
    def __init__(self, file=RAW_DIR / "schema.csv"):
        self.schema = {}
        self.schema_idx = {}
        self.n_tables = 0
        self.n_columns = 0

        with open(file) as f:
            next(f)
            next(f)
            for line in f:
                l = line.split(",")
                if l[0] not in self.schema:
                    self.schema[l[0]] = {"__id__": self.n_tables}
                    self.n_tables += 1
                self.schema[l[0]][l[1].rstrip("\n")] = self.n_columns
                self.n_columns += 1

        # create indexes
        for table, val in self.schema.items():
            id = val["__id__"]
            columns = list(val.values())
            columns.pop(0)
            self.schema_idx[id] = columns

    def get_arms(self, clause):
        if clause == 1:
            return self.n_tables
        return self.n_columns

    def get_table_name(self, id):
        for table, val in self.schema.items():
            if val["__id__"] == id:
                return table

    def get_column_name(self, id):
        for table, columns in self.schema.items():
            for column, val in columns.items():
                if column != "__id__" and val == id:
                    return table + "." + column

    def create_bins(self, discretization_method, columns):
        self.whitelist_columns = []
        self.whitelist_tables = []
        self.stats = {}
        self.bins = []
        self.bins_idx = {}

        bins_idx = 0

        for attr in columns:
            table, col = attr.split(".")
            col_id = self.schema[table][col]
            self.whitelist_columns.append(col_id)

            if table not in self.whitelist_tables:
                self.whitelist_tables.append(table)

            with open(BINS_DIR / f"{table}-{col}.json") as json_file:
                s = json.load(json_file)
                self.stats[col_id] = {"min": s["min"], "max": s["max"]}

            with open(
                BINS_DIR / discretization_method / f"{table}-{col}.json"
            ) as json_file:
                bins = json.load(json_file)
                n_bins = len(bins)

                self.bins.extend(bins)

                self.bins_idx[col_id] = [i for i in range(bins_idx, bins_idx + n_bins)]
                bins_idx += n_bins

        self.bins = np.array(self.bins)


def fix_sqlparse_keywords():
    db = schema()
    removed_keywords = []
    # remove sqlparse keywords that match with column/table names
    for table in db.schema:
        if table.upper() in keywords.KEYWORDS:
            del keywords.KEYWORDS[table.upper()]
            removed_keywords.append(table.upper())
        for column in db.schema[table]:
            if column.upper() in keywords.KEYWORDS:
                del keywords.KEYWORDS[column.upper()]
                removed_keywords.append(column.upper())

    if "LINE" in keywords.KEYWORDS_PLPGSQL:  # a common alias name
        del keywords.KEYWORDS_PLPGSQL["LINE"]
        removed_keywords.append("LINE")

    # print("Removed Keywords from sqlparse:", removed_keywords)


class Parse:
    def __init__(self, query, db, clauses):
        if query.count("select") > 1:
            raise Exception("Nested queries are not supported")

        query = (
            query.rstrip("\n")
            .strip('"')
            .replace(" as ", " ")
            .replace("[", "")
            .replace("]", "")
        )
        query = re.sub("top (\d)+ ", "", query)
        query = sqlparse.parse(query)
        if query[0].tokens[0].ttype is not DML:
            raise Exception("Parsing problem")

        # self.parsed = [[], [], [], [], []]
        # self.indexes = [[], [], [], [], []]
        self.parsed = [[] for i in clauses]
        self.indexes = [[] for i in clauses]

        self.aliases = {}
        self.db = db

        # print(query[0]._pprint_tree())

        for token in query[0].tokens:
            if token.ttype is DML:
                self.cursor = 0
            elif token.ttype is Keyword:
                if token.value == "from" or token.value == "join":
                    self.cursor = 1
                elif token.value == "where" or token.value == "on":
                    self.cursor = 2
                elif token.value == "group by":
                    self.cursor = 3
                elif token.value == "order by":
                    self.cursor = 4

            elif isinstance(token, Where):
                self.cursor = 2
                if 2 in clauses:
                    [self.parseForIdentifier(tok) for tok in token]

            elif self.cursor in clauses:
                self.parseForIdentifier(token)

        # print(self.parsed[0])
        # print(self.parsed[1])
        clauses.remove(1)  # ignore from
        for op in clauses:
            for k, column in self.reverse_enum(self.parsed[op]):
                (
                    self.indexes[op][k],
                    self.parsed[op][k],
                ) = self.resolveColumn(column)

                if self.indexes[op][k] is None:
                    del self.indexes[op][k]
                    del self.parsed[op][k]

            self.indexes[op] = sorted(self.indexes[op])

        # if not any(self.indexes[0]):
        #     raise Exception("No columns resolved")

    def reverse_enum(self, L):
        for index in reversed(range(len(L))):
            yield index, L[index]

    def parseForIdentifier(self, token):
        if isinstance(token, Identifier):
            tokens = token.tokens
            if tokens[0].ttype is Name:
                if self.cursor == 1:
                    self.parseFrom(tokens)
                else:
                    self.parsed[self.cursor].append(tokens)
                    self.indexes[self.cursor].append(None)

            elif isinstance(tokens[0], Operation):  # check this one
                [self.parseForIdentifier(x) for x in tokens[0]]

        elif isinstance(token, (IdentifierList, Comparison)):
            [self.parseForIdentifier(x) for x in token]
        elif isinstance(token, Function):
            # for x in token.tokens[1].tokens:
            #     print(x)
            [self.parseForIdentifier(x) for x in token.tokens[1].tokens]
        elif token.ttype is Keyword and token.value == "from":
            self.cursor = 1

    def parseFrom(self, tokens):
        pos = 0
        for i, token in enumerate(tokens):
            if token.ttype is Punctuation:  # handle schemaA.tableA -> get tableA
                pos = i + 1
        table = tokens[pos].value
        # table_idx = schema.get(table, None)
        # if table_idx is None:
        # table_idx = table_idx.get("__id__")

        if table in self.db.schema:
            table_idx = self.db.schema[table]["__id__"]
        elif "(" in table:
            raise Exception("Function " + table.split("(")[0] + " in from clause")
        else:
            raise Exception("Table " + table + " not in schema")

        self.aliases[tokens[-1].value] = table
        self.parsed[1].append(table)
        self.indexes[1].append(table_idx)

    def resolveColumn(self, tokens):
        id = None
        resolved = None
        if len(tokens) == 1 or tokens[1].ttype is not Punctuation:  # no alias
            column = tokens[0].value
            if len(self.parsed[1]) == 1:  # resolve from unique table
                # print("resolving", self.parsed[1][0], column)
                return self.getColumnId(self.parsed[1][0], column)
            else:
                for t in self.parsed[1]:  # try to resolve from tables in from clause
                    id, resolved = self.getColumnId(t, column)
                    if id is not None:
                        return id, resolved
        else:  # alias exists
            column = tokens[2].value
            # alias = self.aliases.get(tokens[0].value, None)
            if tokens[0].value in self.aliases:  # proper handling of alias s.fiberid
                return self.getColumnId(self.aliases[tokens[0].value], column)
            elif (
                tokens[0].value in self.parsed[1]
            ):  # handling of specobj.fiberid when there is also alias s
                return self.getColumnId(tokens[0].value, column)

            # if alias is not None:  # alias resolved from tables in query
            # id, resolved = self.getColumnId(alias, column)

        # if id is not None:
        return None, None
        # raise Exception("Column " + column + " could not be resolved")

    def getColumnId(self, table, column):
        return self.db.schema[table].get(column, None), table + "." + column


class ParseWhere:
    def __init__(self, sql, db: schema):
        try:
            mo = parse(sql)
        except Exception:
            raise Exception("Mo parsing error")

        if "where" not in mo:
            raise Exception("No where clause")

        self.final = {}
        self.db = db

        self.tables = []
        self.aliases = {}

        if isinstance(mo["from"], list):
            for fr in mo["from"]:
                self.parse_from(fr)
        else:
            self.parse_from(mo["from"])

        if not any([table in db.whitelist_tables for table in self.tables]):
            raise Exception("No whitelisted table in FROM")

        self.parse_and_or(mo["where"])

        for col, val in list(self.final.items()):
            if isinstance(col, str):
                col_id = self.resolve_column(col)

                if val and col_id in db.whitelist_columns:
                    self.final[col_id] = val
            del self.final[col]

        for col_id, conditions in list(self.final.items()):
            for key, condition in enumerate(conditions):
                if condition[0] in ["gt", "gte", "lt", "lte"]:
                    self.final[col_id][key] = self.add_comparison(col_id, condition)

        if not self.final:
            raise Exception("Nothing parsed in where")

    def resolve_column(self, table_column):
        if isinstance(table_column, str):
            if "." in table_column:
                table, column = table_column.split(".")
                if table not in self.tables:
                    table = self.aliases[table]

                return self.db.schema[table][column]
            else:
                for table in self.tables:
                    if table_column in self.db.schema[table]:
                        return self.db.schema[table][table_column]

    def add_comparison(self, col_id, condition):
        if condition[0] in ["gt", "gte"]:
            return ["between", condition[1], self.db.stats[col_id]["max"]]
        return ["between", self.db.stats[col_id]["min"], condition[1]]

    def add_new_value(self, col, val):
        if not isinstance(col, str):
            return None

        if type(val[1]) is dict:
            return None

        if len(val) > 2 and type(val[2]) is dict:
            return None

        if val[0] == "between":
            if type(val[1]) != int and type(val[1]) != float:
                return None
            if type(val[2]) != int and type(val[2]) != float:
                return None
        elif val[0] in ["gt", "gte", "lt", "lte"]:
            if type(val[1]) != int and type(val[1]) != float:
                return None

        if col not in self.final:
            self.final[col] = []

        self.final[col].append(val)

    def parse_and_or(self, obj, condition=""):
        if isinstance(obj, dict):
            condition = list(obj.keys())[0]
            item = obj[condition]
            if condition in ["and", "or"]:
                self.parse_and_or(item, condition)
            else:
                #     if condition == "eq" and isinstance(*item[1:], str):
                #         if self.check_if_string_is_column(*item[1:]):
                #             raise Exception("only contains join")
                if condition in ["gt", "gte", "lt", "lte", "between"]:
                    self.add_new_value(item[0], [condition, *item[1:]])

        elif isinstance(obj, list):
            temp = {}
            for item in obj:
                key = list(item.keys())[0]
                val = item[key]
                if key == "or" or key == "and":
                    self.parse_and_or(val, key)
                # elif key in ["eq", "neq", "gt", "gte", "lt", "lte", "in"]:
                elif key in ["gt", "gte", "lt", "lte"]:
                    col = val[0]
                    if not isinstance(col, str):
                        continue

                    if condition == "and" and key in ("gte", "lte"):
                        if col not in temp:
                            temp[col] = [key, val[1]]
                        else:
                            if key == "gte":
                                gt = val[1]
                                lt = temp[col][1]
                            else:
                                lt = val[1]
                                gt = temp[col][1]

                            self.add_new_value(col, ["between", gt, lt])
                            del temp[col]
                    # elif key == "eq" and isinstance(val[1], str):
                    #     if not self.check_if_string_is_column(val[1]):
                    #         self.final[col].append([key, val[1]])
                    else:
                        self.add_new_value(col, [key, val[1]])
                elif key == "between":
                    self.add_new_value(val[0], [key, val[1], val[2]])

            if condition == "and":
                for col, val in temp.items():
                    self.add_new_value(col, val)
        return self.final

    def parse_from(self, obj):
        if isinstance(obj, str):
            if obj not in self.db.schema:
                raise Exception("table not in schema")
            self.tables.append(obj)

        elif isinstance(obj, dict):
            if "value" in obj:
                if obj["value"] not in self.db.schema:
                    raise Exception("table not in schema")

                self.aliases[obj["name"]] = obj["value"]
                self.tables.append(obj["value"])

            else:
                join_key = [key for key in obj.keys() if "join" in key]
                if len(join_key) == 1:
                    self.parse_from(obj[join_key[0]])

    # def check_if_string_is_column(self, string):
    #     if "." in string:
    #         s = string.split(".")
    #         if s[0] in self.aliases:
    #             t = self.aliases[s[0]]
    #             return t in self.db.schema and s[1] in self.db.schema[t]
    #         return s[0] in self.db.schema and s[1] in self.db.schema[s[0]]
    #     else:
    #         for table in self.tables:
    #             if string in self.db.schema[table]:
    #                 return True
    #     return False
