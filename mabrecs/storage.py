import json
import os
from abc import ABC, abstractmethod
from os.path import exists

from redis import Redis


class Storage(ABC):
    @abstractmethod
    def get(self, key: str):
        pass

    @abstractmethod
    def get_array_index(self, key: str, index: int):
        pass

    @abstractmethod
    def get_nested_key(self, key: str, subkey: str):
        pass

    @abstractmethod
    def set(self, key: str, value: dict):
        pass


class JsonFile(Storage):
    def __init__(self, file: str):
        self.file = file
        self.jsn = {}
        if exists(self.file):
            with open(f"{self.file}", "r", encoding="utf8") as f:
                self.jsn = json.load(f)

    def get(self, key: str):
        if key in self.jsn:
            return self.jsn[key]
        return None

    def get_array_index(self, key: str, index: int):
        return self.get(key)[index]

    def get_nested_key(self, key: str, subkey: str):
        if key in self.jsn and subkey in self.jsn[key]:
            return self.jsn[key][subkey]
        return None

    def set(self, key: str, value: dict):
        if key in self.jsn and isinstance(self.jsn[key], dict):
            self.jsn[key].update(value)
        else:
            self.jsn[key] = value

        par = os.path.dirname(self.file)
        if par != "":
            os.makedirs(par, exist_ok=True)
        with open(self.file, "w", encoding="utf8") as f:
            json.dump(self.jsn, f, indent=4, ensure_ascii=False)


class RedisStorage:
    def __init__(self, url, key):
        self.r: Redis = Redis.from_url(url)
        self.key = key

        if self.r.exists(key) == 0:
            self.set("$", {})

    def get(self, path: str):
        res = self.r.json().get(self.key, "$." + path)
        if len(res) == 0:
            return None
        return res[0]

    def get_array_index(self, path: str, index: int):
        return self.get(path + "[" + str(index) + "]")

    def find_array_index(self, path: str, value: str):
        return self.r.json().arrindex(self.key, path, value)

    def get_nested_key(self, path: str, subkey: str):
        return self.get(path + "." + subkey)

    def set(self, path: str, value: dict):
        self.r.json().merge(self.key, path, value)
