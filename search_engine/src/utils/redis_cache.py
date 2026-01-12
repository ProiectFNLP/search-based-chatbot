from redis import Redis
import pickle
from typing import Optional, TypeVar
from fastapi import UploadFile
import hashlib

T = TypeVar("T")


class BaseRedisCache:
    class SubCache:
        def __init__(self, parent: "BaseRedisCache | BaseRedisCache.SubCache", key: str):
            self.parent = parent
            self.key = key

        def subcache(self, key: str):
            return BaseRedisCache.SubCache(self, key)

        def _key(self, item: Optional[str] = None) -> str:
            return self.key if item is None else f"{self.key}:{item}"

        def get(self, item: str) -> Optional[str]:
            return self.parent.get(self._key(item))

        def set(self, item: str, value: str, ttl: int | None = None):
            return self.parent.set(self._key(item), value, ttl)

        def delete(self, item: str):
            return self.parent.delete(self._key(item))

        def exists(self, item: str) -> bool:
            return self.parent.exists(self._key(item))

        def get_pickled(self, item: str) -> Optional[T]:
            return self.parent.get_pickled(self._key(item))

        def set_pickled(self, item: str, value: object, ttl: int | None = None):
            return self.parent.set_pickled(self._key(item), value, ttl)
        
        def subkeys(self, key: Optional[str] = None) -> list[str]:
            return self.parent.subkeys(self._key(key))
        
        def __getitem__(self, item: str) -> Optional[str]:
            return self.parent[self._key(item)]

        def __setitem__(self, item: str, value: str) -> None:
            self.parent[self._key(item)] = value


    def __init__(self, redis: Redis, prefix: str):
        self.redis = redis
        self.prefix = prefix

    def _key(self, item: str) -> str:
        return f"{self.prefix}:{item}"

    def subcache(self, key: str):
        return BaseRedisCache.SubCache(self, self._key(key))

    def get(self, item: str) -> Optional[str]:
        return  self.redis.get(self._key(item))

    def set(self, item: str, value: str, ttl: int | None = None):
        key = self._key(item)
        if ttl:
            self.redis.setex(key, ttl, value)
        else:
            self.redis.set(key, value)
    
    def delete(self, item: str):
        self.redis.delete(self._key(item))

    def exists(self, item: str) -> bool:
        return  self.redis.exists(self._key(item)) == 1

    def get_pickled(self, item: str) -> Optional[T]:
        key = self._key(item)
        data = self.redis.get(key)
        return pickle.loads(data) if data else None

    def set_pickled(self, item: str, value: object, ttl: int | None = None):
        data = pickle.dumps(value)
        key = self._key(item)
        if ttl:
            self.redis.setex(key, ttl, data)
        else:
            self.redis.set(key, data)

    def subkeys(self, key: Optional[str] = None) -> list[str]:
        prefix = self._key(key) if key else self.prefix
        keys = []

        for k in self.redis.scan_iter(match=f"{prefix}:*"):
            k = k.decode('utf-8')
            keys.append(k.split(":")[-1])

        return keys
    
    def __getitem__(self, item: str) -> Optional[str]:
        return self.get(item)

    def __setitem__(self, item: str, value: str) -> None:
        self.set(item, value)

FileCache = BaseRedisCache | BaseRedisCache.SubCache


async def make_hash(file: UploadFile, reuse: bool = True, chunk_size: int = 8192) -> str:
    hasher = hashlib.sha256()
    file_size = 0

    while True:
        data = await file.read(chunk_size)
        if not data:
            break
        hasher.update(data)
        file_size += len(data)

    hasher.update(str(file_size).encode())

    if reuse:
        await file.seek(0)

    return hasher.hexdigest()