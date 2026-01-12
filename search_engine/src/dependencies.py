import redis.asyncio as redis

from src.utils.redis import get_redis
from src.utils.redis_cache import BaseRedisCache, FileCache

file_cache: FileCache | None = None

def init_dependencies():
    global file_cache
    redis_client = get_redis()
    print("Redis client in dependencies:", redis_client)
    file_cache = BaseRedisCache(redis_client, "nlp-search-engine")

async def get_file_cache() -> FileCache:
    return file_cache
