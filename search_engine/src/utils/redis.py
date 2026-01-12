from redis import Redis
from fastapi import FastAPI

redis: Redis | None = None


def get_redis() -> Redis:
    return redis


async def init_redis(app: FastAPI):
    global redis
    redis = Redis(
        host="localhost",
        port=6379,
        # decode_responses=True,
    )
     # Optional: check connection
    redis.ping()
    print("Connected to Redis")


async def close_redis():
    if redis:
        redis.close()
    print("Disconnected from Redis")
