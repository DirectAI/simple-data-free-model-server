import asyncio
import unittest
import redis.asyncio as redis
import warnings

from server import grab_redis_endpoint

class TestAsyncRedisConnection(unittest.IsolatedAsyncioTestCase):
    async def test_async_connection(self) -> None:
        real_redis_endpoint = grab_redis_endpoint()
        self.redis_connection = await redis.from_url(real_redis_endpoint)
        self.assertTrue(await self.redis_connection.ping())
    
    async def asyncTearDown(self) -> None:
        with warnings.catch_warnings():
            # catching the warning as mypy refuses to recognize .aclose()
            warnings.simplefilter("ignore", DeprecationWarning)
            await self.redis_connection.close()

class TestBadAsyncRedisConnection(unittest.IsolatedAsyncioTestCase):
    async def test_bad_async_connection(self) -> None:
        bad_endpoint = "bad_endpoint"
        with self.assertRaises(ValueError):
            self.redis_connection = await redis.from_url(bad_endpoint)

class TestBadKeyGrab(unittest.IsolatedAsyncioTestCase):
    async def test_bad_key_grab(self) -> None:
        real_redis_endpoint = grab_redis_endpoint()
        self.redis_connection = await redis.from_url(real_redis_endpoint)
        key_name = "key_name"
        value = await self.redis_connection.get(key_name)
        self.assertEqual(value, None)

class TestGoodWriteRead(unittest.IsolatedAsyncioTestCase):
    async def test_good_write_read(self) -> None:
        real_redis_endpoint = grab_redis_endpoint()
        self.redis_connection = await redis.from_url(real_redis_endpoint+"?decode_responses=True")
        key_name = "key_name"
        key_val = "key_val"
        await self.redis_connection.set(key_name, key_val)
        grabbed_val = await self.redis_connection.get(key_name)
        self.assertEqual(grabbed_val, key_val)