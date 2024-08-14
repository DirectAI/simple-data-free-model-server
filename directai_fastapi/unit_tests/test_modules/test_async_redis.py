import asyncio
import time
import unittest
import redis.asyncio as redis

from server import grab_redis_endpoint

class TestAsyncRedisConnection(unittest.IsolatedAsyncioTestCase):
    async def test_async_connection(self) -> None:
        real_redis_endpoint = grab_redis_endpoint()
        self.redis_connection = await redis.from_url(real_redis_endpoint)
        self.assertTrue(await self.redis_connection.ping())
        await self.redis_connection.close()
        # await self.redis_connection.aclose() # type: ignore [attr-defined]

class TestBadAsyncRedisConnection(unittest.IsolatedAsyncioTestCase):
    async def test_bad_async_connection(self) -> None:
        bad_endpoint = "bad_endpoint"
        with self.assertRaises(ValueError):
            self.redis_connection = await redis.from_url(bad_endpoint)
            
class TestGoodDelete(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        real_redis_endpoint = grab_redis_endpoint()
        self.redis_connection = await redis.from_url(real_redis_endpoint)
    
    async def test_good_delete(self) -> None:
        key_name = "key_name"
        key_val = "key_val"
        
        await self.redis_connection.set(key_name, key_val)
        start_time = time.time()
        await self.redis_connection.delete(key_name)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.assertTrue(latency < 5)

class TestBadKeyGrab(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        real_redis_endpoint = grab_redis_endpoint()
        self.redis_connection = await redis.from_url(real_redis_endpoint)
        key_name = "key_name"
        await self.redis_connection.delete(key_name)
        
    async def test_bad_key_grab(self) -> None:
        key_name = "key_name"
        value = await self.redis_connection.get(key_name)
        self.assertEqual(value, None)

class TestGoodWriteRead(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        real_redis_endpoint = grab_redis_endpoint()
        self.redis_connection = await redis.from_url(real_redis_endpoint+"?decode_responses=True")
        key_name = "key_name"
        await self.redis_connection.delete(key_name)
        
    async def test_good_write_read(self) -> None:
        key_name = "key_name"
        key_val = "key_val"
        start_time = time.time()
        await self.redis_connection.set(key_name, key_val)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.assertTrue(latency < 5)
        start_time = time.time()
        grabbed_val = await self.redis_connection.get(key_name)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.assertTrue(latency < 5)
        self.assertEqual(grabbed_val, key_val)