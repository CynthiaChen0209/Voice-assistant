import redis
import json
import pickle
import os
from typing import Any, Optional, Union
from datetime import timedelta
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class CacheManager:
    """缓存管理器 - 支持Redis和内存缓存"""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = 3600  # 默认1小时过期
        self.memory_cache = {}  # 内存缓存作为备选
        self.memory_cache_ttl = {}
        
        # 尝试连接Redis
        self.redis_client = None
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # 支持二进制数据
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # 测试连接
            self.redis_client.ping()
            print("Redis连接成功")
        except Exception as e:
            print(f"Redis连接失败，使用内存缓存: {str(e)}")
            self.redis_client = None
    
    def _serialize(self, data: Any) -> bytes:
        """序列化数据"""
        try:
            return pickle.dumps(data)
        except Exception:
            # 如果pickle失败，尝试JSON
            return json.dumps(data).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        """反序列化数据"""
        try:
            return pickle.loads(data)
        except Exception:
            # 如果pickle失败，尝试JSON
            return json.loads(data.decode('utf-8'))
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        try:
            ttl = ttl or self.default_ttl
            serialized_value = self._serialize(value)
            
            if self.redis_client:
                # 使用Redis
                return self.redis_client.setex(key, ttl, serialized_value)
            else:
                # 使用内存缓存
                import time
                self.memory_cache[key] = serialized_value
                self.memory_cache_ttl[key] = time.time() + ttl
                return True
                
        except Exception as e:
            logging.error(f"缓存设置失败: {str(e)}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            if self.redis_client:
                # 使用Redis
                data = self.redis_client.get(key)
                if data:
                    return self._deserialize(data)
            else:
                # 使用内存缓存
                import time
                if key in self.memory_cache:
                    # 检查是否过期
                    if time.time() < self.memory_cache_ttl.get(key, 0):
                        return self._deserialize(self.memory_cache[key])
                    else:
                        # 删除过期缓存
                        del self.memory_cache[key]
                        del self.memory_cache_ttl[key]
            
            return None
            
        except Exception as e:
            logging.error(f"缓存获取失败: {str(e)}")
            return None
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                # 使用内存缓存
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    if key in self.memory_cache_ttl:
                        del self.memory_cache_ttl[key]
                    return True
                return False
                
        except Exception as e:
            logging.error(f"缓存删除失败: {str(e)}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            if self.redis_client:
                return bool(self.redis_client.exists(key))
            else:
                import time
                if key in self.memory_cache:
                    return time.time() < self.memory_cache_ttl.get(key, 0)
                return False
                
        except Exception as e:
            logging.error(f"缓存检查失败: {str(e)}")
            return False
    
    def clear(self, pattern: str = "*") -> bool:
        """清空缓存"""
        try:
            if self.redis_client:
                # Redis模式匹配删除
                keys = self.redis_client.keys(pattern)
                if keys:
                    return bool(self.redis_client.delete(*keys))
                return True
            else:
                # 内存缓存全部清空
                self.memory_cache.clear()
                self.memory_cache_ttl.clear()
                return True
                
        except Exception as e:
            logging.error(f"缓存清空失败: {str(e)}")
            return False
    
    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    "type": "redis",
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                }
            else:
                import time
                current_time = time.time()
                active_keys = sum(1 for k, ttl in self.memory_cache_ttl.items() if current_time < ttl)
                
                return {
                    "type": "memory",
                    "total_keys": len(self.memory_cache),
                    "active_keys": active_keys,
                    "memory_usage": f"{len(pickle.dumps(self.memory_cache))} bytes"
                }
                
        except Exception as e:
            logging.error(f"缓存统计获取失败: {str(e)}")
            return {"error": str(e)}

# 全局缓存管理器实例
cache_manager = CacheManager()

# 便捷函数
def cache_transcription(session_id: str, text: str, ttl: int = 7200) -> bool:
    """缓存转录结果"""
    key = f"transcription:{session_id}"
    return cache_manager.set(key, text, ttl)

def get_cached_transcription(session_id: str) -> Optional[str]:
    """获取缓存的转录结果"""
    key = f"transcription:{session_id}"
    return cache_manager.get(key)

def cache_translation(text_hash: str, translation: str, ttl: int = 86400) -> bool:
    """缓存翻译结果"""
    key = f"translation:{text_hash}"
    return cache_manager.set(key, translation, ttl)

def get_cached_translation(text_hash: str) -> Optional[str]:
    """获取缓存的翻译结果"""
    key = f"translation:{text_hash}"
    return cache_manager.get(key)

def cache_session_data(session_id: str, data: dict, ttl: int = 3600) -> bool:
    """缓存会话数据"""
    key = f"session:{session_id}"
    return cache_manager.set(key, data, ttl)

def get_cached_session_data(session_id: str) -> Optional[dict]:
    """获取缓存的会话数据"""
    key = f"session:{session_id}"
    return cache_manager.get(key)

if __name__ == "__main__":
    # 测试缓存功能
    print("测试缓存功能...")
    
    # 测试基本操作
    cache_manager.set("test_key", {"message": "Hello World"}, ttl=60)
    result = cache_manager.get("test_key")
    print(f"缓存测试结果: {result}")
    
    # 测试转录缓存
    cache_transcription("session_001", "这是一段测试语音转录内容")
    transcription = get_cached_transcription("session_001")
    print(f"转录缓存测试: {transcription}")
    
    # 获取缓存统计
    stats = cache_manager.get_stats()
    print(f"缓存统计: {stats}")
    
    # 清理测试数据
    cache_manager.delete("test_key")
    cache_manager.delete("transcription:session_001")
    
    print("缓存测试完成")