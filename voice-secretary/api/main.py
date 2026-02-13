from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
import uvicorn
import os
import time
import logging
from dotenv import load_dotenv

# 导入路由
from endpoints.speech import router as speech_router
from endpoints.translate import router as translate_router

# 导入工具模块
from utils.logger import logger_manager, get_logger
from utils.cache import cache_manager
from models.database import create_tables, init_default_config

# 加载环境变量
load_dotenv()

# 创建FastAPI应用
app = FastAPI(
    title="语音小秘书API",
    description="个人语音备忘录后端服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置日志
logger = get_logger("app")

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境需要限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录API请求日志"""
    start_time = time.time()
    
    # 记录请求
    logger_manager.log_api_request(
        method=request.method,
        endpoint=str(request.url),
        user_agent=request.headers.get("user-agent", ""),
        ip=request.client.host
    )
    
    # 处理请求
    response = await call_next(request)
    
    # 记录响应时间
    process_time = time.time() - start_time
    logger.info(f"请求处理完成 - {request.method} {request.url} - 耗时: {process_time:.3f}s")
    
    return response

# 注册路由
app.include_router(speech_router)
app.include_router(translate_router)

@app.get("/")
async def root():
    """健康检查接口"""
    return {"status": "ok", "message": "语音小秘书API运行正常"}

@app.get("/health")
async def health_check():
    """详细健康检查"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "whisper": "待检查",
            "translator": "待检查", 
            "database": "待检查",
            "redis": "待检查"
        }
    }

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("语音小秘书API正在启动...")
    
    # 初始化数据库
    try:
        create_tables()
        init_default_config()
        logger.info("数据库初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
    
    # 检查缓存连接
    cache_stats = cache_manager.get_stats()
    logger.info(f"缓存状态: {cache_stats}")
    
    logger.info("语音小秘书API启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("语音小秘书API正在关闭...")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )