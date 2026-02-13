#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音小秘书云服务 - 使用minimax API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
import requests
import json
import tempfile
import wave
import numpy as np
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="语音小秘书API - MiniMax版本",
    description="基于MiniMax API的语音转文字服务",
    version="1.0.0"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MiniMax API配置
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"

@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "message": "语音小秘书云服务运行正常"}

@app.get("/health")
async def health_check():
    """详细健康检查"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "minimax_api": "待检查",
            "audio_processing": "正常"
        },
        "timestamp": datetime.now().isoformat()
    }

def convert_audio_to_wav(audio_data: bytes, sample_rate: int = 16000) -> bytes:
    """将音频数据转换为WAV格式"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sample_rate)
            
            # 确保音频数据是正确的格式
            if len(audio_data) % 2 != 0:
                audio_data = audio_data[:-1]  # 确保偶数长度
            
            wav_file.writeframes(audio_data)
        
        # 读取WAV文件
        with open(temp_file.name, 'rb') as f:
            wav_data = f.read()
        
        os.unlink(temp_file.name)
        return wav_data

@app.post("/api/v1/speech/transcribe")
async def transcribe_audio_minimax(audio_file: UploadFile = File(...)):
    """
    使用MiniMax API进行语音转文字
    """
    try:
        # 验证文件类型
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="请上传音频文件")
        
        # 读取音频数据
        audio_data = await audio_file.read()
        
        # 转换为WAV格式
        wav_data = convert_audio_to_wav(audio_data)
        
        # 准备MiniMax API请求
        url = f"{MINIMAX_BASE_URL}/audio/transcriptions"
        
        headers = {
            "Authorization": f"Bearer {MINIMAX_API_KEY}",
            "Content-Type": "multipart/form-data"
        }
        
        files = {
            'file': ('audio.wav', wav_data, 'audio/wav'),
            'model': (None, 'speech-01'),
            'language': (None, 'zh'),
            'response_format': (None, 'json')
        }
        
        logger.info("调用MiniMax API进行语音识别...")
        
        # 调用MiniMax API
        response = requests.post(url, headers=headers, files=files, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"MiniMax API调用失败: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=500, 
                detail=f"MiniMax API调用失败: {response.text}"
            )
        
        result = response.json()
        
        # 解析结果
        if 'text' in result:
            transcribed_text = result['text'].strip()
            logger.info(f"转录成功: {transcribed_text[:50]}...")
            
            return JSONResponse(content={
                "success": True,
                "text": transcribed_text,
                "language": "zh",
                "provider": "minimax",
                "timestamp": datetime.now().isoformat()
            })
        else:
            logger.error(f"MiniMax API返回异常: {result}")
            raise HTTPException(status_code=500, detail="转录服务返回异常")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"转录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")

@app.post("/api/v1/translate")
async def translate_text(request: dict):
    """
    翻译文本 - 使用简单的免费翻译服务
    """
    try:
        text = request.get("text", "")
        target_language = request.get("target_language", "en")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="翻译文本不能为空")
        
        # 使用免费翻译API
        import urllib.parse
        text_encoded = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=zh-CN&tl={target_language}&dt=t&q={text_encoded}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0 and result[0]:
                translated_text = ''.join([item[0] for item in result[0] if item[0]])
                return JSONResponse(content={
                    "success": True,
                    "original_text": text,
                    "translated_text": translated_text,
                    "target_language": target_language,
                    "provider": "google"
                })
            else:
                raise HTTPException(status_code=500, detail="翻译解析失败")
        else:
            raise HTTPException(status_code=500, detail="翻译请求失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"翻译失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")

@app.get("/api/v1/test/minimax")
async def test_minimax_api():
    """测试MiniMax API连接"""
    try:
        if not MINIMAX_API_KEY:
            return JSONResponse(content={
                "success": False,
                "error": "MiniMax API密钥未配置"
            })
        
        # 测试API密钥
        url = f"{MINIMAX_BASE_URL}/models"
        headers = {
            "Authorization": f"Bearer {MINIMAX_API_KEY}"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return JSONResponse(content={
                "success": True,
                "message": "MiniMax API连接正常",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return JSONResponse(content={
                "success": False,
                "error": f"API连接失败: {response.status_code}"
            })
            
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": f"连接测试失败: {str(e)}"
        })

if __name__ == "__main__":
    # 检查环境变量
    if not MINIMAX_API_KEY:
        logger.warning("MINIMAX_API_KEY环境变量未设置")
        logger.info("请设置环境变量: export MINIMAX_API_KEY=your_api_key")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )