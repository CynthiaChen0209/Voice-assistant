from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisper
import torch
import io
import numpy as np
import logging
from typing import Optional

router = APIRouter(prefix="/api/v1/speech", tags=["语音识别"])

# 配置日志
logger = logging.getLogger(__name__)

# 全局变量存储模型
whisper_model = None

def load_whisper_model(model_name: str = "base", device: str = "cpu"):
    """加载Whisper模型"""
    global whisper_model
    try:
        if whisper_model is None:
            logger.info(f"正在加载Whisper模型: {model_name}")
            whisper_model = whisper.load_model(model_name, device=device)
            logger.info("Whisper模型加载成功")
        return whisper_model
    except Exception as e:
        logger.error(f"Whisper模型加载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"语音识别模型加载失败: {str(e)}")

@router.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model: str = "base",
    language: Optional[str] = None
):
    """
    语音转文字接口
    支持实时转录和批量转录
    """
    try:
        # 验证文件类型
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="请上传音频文件")
        
        # 读取音频数据
        audio_data = await audio_file.read()
        
        # 加载模型
        model_instance = load_whisper_model(model)
        
        # 将音频数据转换为numpy数组
        audio_stream = io.BytesIO(audio_data)
        audio_array = whisper.load_audio(audio_stream)
        
        # 执行转录
        result = model_instance.transcribe(
            audio_array,
            language=language if language else None  # None表示自动检测
        )
        
        logger.info(f"转录成功，文本长度: {len(result['text'])}")
        
        return JSONResponse(content={
            "success": True,
            "text": result["text"].strip(),
            "language": result["language"],
            "duration": result.get("duration", 0)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"转录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")

@router.post("/stream-transcribe")
async def stream_transcribe(
    audio_chunk: UploadFile = File(...),
    session_id: str,
    model: str = "base",
    language: Optional[str] = None
):
    """
    实时语音转录接口
    支持分块处理音频流
    """
    try:
        # 验证文件类型
        if not audio_chunk.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="请上传音频文件")
        
        # 读取音频块数据
        chunk_data = await audio_chunk.read()
        
        # 加载模型
        model_instance = load_whisper_model(model)
        
        # 处理音频块
        audio_stream = io.BytesIO(chunk_data)
        audio_array = whisper.load_audio(audio_stream)
        
        # 对于实时转录，使用更小的模型参数提高速度
        result = model_instance.transcribe(
            audio_array,
            language=language,
            fp16=False,  # CPU模式下禁用fp16
            beam_size=1  # 实时转录使用贪婪搜索
        )
        
        return JSONResponse(content={
            "success": True,
            "session_id": session_id,
            "text": result["text"].strip(),
            "language": result["language"],
            "timestamp": len(chunk_data)  # 简单的时间戳
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"实时转录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"实时转录失败: {str(e)}")

@router.delete("/model")
async def unload_model():
    """卸载模型释放内存"""
    global whisper_model
    try:
        whisper_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Whisper模型已卸载")
        return {"success": True, "message": "模型已卸载"}
    except Exception as e:
        logger.error(f"模型卸载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型卸载失败: {str(e)}")