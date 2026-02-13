from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from typing import Optional
from googletrans import Translator

router = APIRouter(prefix="/api/v1/translate", tags=["翻译服务"])

# 配置日志
logger = logging.getLogger(__name__)

# 翻译请求模型
class TranslateRequest(BaseModel):
    text: str
    target_language: str = "en"
    source_language: Optional[str] = "auto"

# 翻译响应模型
class TranslateResponse(BaseModel):
    success: bool
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: Optional[float] = None

# 全局翻译器实例
translator = Translator()

def get_translator():
    """获取翻译器实例"""
    try:
        return translator
    except Exception as e:
        logger.error(f"翻译器初始化失败: {str(e)}")
        raise HTTPException(status_code=500, detail="翻译服务不可用")

@router.post("/", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """
    文本翻译接口
    支持多语言翻译，主要使用Google翻译
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="翻译文本不能为空")
        
        translator_instance = get_translator()
        
        # 执行翻译
        result = translator_instance.translate(
            request.text,
            dest=request.target_language,
            src=request.source_language
        )
        
        logger.info(f"翻译成功: {request.source_language} -> {request.target_language}")
        
        return TranslateResponse(
            success=True,
            original_text=request.text,
            translated_text=result.text,
            source_language=result.src,
            target_language=request.target_language,
            confidence=getattr(result, 'confidence', None)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"翻译失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")

@router.post("/batch")
async def batch_translate(
    texts: list[str],
    target_language: str = "en",
    source_language: Optional[str] = "auto"
):
    """
    批量翻译接口
    适用于长文本的分段翻译
    """
    try:
        if not texts:
            raise HTTPException(status_code=400, detail="翻译文本列表不能为空")
        
        translator_instance = get_translator()
        results = []
        
        for i, text in enumerate(texts):
            try:
                if not text.strip():
                    results.append({
                        "index": i,
                        "success": False,
                        "error": "文本为空"
                    })
                    continue
                
                result = translator_instance.translate(
                    text,
                    dest=target_language,
                    src=source_language
                )
                
                results.append({
                    "index": i,
                    "success": True,
                    "original_text": text,
                    "translated_text": result.text,
                    "source_language": result.src,
                    "target_language": target_language
                })
                
            except Exception as e:
                logger.error(f"第{i}段文本翻译失败: {str(e)}")
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        logger.info(f"批量翻译完成，成功: {sum(1 for r in results if r['success'])}/{len(texts)}")
        
        return {
            "success": True,
            "total": len(texts),
            "successful": sum(1 for r in results if r['success']),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量翻译失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量翻译失败: {str(e)}")

@router.get("/languages")
async def get_supported_languages():
    """获取支持的语言列表"""
    try:
        # Google Translate支持的主要语言
        languages = {
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)', 
            'en': 'English',
            'ja': 'Japanese',
            'ko': 'Korean',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
        
        return {
            "success": True,
            "languages": languages
        }
        
    except Exception as e:
        logger.error(f"获取语言列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取语言列表失败")