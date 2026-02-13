import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class VoiceSecretaryLogger:
    """语音小秘书日志管理器"""
    
    def __init__(self):
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "logs/app.log")
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5
        
        # 确保日志目录存在
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # 配置日志格式
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 初始化日志器
        self._setup_loggers()
    
    def _setup_loggers(self):
        """设置不同模块的日志器"""
        # 应用主日志器
        self.app_logger = self._create_logger("app")
        
        # 语音识别日志器
        self.speech_logger = self._create_logger("speech")
        
        # 翻译服务日志器
        self.translate_logger = self._create_logger("translate")
        
        # 数据库日志器
        self.db_logger = self._create_logger("database")
        
        # API请求日志器
        self.api_logger = self._create_logger("api")
        
        # 错误日志器
        self.error_logger = self._create_logger("error")
    
    def _create_logger(self, name: str) -> logging.Logger:
        """创建日志器"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 添加文件处理器（轮转日志）
        file_handler = RotatingFileHandler(
            self.log_file.replace('.log', f'_{name}.log'),
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.formatter)
        logger.addHandler(file_handler)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_api_request(self, method: str, endpoint: str, user_agent: str = "", ip: str = ""):
        """记录API请求"""
        message = f"API请求 - {method} {endpoint}"
        if ip:
            message += f" | IP: {ip}"
        if user_agent:
            message += f" | UserAgent: {user_agent}"
        
        self.api_logger.info(message)
    
    def log_speech_transcription(self, session_id: str, duration: float, text_length: int, success: bool):
        """记录语音转录"""
        status = "成功" if success else "失败"
        message = f"语音转录{status} - 会话: {session_id} | 时长: {duration:.2f}s | 文本长度: {text_length}"
        
        if success:
            self.speech_logger.info(message)
        else:
            self.speech_logger.error(message)
    
    def log_translation(self, text_length: int, source_lang: str, target_lang: str, success: bool):
        """记录翻译操作"""
        status = "成功" if success else "失败"
        message = f"翻译{status} - {source_lang}->{target_lang} | 文本长度: {text_length}"
        
        if success:
            self.translate_logger.info(message)
        else:
            self.translate_logger.error(message)
    
    def log_database_operation(self, operation: str, table: str, record_id: int = None, success: bool = True):
        """记录数据库操作"""
        status = "成功" if success else "失败"
        message = f"数据库{status} - {operation} {table}"
        if record_id:
            message += f" | ID: {record_id}"
        
        if success:
            self.db_logger.info(message)
        else:
            self.db_logger.error(message)
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误"""
        message = f"错误发生"
        if context:
            message += f" | 上下文: {context}"
        message += f" | 错误: {str(error)}"
        
        self.error_logger.error(message, exc_info=True)
    
    def get_system_stats(self) -> dict:
        """获取系统统计信息"""
        try:
            # 这里可以添加更多统计信息
            stats = {
                "timestamp": datetime.now().isoformat(),
                "log_level": self.log_level,
                "log_file": self.log_file
            }
            return stats
        except Exception as e:
            self.error_logger.error(f"获取系统统计失败: {str(e)}")
            return {"error": str(e)}

# 全局日志管理器实例
logger_manager = VoiceSecretaryLogger()

# 便捷函数
def get_logger(name: str = "app") -> logging.Logger:
    """获取指定名称的日志器"""
    return getattr(logger_manager, f"{name}_logger", logger_manager.app_logger)

def log_performance(func_name: str, start_time: float, end_time: float, success: bool = True):
    """记录性能日志"""
    duration = end_time - start_time
    status = "成功" if success else "失败"
    message = f"性能统计 - {func_name}{status} | 耗时: {duration:.3f}s"
    
    if duration > 5.0:  # 超过5秒的操作记录为警告
        logger_manager.app_logger.warning(message)
    else:
        logger_manager.app_logger.info(message)

if __name__ == "__main__":
    # 测试日志功能
    app_logger = get_logger("app")
    app_logger.info("语音小秘书日志系统启动")
    
    # 测试各种日志记录
    logger_manager.log_api_request("POST", "/api/v1/speech/transcribe", "TestAgent", "127.0.0.1")
    logger_manager.log_speech_transcription("session_001", 15.5, 200, True)
    logger_manager.log_translation(200, "zh", "en", True)
    logger_manager.log_database_operation("INSERT", "voice_records", 1, True)
    
    # 测试错误日志
    try:
        raise ValueError("这是一个测试错误")
    except Exception as e:
        logger_manager.log_error(e, "测试环境")
    
    print("日志测试完成，请检查logs目录下的日志文件")