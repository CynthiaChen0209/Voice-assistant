from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 数据库基础配置
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/voice_secretary")

# 创建数据库引擎
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基础模型类
Base = declarative_base()

# 数据库依赖
def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 语音记录模型
class VoiceRecord(Base):
    __tablename__ = "voice_records"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False, comment="会话ID")
    original_text = Column(Text, nullable=False, comment="原始转录文本")
    translated_text = Column(Text, nullable=True, comment="翻译文本")
    source_language = Column(String, length=10, nullable=False, comment="源语言")
    target_language = Column(String, length=10, nullable=True, comment="目标语言")
    duration = Column(Float, nullable=True, comment="音频时长（秒）")
    audio_file_path = Column(String, nullable=True, comment="音频文件路径")
    is_translated = Column(Boolean, default=False, comment="是否已翻译")
    confidence_score = Column(Float, nullable=True, comment="识别置信度")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "duration": self.duration,
            "audio_file_path": self.audio_file_path,
            "is_translated": self.is_translated,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

# 会话模型
class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False, comment="会话唯一标识")
    title = Column(String, nullable=True, comment="会话标题")
    description = Column(Text, nullable=True, comment="会话描述")
    total_duration = Column(Float, default=0.0, comment="总录音时长")
    record_count = Column(Integer, default=0, comment="录音条数")
    is_active = Column(Boolean, default=True, comment="是否活跃")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "title": self.title,
            "description": self.description,
            "total_duration": self.total_duration,
            "record_count": self.record_count,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

# 系统配置模型
class SystemConfig(Base):
    __tablename__ = "system_config"
    
    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String, unique=True, index=True, nullable=False, comment="配置键")
    config_value = Column(Text, nullable=False, comment="配置值")
    description = Column(Text, nullable=True, comment="配置描述")
    is_active = Column(Boolean, default=True, comment="是否启用")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "config_key": self.config_key,
            "config_value": self.config_value,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

# 创建所有表
def create_tables():
    """创建数据库表"""
    Base.metadata.create_all(bind=engine)

# 初始化默认配置
def init_default_config():
    """初始化系统默认配置"""
    db = SessionLocal()
    try:
        # 检查是否已有配置
        existing = db.query(SystemConfig).filter(SystemConfig.config_key == "whisper_model").first()
        if not existing:
            # 默认配置
            default_configs = [
                {"config_key": "whisper_model", "config_value": "base", "description": "Whisper模型名称"},
                {"config_key": "whisper_device", "config_value": "cpu", "description": "Whisper运行设备"},
                {"config_key": "default_target_language", "config_value": "en", "description": "默认目标翻译语言"},
                {"config_key": "max_recording_duration", "config_value": "1800", "description": "最大录音时长（秒）"},
                {"config_key": "audio_sample_rate", "config_value": "16000", "description": "音频采样率"},
                {"config_key": "audio_channels", "config_value": "1", "description": "音频声道数"}
            ]
            
            for config in default_configs:
                db_config = SystemConfig(**config)
                db.add(db_config)
            
            db.commit()
            print("默认配置初始化成功")
    
    except Exception as e:
        print(f"默认配置初始化失败: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # 创建表
    create_tables()
    # 初始化配置
    init_default_config()
    print("数据库初始化完成")