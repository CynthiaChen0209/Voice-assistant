#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音小秘书 - 简体中文最终版本
嘉庚蓝设计风格，简体中文界面
"""

import sys
import queue
import time
import logging
import json
from datetime import datetime
from typing import Optional

# PyQt5相关
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QTextEdit, 
                            QLabel, QStatusBar, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPainter, QPen, QBrush

# 音频处理
import pyaudio
import numpy as np
import whisper

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 嘉庚蓝色调定义
JIAKENG_BLUE = "#1E50A2"       # 嘉庚蓝主色
JIAKENG_LIGHT_BLUE = "#4A90E2"  # 嘉庚浅蓝
JIAKENG_DARK_BLUE = "#0F3460"   # 嘉庚深蓝
JIAKENG_GRAY = "#F5F7FA"        # 嘉庚灰白
JIAKENG_TEXT = "#333333"        # 文字颜色
JIAKENG_BORDER = "#E1E4E8"      # 边框颜色

class AudioRecorder:
    """音频录制器"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        
        self.pa = None
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.start_time = None
        
    def init_audio(self):
        """初始化音频设备"""
        if self.pa is None:
            self.pa = pyaudio.PyAudio()
    
    def start_recording(self):
        """开始录音"""
        try:
            if self.is_recording:
                return False
            
            self.init_audio()
            self.audio_queue.queue.clear()
            self.start_time = time.time()
            
            self.stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            logger.info("开始录音")
            return True
            
        except Exception as e:
            logger.error(f"开始录音失败: {str(e)}")
            return False
    
    def stop_recording(self):
        """停止录音"""
        try:
            if not self.is_recording:
                return False
            
            self.is_recording = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            duration = time.time() - self.start_time if self.start_time else 0
            logger.info(f"录音停止，时长: {duration:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"停止录音失败: {str(e)}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调"""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def get_audio_data(self) -> Optional[np.ndarray]:
        """获取所有录音数据"""
        try:
            audio_chunks = []
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                audio_array = np.frombuffer(chunk, dtype=np.int16)
                audio_chunks.append(audio_array)
            
            if audio_chunks:
                return np.concatenate(audio_chunks)
            return None
            
        except Exception as e:
            logger.error(f"获取音频数据失败: {str(e)}")
            return None
    
    def cleanup(self):
        """清理资源"""
        if self.is_recording:
            self.stop_recording()
        if self.pa:
            self.pa.terminate()

class ChineseTranscriptionThread(QThread):
    """中文转录线程"""
    transcription_ready = pyqtSignal(str, str)
    
    def __init__(self, audio_data: np.ndarray, translate: bool = False):
        super().__init__()
        self.audio_data = audio_data
        self.translate = translate
        self.model = None
        
    def load_model(self):
        """加载模型"""
        if self.model is None:
            logger.info("加载Whisper中文优化模型 (medium)...")
            # 使用medium模型，对中文识别准确率更高
            # small: 75MB, medium: 142MB, large: 155MB
            try:
                self.model = whisper.load_model("medium", device="cpu")
                logger.info("Medium模型加载完成 - 高准确率模式")
            except Exception as e:
                logger.warning(f"Medium模型加载失败，降级到small模型: {str(e)}")
                # 降级到small模型作为备选
                self.model = whisper.load_model("small", device="cpu")
                logger.info("Small模型加载完成 - 标准准确率模式")
    
    def clean_text(self, text: str) -> str:
        """清理语气词和停顿词，修复中文标点符号"""
        import re
        
        # 常见语气词列表
        filler_words = [
            # 单字语气词
            '嗯', '啊', '哦', '唉', '咦', '哟', '嘿', '嗯哼',
            '那个', '这个', '就是', '然后', '还有', '或者',
            '吼', '哈', '嘿咻', '呃', '唔', '嘛', '哦哦',
            # 停顿词
            '嗯嗯', '啊啊', '哦哦', '呃呃', '唔唔', '嘛嘛',
            # 连接词
            '然后呢', '还有就是', '就是说', '也就是说',
            # 思考词
            '让我想想', '我想想', '这个嘛', '那个嘛',
            # 常见口头禅
            '对吧', '是吧', '对吧', '对不对', '是不是',
            # 犹豫词
            '怎么说呢', '怎么说', '大概', '可能', '也许',
            '好像', '似乎', '差不多', '基本上'
        ]
        
        # 清理文本
        cleaned_text = text.strip()
        
        # 第一步：移除错误的英文内容（中文识别不应该包含英文单词）
        cleaned_text = re.sub(r'[a-zA-Z]+', '', cleaned_text)
        
        # 第二步：标准化中文标点符号（符合中文文法习惯）
        # 英文标点转中文标点
        punctuation_map = {
            '.': '。',      # 英文句号 → 中文句号
            ',': '，',      # 英文逗号 → 中文逗号
            '?': '？',      # 英文问号 → 中文问号
            '!': '！',      # 英文感叹号 → 中文感叹号
            ':': '：',      # 英文冒号 → 中文冒号
            ';': '；',      # 英文分号 → 中文分号
            '"': '"',      # 英文双引号 → 中文双引号
            "'": "'",      # 英文单引号 → 中文单引号
            '(': '（',      # 英文左括号 → 中文左括号
            ')': '）',      # 英文右括号 → 中文右括号
        }
        
        for en_punc, zh_punc in punctuation_map.items():
            cleaned_text = cleaned_text.replace(en_punc, zh_punc)
        
        # 第三步：修正中文标点符号使用规范
        # 顿号（、）用于并列词语，逗号（，）用于句子分隔
        # 智能判断：如果是单个汉字的并列，使用顿号
        cleaned_text = re.sub(r'([\u4e00-\u9fff])、([\u4e00-\u9fff])', r'\1、\2', cleaned_text)
        
        # 修正错误使用顿号的情况，改为逗号
        cleaned_text = re.sub(r'，、', '，', cleaned_text)  # 重复标点
        cleaned_text = re.sub(r'、，', '，', cleaned_text)  # 重复标点
        cleaned_text = re.sub(r'、(?!\s*[\u4e00-\u9fff])', '，', cleaned_text)  # 后面不是汉字用逗号
        
        # 第二步：移除语气词
        for filler in filler_words:
            pattern = r'\\s*' + re.escape(filler) + r'\\s*'
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 第三步：清理多余空格（中文通常不使用空格）
        cleaned_text = re.sub(r'\\s+', '', cleaned_text)
        
        # 第四步：修复重复标点符号
        duplicate_puncs = [
            (r'[，，]{2,}', '，'),      # 多个逗号
            (r'[、、]{2,}', '、'),      # 多个顿号
            (r'[。。]{2,}', '。'),      # 多个句号
            (r'[！！]{2,}', '！'),      # 多个感叹号
            (r'[？？]{2,}', '？'),      # 多个问号
            (r'[；；]{2,}', '；'),      # 多个分号
            (r'[：：]{2,}', '：'),      # 多个冒号
            (r'[""]{2,}', '"'),       # 多个双引号
            (r"[']{2,}", "'"),       # 多个单引号
            (r'[（(）（)]{2,}', '（')   # 多个括号
        ]
        
        for pattern, replacement in duplicate_puncs:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # 第五步：移除句首无意义的连接词
        start_words = ['然后', '还有', '就是', '那个', '这个', '嗯', '啊', '哦']
        while any(cleaned_text.startswith(word) for word in start_words):
            for word in start_words:
                if cleaned_text.startswith(word):
                    cleaned_text = cleaned_text[len(word):].strip()
                    break
        
        # 第六步：确保合适的结尾标点（中文文法习惯）
        if cleaned_text:
            # 中文结尾标点优先级：？> ！> 。> 、> ，> ；> ：
            # 注意：顿号（、）通常不用作句子结尾
            if cleaned_text.endswith(('。', '！', '？', '、', '，', '；', '：')):
                pass  # 已有合适标点（顿号结尾较少见但允许）
            elif any(word in cleaned_text for word in ['吗', '呢', '吧', '么']):
                cleaned_text += '？'  # 疑问语气
            elif any(word in cleaned_text for word in ['啊', '呀', '啦']):
                cleaned_text += '！'  # 感叹语气
            else:
                cleaned_text += '。'  # 默认陈述语气
        
        # 第七步：特殊中文表达优化
        chinese_fixes = {
            '过程当中': '过程中',
            '这里边': '这里边',
            '那么': '那么',  # 保留必要的连词
            '因为所以': '因为，所以',
        }
        
        for wrong, correct in chinese_fixes.items():
            cleaned_text = cleaned_text.replace(wrong, correct)
        
        logger.info(f"原文: {text}")
        logger.info(f"清理后: {cleaned_text}")
        
        return cleaned_text
    
    def run(self):
        """执行转录"""
        try:
            self.load_model()
            
            # 音频预处理 - Ultra优化版本
            if self.audio_data.dtype != np.float32:
                audio_float = self.audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = self.audio_data.copy()
            
            # 1. 基础归一化
            if np.max(np.abs(audio_float)) > 0:
                audio_float = audio_float / np.max(np.abs(audio_float))
            
            # 2. 智能音量优化 - 针对语音特性
            # 计算语音活跃度
            speech_energy = np.sqrt(np.mean(audio_float**2))
            if speech_energy < 0.1:  # 音量太小
                gain = 1.2  # 增益
            elif speech_energy > 0.8:  # 音量太大
                gain = 0.9  # 衰减
            else:
                gain = 1.0  # 保持
            
            audio_float = audio_float * gain
            
            # 3. 轻微的动态范围压缩 (提升语音清晰度)
            audio_float = np.tanh(audio_float * 0.95) * 0.95
            
            # 4. 安全限幅
            audio_float = np.clip(audio_float, -1.0, 1.0)
            
            logger.info(f"音频预处理完成 - 语音能量: {speech_energy:.3f}, 增益: {gain:.2f}")
            
            # 转录 - Ultra优化中文识别，最大化准确性
            try:
                # 使用Ultra最佳参数组合 (v1.3优化)
                result = self.model.transcribe(
                    audio_float, 
                    language="zh",  # 明确指定中文
                    task="transcribe",
                    # Ultra优化的提示语，包含专业场景
                    initial_prompt="请准确转录以下中文普通话录音。这是办公场景的语音记录，包含正式的商务用语和专业词汇。",  
                    fp16=False,  # 禁用FP16提高准确性
                    temperature=0.0,  # 完全确定性，无随机性
                    beam_size=10,  # 增大beam size提高准确性
                    best_of=10,  # 生成更多候选结果
                    patience=2.0,  # 增加耐心度提高准确性
                    condition_on_previous_text=False,  # 不依赖前文，提高独立性
                    # 新增优化参数
                    length_penalty=1.0,  # 长度惩罚因子
                    suppress_tokens=[],  # 不抑制特殊标记
                    prepend_punctuations="\"'¿([{-",  # 标点符号前置处理
                    append_punctuations="\"'.。,，!！?？:：\"",  # 标点符号后置处理
                    compression_ratio_threshold=2.4,  # 压缩比阈值
                    logprob_threshold=-1.0,  # 对数概率阈值
                    no_speech_threshold=0.6  # 语音检测阈值
                )
            except TypeError as e:
                if any(param in str(e) for param in ["compression_ratio_threshold", "logprob_threshold", "no_speech_threshold", "prompt_reset_on_temperature"]):
                    logger.warning("使用兼容模式转录（去除不支持的参数）")
                    # 使用兼容参数，保持高准确性
                    result = self.model.transcribe(
                        audio_float, 
                        language="zh",
                        task="transcribe",
                        initial_prompt="请准确转录以下中文普通话录音。办公场景语音记录，包含商务用语和专业词汇。",
                        fp16=False,
                        temperature=0.0,  # 保持无随机性
                        beam_size=8,  # 增大beam size提高准确性
                        best_of=8,  # 生成更多候选结果
                        condition_on_previous_text=False  # 提高独立性
                    )
                else:
                    raise e
            
            original_text = result["text"].strip()
            
            # 清理语气词
            cleaned_text = self.clean_text(original_text)
            
            # 如果清理后内容太短，使用原文
            if len(cleaned_text) < 3:
                cleaned_text = original_text
                logger.warning("清理后文本过短，使用原文")
            
            # 翻译 - 使用清理后的文本
            translated_text = ""
            if self.translate and cleaned_text:
                try:
                    import requests
                    import urllib.parse
                    
                    text_encoded = urllib.parse.quote(cleaned_text)
                    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=zh-CN&tl=en&dt=t&q={text_encoded}"
                    
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        if result and len(result) > 0 and result[0]:
                            translated_text = ''.join([item[0] for item in result[0] if item[0]])
                        else:
                            translated_text = "翻译解析失败"
                    else:
                        translated_text = "翻译请求失败"
                        
                except Exception as e:
                    logger.warning(f"翻译失败: {str(e)}")
                    translated_text = f"[翻译服务暂时不可用]"
            
            # 修改转录流程，只处理中文，不自动翻译
            self.transcription_ready.emit(cleaned_text, "")
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            self.transcription_ready.emit(f"[转录失败: {str(e)}]", "")

class TranslateThread(QThread):
    """翻译线程"""
    translation_ready = pyqtSignal(str)
    
    def __init__(self, text: str):
        super().__init__()
        self.text = text
    
    def run(self):
        """执行翻译"""
        try:
            if not self.text:
                return
            
            logger.info(f"开始翻译: {self.text}")
            
            import requests
            import urllib.parse
            
            text_encoded = urllib.parse.quote(self.text)
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=zh-CN&tl=en&dt=t&q={text_encoded}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0 and result[0]:
                    translated_text = ''.join([item[0] for item in result[0] if item[0]])
                    logger.info(f"翻译成功: {translated_text}")
                    self.translation_ready.emit(translated_text)
                else:
                    self.translation_ready.emit("翻译解析失败")
            else:
                self.translation_ready.emit("翻译请求失败")
                
        except Exception as e:
            logger.error(f"翻译失败: {str(e)}")
            self.translation_ready.emit(f"[翻译服务暂时不可用]")

class MicrophoneButton(QPushButton):
    """圆形麦克风按钮"""
    
    def __init__(self, is_recording=False):
        super().__init__()
        self.is_recording = is_recording
        self.setFixedSize(60, 60)
        self.setText("")
        self.setStyleSheet("""
            QPushButton {
                background-color: #1E50A2;
                border: none;
                border-radius: 30px;
            }
            QPushButton:hover {
                background-color: #4A90E2;
            }
            QPushButton:pressed {
                background-color: #0F3460;
            }
        """)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制圆形背景
        rect = self.rect()
        if self.is_recording:
            color = QColor("#FF4444")  # 录音时红色
        else:
            color = QColor(JIAKENG_BLUE)  # 正常时嘉庚蓝
            
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(rect.adjusted(2, 2, -2, -2))
        
        # 绘制精美的麦克风图标
        center_x = rect.width() // 2
        center_y = rect.height() // 2
        
        # 设置画笔和画刷
        painter.setPen(QPen(QColor("white"), 2, Qt.SolidLine, Qt.RoundCap))
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 麦克风主体（更精致的椭圆形）
        mic_top = QRect(center_x - 7, center_y - 14, 14, 20)
        painter.setBrush(QBrush(QColor("white")))
        painter.drawEllipse(mic_top)
        
        # 麦克风网格纹理（增加细节）
        grid_color = QColor("#FF4444") if self.is_recording else QColor(JIAKENG_BLUE)
        painter.setPen(QPen(grid_color, 1))
        painter.drawLine(center_x - 4, center_y - 10, center_x + 4, center_y - 10)
        painter.drawLine(center_x - 4, center_y - 6, center_x + 4, center_y - 6)
        painter.drawLine(center_x - 4, center_y - 2, center_x + 4, center_y - 2)
        painter.drawLine(center_x - 4, center_y + 2, center_x + 4, center_y + 2)
        
        # 麦克风底部支架
        painter.setPen(QPen(QColor("white"), 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(center_x - 10, center_y + 6, center_x + 10, center_y + 6)
        
        # 麦克风连接杆
        painter.drawLine(center_x, center_y + 6, center_x, center_y + 14)
        
        # 麦克风底座（更宽的基座）
        painter.drawLine(center_x - 12, center_y + 14, center_x + 12, center_y + 14)
        
        # 录音状态指示器（录音时显示声波）
        if self.is_recording:
            painter.setPen(QPen(QColor("#FFB74D"), 1))
            # 左声波
            painter.drawLine(center_x - 18, center_y - 4, center_x - 18, center_y + 4)
            painter.drawLine(center_x - 22, center_y - 7, center_x - 22, center_y + 7)
            # 右声波
            painter.drawLine(center_x + 18, center_y - 4, center_x + 18, center_y + 4)
            painter.drawLine(center_x + 22, center_y - 7, center_x + 22, center_y + 7)



class StyledTextEdit(QTextEdit):
    """嘉庚蓝风格文本框"""
    
    def __init__(self, placeholder=""):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: white;
                border: 2px solid {JIAKENG_BORDER};
                border-radius: 8px;
                padding: 12px;
                font-family: "宋体", "SimSun", serif;
                font-size: 14px;
                color: {JIAKENG_TEXT};
                line-height: 1.6;
            }}
            QTextEdit:focus {{
                border: 2px solid {JIAKENG_LIGHT_BLUE};
            }}
        """)

class StyledGroupBox(QGroupBox):
    """嘉庚蓝风格分组框"""
    
    def __init__(self, title):
        super().__init__(title)
        self.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                color: {JIAKENG_BLUE};
                border: 2px solid {JIAKENG_BORDER};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {JIAKENG_GRAY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)

class VoiceSecretaryGUI(QMainWindow):
    """语音小秘书主界面 - 简体中文版"""
    
    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.transcription_thread = None
        self.current_session = []
        
        self.init_ui()
        self.init_timers()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("语音小秘")
        self.setGeometry(100, 100, 400, 600)
        
        # 设置应用样式
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {JIAKENG_GRAY};
            }}
            QLabel {{
                color: {JIAKENG_TEXT};
                font-family: "微软雅黑", "Microsoft YaHei", sans-serif;
            }}
            QCheckBox {{
                color: {JIAKENG_TEXT};
                font-family: "微软雅黑", "Microsoft YaHei", sans-serif;
                font-size: 13px;
            }}
        """)
        
        # 主部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        central_widget.setLayout(layout)
        
        # 顶部标题区域
        title_label = QLabel("🎙️ 语音小秘")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 20px;
                font-weight: bold;
                color: {JIAKENG_BLUE};
                padding: 10px;
                background-color: white;
                border-radius: 10px;
                border: 2px solid {JIAKENG_LIGHT_BLUE};
            }}
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 麦克风控制区域（视觉统一）
        mic_control_widget = QWidget()
        mic_control_layout = QVBoxLayout(mic_control_widget)
        mic_control_layout.setSpacing(8)
        mic_control_layout.setAlignment(Qt.AlignCenter)
        
        # 麦克风按钮容器
        mic_container = QWidget()
        mic_container_layout = QHBoxLayout(mic_container)
        mic_container_layout.setSpacing(10)
        
        # 麦克风按钮
        self.record_button = MicrophoneButton()
        self.record_button.clicked.connect(self.toggle_recording)
        
        # 录音状态文本（紧贴麦克风按钮）
        self.record_status_label = QLabel("点击开始录音")
        self.record_status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                color: {JIAKENG_BLUE};
                font-weight: bold;
                padding: 8px 12px;
                background-color: white;
                border-radius: 15px;
                border: 1px solid {JIAKENG_BORDER};
                min-width: 120px;
            }}
        """)
        self.record_status_label.setAlignment(Qt.AlignCenter)
        
        mic_container_layout.addStretch()
        mic_container_layout.addWidget(self.record_button, 0, Qt.AlignCenter)
        mic_container_layout.addWidget(self.record_status_label, 0, Qt.AlignVCenter)
        mic_container_layout.addStretch()
        
        mic_control_layout.addWidget(mic_container)
        layout.addWidget(mic_control_widget)
        
        # 简约文本区域
        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        text_layout.setSpacing(12)
        
        # 中文区域
        chinese_section = QWidget()
        chinese_section.setStyleSheet(f"""
            QWidget {{
                background-color: white;
                border-radius: 12px;
                border: 1px solid {JIAKENG_BORDER};
                padding: 12px;
            }}
        """)
        chinese_layout = QVBoxLayout(chinese_section)
        chinese_layout.setSpacing(8)
        
        # 中文标题行（标题 + 清空按钮）
        chinese_header = QWidget()
        chinese_header_layout = QHBoxLayout(chinese_header)
        chinese_header_layout.setSpacing(8)
        
        chinese_label = QLabel("中文转录")
        chinese_label.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                font-weight: 600;
                color: {JIAKENG_BLUE};
            }}
        """)
        
        # 优雅的清空按钮
        self.clear_button = QPushButton("全部清空")
        self.clear_button.setFixedSize(80, 36)
        self.clear_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #FFF8F8;
                color: #666;
                border: 1px solid #E0E0E0;
                border-radius: 18px;
                font-size: 14px;
                font-weight: 600;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: #FF5252;
                color: white;
                border-color: #FF1744;
            }}
            QPushButton:pressed {{
                background-color: #D32F2F;
                border-color: #B71C1C;
            }}
        """)
        self.clear_button.clicked.connect(self.clear_text)
        
        chinese_header_layout.addWidget(chinese_label)
        chinese_header_layout.addStretch()
        chinese_header_layout.addWidget(self.clear_button)
        chinese_layout.addWidget(chinese_header)
        
        # 中文文本框（全宽度）
        self.chinese_text = QTextEdit()
        self.chinese_text.setPlaceholderText("转录的中文内容将在这里显示...")
        self.chinese_text.setMinimumHeight(110)
        self.chinese_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {JIAKENG_GRAY};
                border: 1px solid {JIAKENG_BORDER};
                border-radius: 8px;
                padding: 12px;
                font-family: "微软雅黑", "Microsoft YaHei", sans-serif;
                font-size: 13px;
                color: {JIAKENG_TEXT};
                line-height: 1.6;
            }}
            QTextEdit:focus {{
                background-color: white;
                border: 1px solid {JIAKENG_LIGHT_BLUE};
            }}
        """)
        self.chinese_text.setReadOnly(False)
        chinese_layout.addWidget(self.chinese_text)
        text_layout.addWidget(chinese_section)
        
        # 英文区域
        english_section = QWidget()
        english_section.setStyleSheet(f"""
            QWidget {{
                background-color: white;
                border-radius: 12px;
                border: 1px solid {JIAKENG_BORDER};
                padding: 12px;
            }}
        """)
        english_layout = QVBoxLayout(english_section)
        english_layout.setSpacing(8)
        
        # 英文标题行（标题 + 翻译按钮）
        english_header = QWidget()
        english_header_layout = QHBoxLayout(english_header)
        english_header_layout.setSpacing(8)
        
        english_label = QLabel("英文翻译")
        english_label.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                font-weight: 600;
                color: {JIAKENG_BLUE};
            }}
        """)
        
        # 优雅的翻译按钮
        self.translate_button = QPushButton("全文翻译")
        self.translate_button.setFixedSize(100, 36)
        self.translate_button.setToolTip("翻译中文文本框中的所有内容")
        self.translate_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #F8FBFF;
                color: #666;
                border: 1px solid {JIAKENG_BORDER};
                border-radius: 18px;
                font-size: 14px;
                font-weight: 600;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: {JIAKENG_LIGHT_BLUE};
                color: white;
                border-color: {JIAKENG_BLUE};
            }}
            QPushButton:pressed {{
                background-color: {JIAKENG_BLUE};
                border-color: {JIAKENG_DARK_BLUE};
            }}
        """)
        self.translate_button.clicked.connect(self.manual_translate)
        
        english_header_layout.addWidget(english_label)
        english_header_layout.addStretch()
        english_header_layout.addWidget(self.translate_button)
        english_layout.addWidget(english_header)
        
        # 英文文本框（全宽度）
        self.english_text = QTextEdit()
        self.english_text.setPlaceholderText("点击'全文翻译'按钮获取所有中文内容的英文翻译...")
        self.english_text.setMinimumHeight(110)
        self.english_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {JIAKENG_GRAY};
                border: 1px solid {JIAKENG_BORDER};
                border-radius: 8px;
                padding: 12px;
                font-family: "微软雅黑", "Microsoft YaHei", sans-serif;
                font-size: 13px;
                color: {JIAKENG_TEXT};
                line-height: 1.6;
            }}
            QTextEdit:focus {{
                background-color: white;
                border: 1px solid {JIAKENG_LIGHT_BLUE};
            }}
        """)
        self.english_text.setReadOnly(True)
        english_layout.addWidget(self.english_text)
        text_layout.addWidget(english_section)
        
        layout.addWidget(text_widget)
        
        # 状态区域（紧凑设计）
        status_widget = QWidget()
        status_widget.setStyleSheet(f"""
            QWidget {{
                background-color: white;
                border-radius: 10px;
                border: 1px solid {JIAKENG_BORDER};
                padding: 8px;
            }}
        """)
        status_layout = QHBoxLayout(status_widget)
        status_layout.setSpacing(8)
        
        # 状态圆点
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"""
            QLabel {{
                font-size: 16px;
                color: #4CAF50;
                font-weight: bold;
            }}
        """)
        
        # 状态文本
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 13px;
                color: {JIAKENG_BLUE};
                font-weight: bold;
            }}
        """)
        
        status_layout.addWidget(self.status_dot)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        layout.addWidget(status_widget)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {JIAKENG_BLUE};
                color: white;
                font-size: 12px;
                border-radius: 0px 0px 10px 10px;
                padding: 5px;
            }}
        """)
        self.status_bar.showMessage("🎙️ 语音小秘就绪")
        
    def init_timers(self):
        """初始化定时器"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(100)
    
    def toggle_recording(self):
        """切换录音状态"""
        if not self.recorder.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """开始录音"""
        if self.recorder.start_recording():
            # 更新麦克风按钮为录音状态
            self.record_button.is_recording = True
            self.record_button.update()
            
            # 更新状态标签
            self.record_status_label.setText("正在录音...")
            self.record_status_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 16px;
                    color: #FF4444;
                    font-weight: bold;
                    padding: 8px 16px;
                    background-color: white;
                    border-radius: 20px;
                    border: 2px solid #FF4444;
                }}
            """)
            
            # 更新状态指示器
            self.status_label.setText("状态: 正在录音... 请说中文普通话")
            self.status_dot.setStyleSheet(f"""
                QLabel {{
                    font-size: 20px;
                    color: #FF4444;
                    font-weight: bold;
                }}
            """)
            self.status_bar.showMessage("🔴 正在录音...")
            logger.info("用户开始录音")
        else:
            QMessageBox.warning(self, "错误", "无法开始录音，请检查麦克风设备")
    
    def stop_recording(self):
        """停止录音"""
        if self.recorder.stop_recording():
            # 更新麦克风按钮为正常状态
            self.record_button.is_recording = False
            self.record_button.update()
            
            # 更新状态标签
            self.record_status_label.setText("点击开始录音")
            self.record_status_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 16px;
                    color: {JIAKENG_BLUE};
                    font-weight: bold;
                    padding: 8px 16px;
                    background-color: white;
                    border-radius: 20px;
                    border: 1px solid {JIAKENG_BORDER};
                }}
            """)
            
            # 更新状态指示器
            self.status_label.setText("状态: 正在转录中文...")
            self.status_dot.setStyleSheet(f"""
                QLabel {{
                    font-size: 20px;
                    color: #FF9800;
                    font-weight: bold;
                }}
            """)
            self.status_bar.showMessage("⚙️ 正在转录...")
            
            # 获取音频数据
            audio_data = self.recorder.get_audio_data()
            if audio_data is not None and len(audio_data) > 0:
                self.transcription_thread = ChineseTranscriptionThread(
                    audio_data, 
                    translate=False  # 不自动翻译
                )
                self.transcription_thread.transcription_ready.connect(self.on_transcription_ready)
                self.transcription_thread.start()
            else:
                self.status_label.setText("状态: 没有录音数据，请重试")
                self.status_dot.setStyleSheet(f"""
                    QLabel {{
                        font-size: 20px;
                        color: #F44336;
                        font-weight: bold;
                    }}
                """)
                self.status_bar.showMessage("❌ 没有录音数据")
                logger.warning("没有获取到音频数据")
                
        else:
            QMessageBox.warning(self, "错误", "无法停止录音")
    
    def on_transcription_ready(self, original_text: str, translated_text: str):
        """转录完成处理"""
        if original_text and not original_text.startswith("["):
            # 直接显示纯文字内容，不加时间戳
            self.chinese_text.append(original_text)
            
            # 保存中文内容
            self.current_session.append({
                "chinese": original_text,
                "english": "",  # 留空，等待手动翻译
                "timestamp": datetime.now().isoformat()
            })
            
            # 更新状态指示器
            self.status_label.setText("状态: 转录完成，您可以修改中文后点击'全文翻译'")
            self.status_dot.setStyleSheet(f"""
                QLabel {{
                    font-size: 20px;
                    color: #4CAF50;
                    font-weight: bold;
                }}
            """)
            self.status_bar.showMessage("✅ 转录完成")
        else:
            self.status_label.setText(f"状态: {original_text}")
            self.status_dot.setStyleSheet(f"""
                QLabel {{
                    font-size: 20px;
                    color: #F44336;
                    font-weight: bold;
                }}
            """)
            self.status_bar.showMessage("❌ 转录失败")
        
        logger.info(f"转录完成: {original_text[:50]}...")
    
    def manual_translate(self):
        """手动翻译中文文本框中的所有内容"""
        try:
            chinese_text = self.chinese_text.toPlainText().strip()
            
            if not chinese_text:
                QMessageBox.warning(self, "提示", "请先输入或转录中文内容")
                return
            
            # 清理所有内容，获取完整的中文文本
            import re
            full_text = self.clean_text(chinese_text)
            
            # 进一步清理，移除可能的空行和重复内容
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            clean_chinese = '\n'.join(lines)
            
            logger.info(f"准备翻译全文内容: '{clean_chinese[:100]}...'")
            
            if not clean_chinese:
                QMessageBox.warning(self, "提示", "未找到可翻译的中文内容")
                return
            
            # 简单更新状态
            self.status_label.setText("状态: 正在进行全文翻译...")
            self.status_bar.showMessage("🌐 正在进行全文翻译...")
            
            # 创建翻译线程，翻译所有内容
            self.translate_thread = TranslateThread(clean_chinese)
            self.translate_thread.translation_ready.connect(self.on_manual_translation_ready)
            self.translate_thread.start()
            
        except Exception as e:
            logger.error(f"翻译时出错: {str(e)}")
            QMessageBox.warning(self, "提示", f"翻译时出现错误: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """清理语气词和停顿词，修复中文标点符号"""
        import re
        
        # 常见语气词列表
        filler_words = [
            # 单字语气词
            '嗯', '啊', '哦', '唉', '咦', '哟', '嘿', '嗯哼',
            '那个', '这个', '就是', '然后', '还有', '或者',
            '吼', '哈', '嘿咻', '呃', '唔', '嘛', '哦哦',
            # 停顿词
            '嗯嗯', '啊啊', '哦哦', '呃呃', '唔唔', '嘛嘛',
            # 连接词
            '然后呢', '还有就是', '就是说', '也就是说',
            # 思考词
            '让我想想', '我想想', '这个嘛', '那个嘛',
            # 常见口头禅
            '对吧', '是吧', '对吧', '对不对', '是不是',
            # 犹豫词
            '怎么说呢', '怎么说', '大概', '可能', '也许',
            '好像', '似乎', '差不多', '基本上'
        ]
        
        # 清理文本
        cleaned_text = text.strip()
        
        # 第一步：移除错误的英文内容（中文识别不应该包含英文单词）
        cleaned_text = re.sub(r'[a-zA-Z]+', '', cleaned_text)
        
        # 第二步：标准化中文标点符号（符合中文文法习惯）
        # 英文标点转中文标点
        punctuation_map = {
            '.': '。',      # 英文句号 → 中文句号
            ',': '，',      # 英文逗号 → 中文逗号
            '?': '？',      # 英文问号 → 中文问号
            '!': '！',      # 英文感叹号 → 中文感叹号
            ':': '：',      # 英文冒号 → 中文冒号
            ';': '；',      # 英文分号 → 中文分号
            '"': '"',      # 英文双引号 → 中文双引号
            "'": "'",      # 英文单引号 → 中文单引号
            '(': '（',      # 英文左括号 → 中文左括号
            ')': '）',      # 英文右括号 → 中文右括号
        }
        
        for en_punc, zh_punc in punctuation_map.items():
            cleaned_text = cleaned_text.replace(en_punc, zh_punc)
        
        # 第三步：修正中文标点符号使用规范
        # 顿号（、）用于并列词语，逗号（，）用于句子分隔
        # 智能判断：如果是单个汉字的并列，使用顿号
        cleaned_text = re.sub(r'([\u4e00-\u9fff])、([\u4e00-\u9fff])', r'\1、\2', cleaned_text)
        
        # 修正错误使用顿号的情况，改为逗号
        cleaned_text = re.sub(r'，、', '，', cleaned_text)  # 重复标点
        cleaned_text = re.sub(r'、，', '，', cleaned_text)  # 重复标点
        cleaned_text = re.sub(r'、(?!\s*[\u4e00-\u9fff])', '，', cleaned_text)  # 后面不是汉字用逗号
        
        # 第二步：移除语气词
        for filler in filler_words:
            pattern = r'\s*' + re.escape(filler) + r'\s*'
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 第三步：清理多余空格（中文通常不使用空格）
        cleaned_text = re.sub(r'\s+', '', cleaned_text)
        
        # 第四步：修复重复标点符号
        duplicate_puncs = [
            (r'[，，]{2,}', '，'),      # 多个逗号
            (r'[、、]{2,}', '、'),      # 多个顿号
            (r'[。。]{2,}', '。'),      # 多个句号
            (r'[！！]{2,}', '！'),      # 多个感叹号
            (r'[？？]{2,}', '？'),      # 多个问号
            (r'[；；]{2,}', '；'),      # 多个分号
            (r'[：：]{2,}', '：'),      # 多个冒号
            (r'[""]{2,}', '"'),       # 多个双引号
            (r"[']{2,}", "'"),       # 多个单引号
            (r'[（(）（)]{2,}', '（')   # 多个括号
        ]
        
        for pattern, replacement in duplicate_puncs:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # 第五步：移除句首无意义的连接词
        start_words = ['然后', '还有', '就是', '那个', '这个', '嗯', '啊', '哦']
        while any(cleaned_text.startswith(word) for word in start_words):
            for word in start_words:
                if cleaned_text.startswith(word):
                    cleaned_text = cleaned_text[len(word):].strip()
                    break
        
        # 第六步：确保合适的结尾标点（中文文法习惯）
        if cleaned_text:
            # 中文结尾标点优先级：？> ！> 。> 、> ，> ；> ：
            # 注意：顿号（、）通常不用作句子结尾
            if cleaned_text.endswith(('。', '！', '？', '、', '，', '；', '：')):
                pass  # 已有合适标点（顿号结尾较少见但允许）
            elif any(word in cleaned_text for word in ['吗', '呢', '吧', '么']):
                cleaned_text += '？'  # 疑问语气
            elif any(word in cleaned_text for word in ['啊', '呀', '啦']):
                cleaned_text += '！'  # 感叹语气
            else:
                cleaned_text += '。'  # 默认陈述语气
        
        # 第七步：特殊中文表达优化
        chinese_fixes = {
            '过程当中': '过程中',
            '这里边': '这里边',
            '那么': '那么',  # 保留必要的连词
            '因为所以': '因为，所以',
        }
        
        for wrong, correct in chinese_fixes.items():
            cleaned_text = cleaned_text.replace(wrong, correct)
        
        logger.info(f"原文: {text}")
        logger.info(f"清理后: {cleaned_text}")
        
        return cleaned_text
    
    def fix_english_punctuation(self, text: str) -> str:
        """修复英文标点符号，符合英文文法习惯"""
        import re
        
        if not text:
            return text
        
        # 英文标点符号标准化
        cleaned = text.strip()
        
        # 第一步：确保英文标点符号格式正确
        # 中文标点转英文标点
        zh_to_en_map = {
            '。': '.',      # 中文句号 → 英文句号
            '，': ',',      # 中文逗号 → 英文逗号
            '？': '?',      # 中文问号 → 英文问号
            '！': '!',      # 中文感叹号 → 英文感叹号
            '：': ':',      # 中文冒号 → 英文冒号
            '；': ';',      # 中文分号 → 英文分号
            '"': '"',      # 中文双引号 → 英文双引号
            "'": "'",      # 中文单引号 → 英文单引号
            '（': '(',      # 中文左括号 → 英文左括号
            '）': ')',      # 中文右括号 → 英文右括号
        }
        
        for zh_punc, en_punc in zh_to_en_map.items():
            cleaned = cleaned.replace(zh_punc, en_punc)
        
        # 第二步：修复空格问题（英文需要适当的空格）
        # 标点符号前的空格应该去除
        cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)
        
        # 标点符号后的空格（除了句末）
        cleaned = re.sub(r'([.,!?;:])(?=[A-Za-z])', r'\1 ', cleaned)
        
        # 括号周围的空格
        cleaned = re.sub(r'\(\s+', '(', cleaned)
        cleaned = re.sub(r'\s+\)', ')', cleaned)
        
        # 引号周围的空格
        cleaned = re.sub(r'"\s+', '"', cleaned)
        cleaned = re.sub(r'\s+"', '"', cleaned)
        cleaned = re.sub(r"'\s+", "'", cleaned)
        cleaned = re.sub(r"\s+'", "'", cleaned)
        
        # 第三步：确保句子首字母大写
        sentences = re.split(r'([.!?]+)', cleaned)
        for i in range(0, len(sentences), 2):
            if sentences[i].strip():
                sentences[i] = sentences[i].strip()
                if sentences[i] and sentences[i][0].islower():
                    sentences[i] = sentences[i][0].upper() + sentences[i][1:]
        
        cleaned = ''.join(sentences)
        
        # 第四步：修复多余的空格
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 第五步：确保句子以合适的标点结尾
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            # 根据内容判断结尾标点
            if any(word in cleaned.lower() for word in ['?', 'how', 'what', 'when', 'where', 'why', 'who']):
                cleaned += '?'
            elif any(word in cleaned.lower() for word in ['!', 'wow', 'great', 'amazing']):
                cleaned += '!'
            else:
                cleaned += '.'
        
        return cleaned.strip()

    def on_manual_translation_ready(self, translated_text: str):
        """手动翻译完成处理"""
        try:
            if translated_text and not translated_text.startswith('['):
                # 修复英文标点符号
                fixed_text = self.fix_english_punctuation(translated_text)
                
                # 清空之前的翻译内容，只显示最新的翻译结果
                self.english_text.clear()
                self.english_text.append(fixed_text)
                
                # 简单更新状态
                self.status_label.setText("状态: 翻译完成")
                self.status_bar.showMessage("✅ 翻译完成")
            else:
                self.status_label.setText(f"状态: {translated_text}")
                self.status_bar.showMessage("❌ 翻译失败")
                
        except Exception as e:
            logger.error(f"翻译完成处理时出错: {str(e)}")
            QMessageBox.warning(self, "提示", f"翻译完成处理时出现错误: {str(e)}")
    
    def clear_text(self):
        """清空文本"""
        try:
            reply = QMessageBox.question(self, "确认清空", "确定要清空所有文本内容吗？",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # 简单直接地清空
                self.chinese_text.clear()
                self.english_text.clear()
                self.current_session = []
                
                # 更新状态
                self.status_label.setText("状态: 已清空")
                self.status_bar.showMessage("🗑️ 已清空")
                
        except Exception as e:
            logger.error(f"清空文本时出错: {str(e)}")
            QMessageBox.warning(self, "提示", f"清空时出现错误: {str(e)}")
    
    def update_status(self):
        """更新状态显示"""
        if self.recorder.is_recording:
            duration = time.time() - self.recorder.start_time if self.recorder.start_time else 0
            self.record_status_label.setText(f"正在录音... {duration:.1f}秒")
            self.status_label.setText(f"状态: 正在录音... {duration:.1f}秒")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        try:
            if self.recorder.is_recording:
                self.recorder.stop_recording()
            self.recorder.cleanup()
            
            # 保存会话记录
            if self.current_session:
                session_file = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(self.current_session, f, ensure_ascii=False, indent=2)
                logger.info(f"会话记录已保存到: {session_file}")
            
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")
        
        event.accept()

def create_app_icon():
    """创建应用图标"""
    # 创建32x32的图标
    icon_pixmap = QPixmap(32, 32)
    icon_pixmap.fill(Qt.transparent)
    
    painter = QPainter(icon_pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 绘制圆形背景
    painter.setBrush(QBrush(QColor(JIAKENG_BLUE)))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(2, 2, 28, 28)
    
    # 绘制白色麦克风
    painter.setPen(QPen(QColor("white"), 2, Qt.SolidLine, Qt.RoundCap))
    painter.setBrush(QBrush(QColor("white")))
    
    center_x, center_y = 16, 16
    
    # 麦克风主体
    mic_rect = QRect(center_x - 4, center_y - 6, 8, 10)
    painter.drawEllipse(mic_rect)
    
    # 麦克风网格
    painter.setPen(QPen(QColor(JIAKENG_BLUE), 1))
    painter.drawLine(center_x - 2, center_y - 4, center_x + 2, center_y - 4)
    painter.drawLine(center_x - 2, center_y - 2, center_x + 2, center_y - 2)
    painter.drawLine(center_x - 2, center_y, center_x + 2, center_y)
    
    # 麦克风支架
    painter.setPen(QPen(QColor("white"), 2))
    painter.drawLine(center_x - 5, center_y + 5, center_x + 5, center_y + 5)
    painter.drawLine(center_x, center_y + 5, center_x, center_y + 10)
    painter.drawLine(center_x - 7, center_y + 10, center_x + 7, center_y + 10)
    
    painter.end()
    
    return QIcon(icon_pixmap)

def main():
    """主函数"""
    try:
        app = QApplication(sys.argv)
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # 设置应用图标
        app_icon = create_app_icon()
        app.setWindowIcon(app_icon)
        
        window = VoiceSecretaryGUI()
        window.setWindowIcon(app_icon)
        window.show()
        
        logger.info("语音小秘启动成功")
        return app.exec_()
        
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())