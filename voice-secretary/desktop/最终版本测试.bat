@echo off
chcp 65001 >nul
echo 🎉 启动语音小秘最终版本 v1.2.0
echo.
echo ✨ 最新更新：
echo - 界面优化："清空"→"全部清空"，"翻译全部"→"全文翻译"
echo - 功能修复：翻译中文文本框所有内容
echo - 错误修复：clean_text方法定位问题
echo.
echo 🎮 测试要点：
echo 1. 按钮显示："全部清空" 和 "全文翻译"
echo 2. 连续录音后点击"全文翻译"测试
echo 3. 验证翻译包含所有内容
echo.
cd /d "C:\Users\Cynthia Chen\.codebuddy\extensions\voice-secretary\desktop"

python voice_simplified.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ 程序启动失败，请检查错误信息
)

pause