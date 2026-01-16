@echo off
chcp 65001 >nul
echo ========================================
echo 运行新方法测试脚本
echo ========================================
echo.

echo [1/3] 运行详细诊断测试...
echo.
python test_detailed.py > test_output.txt 2>&1
echo 诊断完成！结果已保存到 test_output.txt
echo.

echo [2/3] 检查输出文件...
echo.
if exist test_detailed_log_*.txt (
    echo ✓ 找到详细日志文件:
    dir /b test_detailed_log_*.txt
    echo.
    echo 最后20行内容:
    powershell -Command "Get-Content (Get-ChildItem test_detailed_log_*.txt | Select-Object -Last 1) | Select-Object -Last 20"
) else (
    echo ⚠ 未找到详细日志文件
    echo.
    echo test_output.txt 内容:
    type test_output.txt
)
echo.

echo [3/3] 检查错误...
echo.
findstr /i /c:"error" /c:"failed" /c:"exception" test_output.txt >nul
if %errorlevel% equ 0 (
    echo ⚠ 发现错误！
    echo.
    echo 包含"error/failed/exception"的行:
    findstr /i /c:"error" /c:"failed" /c:"exception" test_output.txt
) else (
    echo ✓ 未发现错误
)
echo.

echo ========================================
echo 测试完成！
echo ========================================
echo.
echo 详细输出文件:
echo   - test_output.txt (脚本输出)
echo   - test_detailed_log_*.txt (详细日志)
echo.
pause
