# PowerShell脚本 - 运行新方法测试

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "运行新方法测试脚本"  -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

# 设置错误动作
$ErrorActionPreference = "Continue"

# 步骤1: 运行详细诊断测试
Write-Host "[1/3] 运行详细诊断测试..." -ForegroundColor Yellow
Write-Host ""

$outputFile = "test_output_powershell.txt"
Start-Process -FilePath "python" -ArgumentList "test_detailed.py" -NoNewWindow -Wait -RedirectStandardOutput $outputFile -RedirectStandardError "test_error_powershell.txt"

Write-Host "诊断完成！" -ForegroundColor Green
Write-Host ""

# 步骤2: 检查输出文件
Write-Host "[2/3] 检查输出文件..." -ForegroundColor Yellow
Write-Host ""

$logFile = Get-ChildItem test_detailed_log_*.txt | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($logFile) {
    Write-Host "✓ 找到详细日志文件:" -ForegroundColor Green
    Write-Host "  $($logFile.Name)" -ForegroundColor White
    Write-Host ""
    Write-Host "最后30行内容:" -ForegroundColor Yellow
    Write-Host ""
    Get-Content $logFile.FullName | Select-Object -Last 30 | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "⚠ 未找到详细日志文件" -ForegroundColor Red
    Write-Host ""
    Write-Host "test_output_powershell.txt 内容:" -ForegroundColor Yellow
    Write-Host ""
    if (Test-Path $outputFile) {
        Get-Content $outputFile | ForEach-Object { Write-Host $_ }
    } else {
        Write-Host "文件不存在" -ForegroundColor Red
    }
}

Write-Host ""

# 步骤3: 检查错误
Write-Host "[3/3] 检查错误..." -ForegroundColor Yellow
Write-Host ""

$hasErrors = $false

if (Test-Path $outputFile) {
    $errors = Select-String -Path $outputFile -Pattern "(error|failed|exception|Error|ERROR|Exception)" -CaseSensitive:$false
    if ($errors) {
        $hasErrors = $true
        Write-Host "⚠ 发现错误！" -ForegroundColor Red
        Write-Host ""
        Write-Host "包含'error/failed/exception'的行:" -ForegroundColor Yellow
        Write-Host ""
        foreach ($line in $errors.Line) {
            Write-Host $line -ForegroundColor Red
        }
    } else {
        Write-Host "✓ 未发现错误" -ForegroundColor Green
    }
}

Write-Host ""

# 总结
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "测试完成！"  -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""
Write-Host "详细输出文件:" -ForegroundColor Yellow
Write-Host "  - test_output_powershell.txt (脚本输出)" -ForegroundColor White
if ($logFile) {
    Write-Host "  - $($logFile.Name) (详细日志)" -ForegroundColor White
}
if (Test-Path "test_error_powershell.txt") {
    Write-Host "  - test_error_powershell.txt (错误输出)" -ForegroundColor White
}
Write-Host ""

if (-not $hasErrors) {
    Write-Host "✓ 所有测试似乎都成功通过！" -ForegroundColor Green
    Write-Host ""
    Write-Host "下一步操作:" -ForegroundColor Yellow
    Write-Host "  1. 运行快速测试: python quick_test_new_methods.py" -ForegroundColor White
    Write-Host "  2. 运行完整实验: python experiment_new_methods.py" -ForegroundColor White
    Write-Host "  3. 使用启动脚本: python START_HERE.py" -ForegroundColor White
} else {
    Write-Host "⚠ 发现错误，请检查日志文件以获取详细信息" -ForegroundColor Red
}

Write-Host ""
Write-Host "按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
