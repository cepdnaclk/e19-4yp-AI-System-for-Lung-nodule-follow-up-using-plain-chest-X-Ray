# PowerShell script to move experimental files to archive
$experimental_files = @(
    "architecture_analysis.md",
    "CRITICAL_FIXES.md", 
    "debug_analysis.py",
    "emergency_hybrid.py",
    "improved_model.py",
    "improved_train.py", 
    "improved_train_fixed.py",
    "improved_utils.py",
    "IMPROVEMENTS.md",
    "IMPROVEMENTS_SUMMARY.md",
    "minimalist_model.py",
    "quick_eval.py",
    "README_IMPROVEMENTS.md",
    "simple_config.py",
    "simple_evaluate.py",
    "test_imports.py",
    "ultra_conservative_train.py",
    "cleanup_pipeline.py"
)

Write-Host "Moving experimental files to archive..." -ForegroundColor Green

foreach ($file in $experimental_files) {
    if (Test-Path $file) {
        Move-Item $file "archive_experimental/" -Force
        Write-Host "Moved: $file" -ForegroundColor Yellow
    } else {
        Write-Host "Not found: $file" -ForegroundColor Red
    }
}

Write-Host "Cleanup completed!" -ForegroundColor Green
