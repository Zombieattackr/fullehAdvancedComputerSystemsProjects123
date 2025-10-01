@echo off
setlocal

if "%1"=="" (
    echo Usage: run_single.bat [experiment_name]
    echo Experiments: baseline, locality, alignment, stride, datatype
    exit /b 1
)

:: Change to script directory and set proper paths
cd /d %~dp0
cd ..

if not exist "build\simd_profiling_auto.exe" (
    echo ERROR: simd_profiling_auto.exe not found in build directory!
    echo Please run build.bat first.
    exit /b 1
)

if not exist "data" mkdir "data"

echo Running experiment: %1
build\simd_profiling_auto.exe %1 data\%1.csv

if errorlevel 1 (
    echo ERROR: Experiment failed with error code %errorlevel%
) else (
    echo Experiment completed successfully.
)

cd scripts