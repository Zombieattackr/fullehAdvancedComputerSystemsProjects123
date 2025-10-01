@echo off
setlocal enabledelayedexpansion

echo Running SIMD Profiling Experiments...

:: Change to script directory and set proper paths
cd /d %~dp0
cd ..

:: Create data directory if it doesn't exist
if not exist "data" mkdir "data"

:: Set CPU to performance mode (requires admin privileges)
echo Setting CPU performance mode...
powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1
if errorlevel 1 (
    echo Note: Could not set performance mode (admin rights required)
) else (
    echo Performance mode activated
)

echo Running baseline comparison...
build\simd_profiling_auto.exe baseline data\baseline.csv

echo Running locality sweep...
build\simd_profiling_auto.exe locality data\locality.csv

echo Running alignment study...
build\simd_profiling_auto.exe alignment data\alignment.csv

echo Running stride study...
build\simd_profiling_auto.exe stride data\stride.csv

echo Running data type comparison...
build\simd_profiling_auto.exe datatype data\datatype.csv

echo All experiments completed!
echo Results saved to data\

:: List generated files
if exist data\*.csv (
    echo Generated CSV files:
    dir data\*.csv
) else (
    echo WARNING: No CSV files were generated!
)

cd scripts