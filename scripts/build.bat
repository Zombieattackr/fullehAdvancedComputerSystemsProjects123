@echo off
setlocal enabledelayedexpansion

echo Building SIMD Profiling Project...

:: Create directories if they don't exist
if not exist "build" mkdir "build"
if not exist "data" mkdir "data"

:: Set compiler flags
set COMMON_FLAGS=-O3 -march=native -mtune=native -ffast-math -fno-math-errno -fno-trapping-math
set SCALAR_FLAGS=%COMMON_FLAGS% -fno-tree-vectorize -fno-tree-slp-vectorize
set SIMD_FLAGS=%COMMON_FLAGS% -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed -fopt-info-vec-all

:: Change to script directory and set proper paths
cd /d %~dp0
cd ..

echo Building scalar version...
g++ -std=c++17 %SCALAR_FLAGS% -DSCALAR_BUILD -I src -o build/simd_profiling_scalar.exe src/main.cpp src/kernels.cpp src/experiments.cpp

echo Building auto-vectorized version...  
g++ -std=c++17 %SIMD_FLAGS% -I src -o build/simd_profiling_auto.exe src/main.cpp src/kernels.cpp src/experiments.cpp

echo Building with AVX2...
g++ -std=c++17 %COMMON_FLAGS% -mavx2 -mfma -I src -o build/simd_profiling_avx2.exe src/main.cpp src/kernels.cpp src/experiments.cpp

echo Build completed!
if exist build\*.exe (
    echo Executables created:
    dir build\*.exe
) else (
    echo ERROR: No executables were created!
)

cd scripts