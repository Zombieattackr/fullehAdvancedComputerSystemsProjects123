#include "kernels.h"
#include "timer.h"
#include <cstdlib>
#include <cstring>
#include <random>

void initialize_data(void* data, size_t size, DataType dtype, int seed) {
    std::mt19937 gen(seed);
    
    switch (dtype) {
        case DataType::FLOAT32: {
            std::uniform_real_distribution<float> dist(0.1f, 1.0f);
            float* ptr = static_cast<float*>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = dist(gen);
            }
            break;
        }
        case DataType::FLOAT64: {
            std::uniform_real_distribution<double> dist(0.1, 1.0);
            double* ptr = static_cast<double*>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = dist(gen);
            }
            break;
        }
        case DataType::INT32: {
            std::uniform_int_distribution<int32_t> dist(1, 100);
            int32_t* ptr = static_cast<int32_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = dist(gen);
            }
            break;
        }
    }
}

void* allocate_aligned(size_t size, size_t alignment) {
    void* ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    return ptr;
}

void* allocate_misaligned(size_t size, size_t alignment, size_t offset) {
    char* base = static_cast<char*>(allocate_aligned(size + alignment, alignment));
    return base ? (base + offset) : nullptr;
}

void free_aligned(void* ptr) {
    if (!ptr) return;
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Scalar implementations
double saxpy_scalar(const ExperimentConfig& config, float a, const float* x, float* y) {
    auto start = HighResolutionTimer::now();
    
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        for (size_t i = 0; i < config.size; i += config.stride) {
            y[i] = a * x[i] + y[i];
        }
    }
    
    auto end = HighResolutionTimer::now();
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double saxpy_scalar(const ExperimentConfig& config, double a, const double* x, double* y) {
    auto start = HighResolutionTimer::now();
    
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        for (size_t i = 0; i < config.size; i += config.stride) {
            y[i] = a * x[i] + y[i];
        }
    }
    
    auto end = HighResolutionTimer::now();
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double dot_product_scalar(const ExperimentConfig& config, const float* x, const float* y) {
    auto start = HighResolutionTimer::now();
    
    float result = 0.0f;
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        result = 0.0f;
        for (size_t i = 0; i < config.size; i += config.stride) {
            result += x[i] * y[i];
        }
    }
    
    auto end = HighResolutionTimer::now();
    // Use result to prevent optimization
    volatile float sink = result;
    (void)sink;
    
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double dot_product_scalar(const ExperimentConfig& config, const double* x, const double* y) {
    auto start = HighResolutionTimer::now();
    
    double result = 0.0;
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        result = 0.0;
        for (size_t i = 0; i < config.size; i += config.stride) {
            result += x[i] * y[i];
        }
    }
    
    auto end = HighResolutionTimer::now();
    volatile double sink = result;
    (void)sink;
    
    return HighResolutionTimer::elapsedSeconds(start, end);
}

// Auto-vectorized implementations (identical to scalar but rely on compiler optimization)
double saxpy_auto(const ExperimentConfig& config, float a, const float* x, float* y) {
    return saxpy_scalar(config, a, x, y);
}

double saxpy_auto(const ExperimentConfig& config, double a, const double* x, double* y) {
    return saxpy_scalar(config, a, x, y);
}

double dot_product_auto(const ExperimentConfig& config, const float* x, const float* y) {
    return dot_product_scalar(config, x, y);
}

double dot_product_auto(const ExperimentConfig& config, const double* x, const double* y) {
    return dot_product_scalar(config, x, y);
}

// Add other kernel implementations following the same pattern...
// [Additional implementations for elementwise_multiply and stencil_3point would go here]
// For brevity, I'm showing the pattern with SAXPY and Dot Product

double elementwise_multiply_scalar(const ExperimentConfig& config, const float* x, const float* y, float* z) {
    auto start = HighResolutionTimer::now();
    
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        for (size_t i = 0; i < config.size; i += config.stride) {
            z[i] = x[i] * y[i];
        }
    }
    
    auto end = HighResolutionTimer::now();
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double elementwise_multiply_auto(const ExperimentConfig& config, const float* x, const float* y, float* z) {
    return elementwise_multiply_scalar(config, x, y, z);
}

double stencil_3point_scalar(const ExperimentConfig& config, float a, float b, float c, const float* x, float* y) {
    auto start = HighResolutionTimer::now();
    
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        for (size_t i = 1; i < config.size - 1; i += config.stride) {
            y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
        }
    }
    
    auto end = HighResolutionTimer::now();
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double stencil_3point_auto(const ExperimentConfig& config, float a, float b, float c, const float* x, float* y) {
    return stencil_3point_scalar(config, a, b, c, x, y);
}

// Add these to the existing kernels.cpp file

double elementwise_multiply_scalar(const ExperimentConfig& config, const double* x, const double* y, double* z) {
    auto start = HighResolutionTimer::now();
    
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        for (size_t i = 0; i < config.size; i += config.stride) {
            z[i] = x[i] * y[i];
        }
    }
    
    auto end = HighResolutionTimer::now();
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double elementwise_multiply_auto(const ExperimentConfig& config, const double* x, const double* y, double* z) {
    return elementwise_multiply_scalar(config, x, y, z);
}

double stencil_3point_scalar(const ExperimentConfig& config, double a, double b, double c, const double* x, double* y) {
    auto start = HighResolutionTimer::now();
    
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        for (size_t i = 1; i < config.size - 1; i += config.stride) {
            y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
        }
    }
    
    auto end = HighResolutionTimer::now();
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double stencil_3point_auto(const ExperimentConfig& config, double a, double b, double c, const double* x, double* y) {
    return stencil_3point_scalar(config, a, b, c, x, y);
}

// Add these INT32 implementations to kernels.cpp

// Scalar INT32 implementations
double saxpy_scalar(const ExperimentConfig& config, int32_t a, const int32_t* x, int32_t* y) {
    auto start = HighResolutionTimer::now();
    
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        for (size_t i = 0; i < config.size; i += config.stride) {
            y[i] = a * x[i] + y[i];
        }
    }
    
    auto end = HighResolutionTimer::now();
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double dot_product_scalar(const ExperimentConfig& config, const int32_t* x, const int32_t* y) {
    auto start = HighResolutionTimer::now();
    
    int32_t result = 0;
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        result = 0;
        for (size_t i = 0; i < config.size; i += config.stride) {
            result += x[i] * y[i];
        }
    }
    
    auto end = HighResolutionTimer::now();
    volatile int32_t sink = result;
    (void)sink;
    
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double elementwise_multiply_scalar(const ExperimentConfig& config, const int32_t* x, const int32_t* y, int32_t* z) {
    auto start = HighResolutionTimer::now();
    
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        for (size_t i = 0; i < config.size; i += config.stride) {
            z[i] = x[i] * y[i];
        }
    }
    
    auto end = HighResolutionTimer::now();
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double stencil_3point_scalar(const ExperimentConfig& config, int32_t a, int32_t b, int32_t c, const int32_t* x, int32_t* y) {
    auto start = HighResolutionTimer::now();
    
    for (size_t iter = 0; iter < config.iterations; ++iter) {
        for (size_t i = 1; i < config.size - 1; i += config.stride) {
            y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
        }
    }
    
    auto end = HighResolutionTimer::now();
    return HighResolutionTimer::elapsedSeconds(start, end);
}

// Auto-vectorized INT32 implementations
double saxpy_auto(const ExperimentConfig& config, int32_t a, const int32_t* x, int32_t* y) {
    return saxpy_scalar(config, a, x, y);
}

double dot_product_auto(const ExperimentConfig& config, const int32_t* x, const int32_t* y) {
    return dot_product_scalar(config, x, y);
}

double elementwise_multiply_auto(const ExperimentConfig& config, const int32_t* x, const int32_t* y, int32_t* z) {
    return elementwise_multiply_scalar(config, x, y, z);
}

double stencil_3point_auto(const ExperimentConfig& config, int32_t a, int32_t b, int32_t c, const int32_t* x, int32_t* y) {
    return stencil_3point_scalar(config, a, b, c, x, y);
}

// Optimized versions with hints for the compiler
double saxpy_optimized(const ExperimentConfig& config, float a, const float* __restrict x, float* __restrict y) {
    auto start = HighResolutionTimer::now();
    
    size_t size = config.size;
    size_t iterations = config.iterations;
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        // Hint to compiler that loops can be vectorized
        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            y[i] = a * x[i] + y[i];
        }
    }
    
    auto end = HighResolutionTimer::now();
    return HighResolutionTimer::elapsedSeconds(start, end);
}

double dot_product_optimized(const ExperimentConfig& config, const float* __restrict x, const float* __restrict y) {
    auto start = HighResolutionTimer::now();
    
    size_t size = config.size;
    size_t iterations = config.iterations;
    
    float result = 0.0f;
    for (size_t iter = 0; iter < iterations; ++iter) {
        float sum = 0.0f;
        // Use multiple accumulators to break dependency chain
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i) {
            sum += x[i] * y[i];
        }
        result = sum; // Prevent optimization
    }
    
    auto end = HighResolutionTimer::now();
    volatile float sink = result;
    (void)sink;
    return HighResolutionTimer::elapsedSeconds(start, end);
}