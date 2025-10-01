#ifndef KERNELS_H
#define KERNELS_H

#include <cstddef>
#include <cstdint>

// Add these includes at the top of src/kernels.h
#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif


// Kernel types
enum class KernelType {
    SAXPY,
    DOT_PRODUCT,
    ELEMENTWISE_MULTIPLY,
    STENCIL_3POINT
};

// Data types
enum class DataType {
    FLOAT32,
    FLOAT64,
    INT32
};

// Experiment configuration
struct ExperimentConfig {
    KernelType kernel;
    DataType dtype;
    size_t size;
    size_t stride;
    bool aligned;
    size_t iterations;
    size_t warmup;
    
    ExperimentConfig() 
        : kernel(KernelType::SAXPY), dtype(DataType::FLOAT32), 
          size(1024), stride(1), aligned(true), iterations(100), warmup(10) {}
};

// Function declarations
void initialize_data(void* data, size_t size, DataType dtype, int seed = 42);
void* allocate_aligned(size_t size, size_t alignment);
void* allocate_misaligned(size_t size, size_t alignment, size_t offset);
void free_aligned(void* ptr);

// Scalar (baseline) implementations
double saxpy_scalar(const ExperimentConfig& config, float a, const float* x, float* y);
double saxpy_scalar(const ExperimentConfig& config, double a, const double* x, double* y);

double dot_product_scalar(const ExperimentConfig& config, const float* x, const float* y);
double dot_product_scalar(const ExperimentConfig& config, const double* x, const double* y);

double elementwise_multiply_scalar(const ExperimentConfig& config, const float* x, const float* y, float* z);
double elementwise_multiply_scalar(const ExperimentConfig& config, const double* x, const double* y, double* z);

double stencil_3point_scalar(const ExperimentConfig& config, float a, float b, float c, const float* x, float* y);
double stencil_3point_scalar(const ExperimentConfig& config, double a, double b, double c, const double* x, double* y);

// Auto-vectorized implementations (same interface, rely on compiler)
double saxpy_auto(const ExperimentConfig& config, float a, const float* x, float* y);
double saxpy_auto(const ExperimentConfig& config, double a, const double* x, double* y);

double dot_product_auto(const ExperimentConfig& config, const float* x, const float* y);
double dot_product_auto(const ExperimentConfig& config, const double* x, const double* y);

double elementwise_multiply_auto(const ExperimentConfig& config, const float* x, const float* y, float* z);
double elementwise_multiply_auto(const ExperimentConfig& config, const double* x, const double* y, double* z);

double stencil_3point_auto(const ExperimentConfig& config, float a, float b, float c, const float* x, float* y);
double stencil_3point_auto(const ExperimentConfig& config, double a, double b, double c, const double* x, double* y);

// Add these INT32 declarations to the existing function declarations

// Scalar INT32 implementations
double saxpy_scalar(const ExperimentConfig& config, int32_t a, const int32_t* x, int32_t* y);
double dot_product_scalar(const ExperimentConfig& config, const int32_t* x, const int32_t* y);
double elementwise_multiply_scalar(const ExperimentConfig& config, const int32_t* x, const int32_t* y, int32_t* z);
double stencil_3point_scalar(const ExperimentConfig& config, int32_t a, int32_t b, int32_t c, const int32_t* x, int32_t* y);

// Auto-vectorized INT32 implementations
double saxpy_auto(const ExperimentConfig& config, int32_t a, const int32_t* x, int32_t* y);
double dot_product_auto(const ExperimentConfig& config, const int32_t* x, const int32_t* y);
double elementwise_multiply_auto(const ExperimentConfig& config, const int32_t* x, const int32_t* y, int32_t* z);
double stencil_3point_auto(const ExperimentConfig& config, int32_t a, int32_t b, int32_t c, const int32_t* x, int32_t* y);

#endif