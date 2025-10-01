#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include "kernels.h"
#include "timer.h"
#include <string>
#include <vector>
#include <fstream>

struct ExperimentResult {
    std::string name;
    double scalar_time;
    double simd_time;
    double speedup;
    double gflops;
    double bandwidth_gbs;
    size_t data_size_bytes;
    size_t flops_per_iteration;
};

class ExperimentRunner {
public:
    ExperimentRunner();
    
    // Main experiment interfaces
    void run_baseline_comparison(const std::string& output_csv);
    void run_locality_sweep(const std::string& output_csv);
    void run_alignment_study(const std::string& output_csv);
    void run_stride_study(const std::string& output_csv);
    void run_datatype_comparison(const std::string& output_csv);
    
    // Utility functions
    static std::vector<size_t> generate_sizes_for_cache_levels();
    static double calculate_gflops(double time_seconds, size_t operations, size_t iterations, size_t size);
    static double calculate_bandwidth(double time_seconds, size_t bytes_transferred, size_t iterations);
    
private:
    ExperimentResult run_single_experiment(const ExperimentConfig& config, int num_repeats = 5);
    void warmup_cache(void* data, size_t size);
    
    // Individual experiment implementations - fixed signatures
    void baseline_saxpy(std::ofstream& file, DataType dtype, size_t size);
    void baseline_dot_product(std::ofstream& file, DataType dtype, size_t size);
    void baseline_elementwise_multiply(std::ofstream& file, DataType dtype, size_t size);
    void baseline_stencil(std::ofstream& file, DataType dtype, size_t size);
};

#endif