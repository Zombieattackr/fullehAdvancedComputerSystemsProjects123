#include "experiments.h"
#include <fstream>
#include <iostream>
#include <cmath>

#ifdef DEBUG_TIMING
#include <iostream>
#endif

// Inside run_single_experiment method, after the timing loops:
#ifdef DEBUG_TIMING
    std::cout << "DEBUG: " 
              << (config.kernel == KernelType::SAXPY ? "SAXPY" : 
                  config.kernel == KernelType::DOT_PRODUCT ? "DOT_PRODUCT" : "OTHER")
              << " size=" << config.size 
              << " dtype=" << (config.dtype == DataType::FLOAT32 ? "float32" : 
                              config.dtype == DataType::FLOAT64 ? "float64" : "int32")
              << " scalar_time=" << scalar_stats.mean()
              << " simd_time=" << simd_stats.mean()
              << " speedup=" << (scalar_stats.mean() / simd_stats.mean())
              << std::endl;
#endif

ExperimentRunner::ExperimentRunner() {}

std::vector<size_t> ExperimentRunner::generate_sizes_for_cache_levels() {
    // Typical cache sizes (adjust based on your CPU)
    std::vector<size_t> sizes;
    
    // L1 cache sizes (32-64KB)
    for (size_t size = 256; size <= 16384; size *= 2) {
        sizes.push_back(size);
    }
    
    // L2 cache sizes (256KB-1MB)
    for (size_t size = 32768; size <= 262144; size *= 2) {
        sizes.push_back(size);
    }
    
    // L3 cache sizes (8-32MB)
    for (size_t size = 524288; size <= 8388608; size *= 2) {
        sizes.push_back(size);
    }
    
    // DRAM sizes
    sizes.push_back(16777216);  // 16M
    sizes.push_back(33554432);  // 32M
    sizes.push_back(67108864);  // 64M
    
    return sizes;
}

double ExperimentRunner::calculate_gflops(double time_seconds, size_t operations, size_t iterations, size_t size) {
    if (time_seconds <= 0) return 0.0;
    double total_operations = static_cast<double>(operations) * iterations * size;
    return (total_operations / time_seconds) / 1e9;
}

double ExperimentRunner::calculate_bandwidth(double time_seconds, size_t bytes_transferred, size_t iterations) {
    if (time_seconds <= 0) return 0.0;
    double total_bytes = static_cast<double>(bytes_transferred) * iterations;
    return (total_bytes / time_seconds) / (1024 * 1024 * 1024); // GB/s
}

void ExperimentRunner::warmup_cache(void* data, size_t size) {
    // Simple cache warmup by touching each cache line
    volatile char* ptr = static_cast<volatile char*>(data);
    for (size_t i = 0; i < size; i += 64) {
        ptr[i] = ptr[i];
    }
}

ExperimentResult ExperimentRunner::run_single_experiment(const ExperimentConfig& config, int num_repeats) {
    ExperimentResult result;
    MeasurementStats scalar_stats, simd_stats;
    
    // Determine element size
    size_t element_size = 4; // default float32
    if (config.dtype == DataType::FLOAT64) element_size = 8;
    else if (config.dtype == DataType::INT32) element_size = 4;
    
    // Allocate and initialize data
    size_t alignment = 64; // Cache line alignment
    void *x, *y, *z;
    
    if (config.aligned) {
        x = allocate_aligned(config.size * element_size, alignment);
        y = allocate_aligned(config.size * element_size, alignment);
        z = allocate_aligned(config.size * element_size, alignment);
    } else {
        x = allocate_misaligned(config.size * element_size, alignment, 7); // 7 byte offset
        y = allocate_misaligned(config.size * element_size, alignment, 7);
        z = allocate_misaligned(config.size * element_size, alignment, 7);
    }
    
    if (!x || !y || !z) {
        std::cerr << "Memory allocation failed for size " << config.size << std::endl;
        return result;
    }
    
    initialize_data(x, config.size, config.dtype, 42);
    initialize_data(y, config.size, config.dtype, 43);
    initialize_data(z, config.size, config.dtype, 44);
    
    // Run experiments multiple times
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
        // Warm up
        warmup_cache(x, config.size * element_size);
        warmup_cache(y, config.size * element_size);
        warmup_cache(z, config.size * element_size);
        
        // Run scalar version
        double scalar_time = 0.0;
        if (config.dtype == DataType::FLOAT32) {
            if (config.kernel == KernelType::SAXPY) {
                scalar_time = saxpy_scalar(config, 2.0f, static_cast<const float*>(x), static_cast<float*>(y));
            } else if (config.kernel == KernelType::DOT_PRODUCT) {
                scalar_time = dot_product_scalar(config, static_cast<const float*>(x), static_cast<const float*>(y));
            } else if (config.kernel == KernelType::ELEMENTWISE_MULTIPLY) {
                scalar_time = elementwise_multiply_scalar(config, static_cast<const float*>(x), static_cast<const float*>(y), static_cast<float*>(z));
            } else if (config.kernel == KernelType::STENCIL_3POINT) {
                scalar_time = stencil_3point_scalar(config, 0.25f, 0.5f, 0.25f, static_cast<const float*>(x), static_cast<float*>(y));
            }
        } else if (config.dtype == DataType::FLOAT64) {
            if (config.kernel == KernelType::SAXPY) {
                scalar_time = saxpy_scalar(config, 2.0, static_cast<const double*>(x), static_cast<double*>(y));
            } else if (config.kernel == KernelType::DOT_PRODUCT) {
                scalar_time = dot_product_scalar(config, static_cast<const double*>(x), static_cast<const double*>(y));
            } else if (config.kernel == KernelType::ELEMENTWISE_MULTIPLY) {
                scalar_time = elementwise_multiply_scalar(config, static_cast<const double*>(x), static_cast<const double*>(y), static_cast<double*>(z));
            } else if (config.kernel == KernelType::STENCIL_3POINT) {
                scalar_time = stencil_3point_scalar(config, 0.25, 0.5, 0.25, static_cast<const double*>(x), static_cast<double*>(y));
            }
        } else if (config.dtype == DataType::INT32) {
            if (config.kernel == KernelType::SAXPY) {
                scalar_time = saxpy_scalar(config, 2, static_cast<const int32_t*>(x), static_cast<int32_t*>(y));
            } else if (config.kernel == KernelType::DOT_PRODUCT) {
                scalar_time = dot_product_scalar(config, static_cast<const int32_t*>(x), static_cast<const int32_t*>(y));
            } else if (config.kernel == KernelType::ELEMENTWISE_MULTIPLY) {
                scalar_time = elementwise_multiply_scalar(config, static_cast<const int32_t*>(x), static_cast<const int32_t*>(y), static_cast<int32_t*>(z));
            } else if (config.kernel == KernelType::STENCIL_3POINT) {
                scalar_time = stencil_3point_scalar(config, 2, 3, 2, static_cast<const int32_t*>(x), static_cast<int32_t*>(y));
            }
        }
        scalar_stats.add(scalar_time);
        
        // Reset data for SIMD run
        initialize_data(y, config.size, config.dtype, 43 + repeat);
        if (config.kernel == KernelType::ELEMENTWISE_MULTIPLY || config.kernel == KernelType::STENCIL_3POINT) {
            initialize_data(z, config.size, config.dtype, 44 + repeat);
        }
        
        // Run auto-vectorized version
        double simd_time = 0.0;
        if (config.dtype == DataType::FLOAT32) {
            if (config.kernel == KernelType::SAXPY) {
                simd_time = saxpy_auto(config, 2.0f, static_cast<const float*>(x), static_cast<float*>(y));
            } else if (config.kernel == KernelType::DOT_PRODUCT) {
                simd_time = dot_product_auto(config, static_cast<const float*>(x), static_cast<const float*>(y));
            } else if (config.kernel == KernelType::ELEMENTWISE_MULTIPLY) {
                simd_time = elementwise_multiply_auto(config, static_cast<const float*>(x), static_cast<const float*>(y), static_cast<float*>(z));
            } else if (config.kernel == KernelType::STENCIL_3POINT) {
                simd_time = stencil_3point_auto(config, 0.25f, 0.5f, 0.25f, static_cast<const float*>(x), static_cast<float*>(y));
            }
        } else if (config.dtype == DataType::FLOAT64) {
            if (config.kernel == KernelType::SAXPY) {
                simd_time = saxpy_auto(config, 2.0, static_cast<const double*>(x), static_cast<double*>(y));
            } else if (config.kernel == KernelType::DOT_PRODUCT) {
                simd_time = dot_product_auto(config, static_cast<const double*>(x), static_cast<const double*>(y));
            } else if (config.kernel == KernelType::ELEMENTWISE_MULTIPLY) {
                simd_time = elementwise_multiply_auto(config, static_cast<const double*>(x), static_cast<const double*>(y), static_cast<double*>(z));
            } else if (config.kernel == KernelType::STENCIL_3POINT) {
                simd_time = stencil_3point_auto(config, 0.25, 0.5, 0.25, static_cast<const double*>(x), static_cast<double*>(y));
            }
        } else if (config.dtype == DataType::INT32) {
            if (config.kernel == KernelType::SAXPY) {
                simd_time = saxpy_auto(config, 2, static_cast<const int32_t*>(x), static_cast<int32_t*>(y));
            } else if (config.kernel == KernelType::DOT_PRODUCT) {
                simd_time = dot_product_auto(config, static_cast<const int32_t*>(x), static_cast<const int32_t*>(y));
            } else if (config.kernel == KernelType::ELEMENTWISE_MULTIPLY) {
                simd_time = elementwise_multiply_auto(config, static_cast<const int32_t*>(x), static_cast<const int32_t*>(y), static_cast<int32_t*>(z));
            } else if (config.kernel == KernelType::STENCIL_3POINT) {
                simd_time = stencil_3point_auto(config, 2, 3, 2, static_cast<const int32_t*>(x), static_cast<int32_t*>(y));
            }
        }
        simd_stats.add(simd_time);
    }
    
    // Compute statistics
    scalar_stats.compute();
    simd_stats.compute();
    
    // Fill result
    result.scalar_time = scalar_stats.mean();
    result.simd_time = simd_stats.mean();
    result.speedup = scalar_stats.mean() / simd_stats.mean();
    
    // Calculate FLOPS based on kernel type
    size_t flops_per_element = 0;
    if (config.kernel == KernelType::SAXPY) flops_per_element = 2; // mul + add
    else if (config.kernel == KernelType::DOT_PRODUCT) flops_per_element = 2; // mul + add (reduction)
    else if (config.kernel == KernelType::ELEMENTWISE_MULTIPLY) flops_per_element = 1;
    else if (config.kernel == KernelType::STENCIL_3POINT) flops_per_element = 5; // 3 mul + 2 add
    
    result.flops_per_iteration = flops_per_element;
    result.gflops = calculate_gflops(simd_stats.mean(), flops_per_element, config.iterations, config.size);
    
    // Calculate bandwidth (bytes per iteration)
    size_t bytes_per_iteration = 0;
    if (config.kernel == KernelType::SAXPY) bytes_per_iteration = 3 * config.size * element_size; // read x,y write y
    else if (config.kernel == KernelType::DOT_PRODUCT) bytes_per_iteration = 2 * config.size * element_size; // read x,y
    else if (config.kernel == KernelType::ELEMENTWISE_MULTIPLY) bytes_per_iteration = 3 * config.size * element_size; // read x,y write z
    else if (config.kernel == KernelType::STENCIL_3POINT) bytes_per_iteration = 4 * config.size * element_size; // read x[i-1],x[i],x[i+1] write y[i]
    
    result.bandwidth_gbs = calculate_bandwidth(simd_stats.mean(), bytes_per_iteration, config.iterations);
    result.data_size_bytes = config.size * element_size;
    
    // Cleanup
    free_aligned(x);
    free_aligned(y);
    free_aligned(z);
    
    return result;
}

void ExperimentRunner::run_baseline_comparison(const std::string& output_csv) {
    std::cout << "Running baseline comparison experiment..." << std::endl;
    
    std::ofstream file(output_csv);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_csv << std::endl;
        return;
    }
    
    file << "Kernel,DataType,Size,ScalarTime,SIMDTime,Speedup,GFLOPs,BandwidthGBs\n";
    
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
    std::vector<DataType> dtypes = {DataType::FLOAT32, DataType::FLOAT64};
    
    for (DataType dtype : dtypes) {
        for (size_t size : sizes) {
            baseline_saxpy(file, dtype, size);
            baseline_dot_product(file, dtype, size);
            baseline_elementwise_multiply(file, dtype, size);
            baseline_stencil(file, dtype, size);
        }
    }
    
    file.close();
    std::cout << "Baseline comparison completed. Results saved to " << output_csv << std::endl;
}

void ExperimentRunner::baseline_saxpy(std::ofstream& file, DataType dtype, size_t size) {
    ExperimentConfig config;
    config.kernel = KernelType::SAXPY;
    config.dtype = dtype;
    config.size = size;
    config.iterations = size < 10000 ? 1000 : 100;
    config.warmup = 10;
    
    ExperimentResult result = run_single_experiment(config);
    
    file << "SAXPY," 
         << (dtype == DataType::FLOAT32 ? "float32" : "float64") << ","
         << size << ","
         << result.scalar_time << ","
         << result.simd_time << ","
         << result.speedup << ","
         << result.gflops << ","
         << result.bandwidth_gbs << "\n";
    
    file.flush(); // Ensure data is written immediately
}

void ExperimentRunner::baseline_dot_product(std::ofstream& file, DataType dtype, size_t size) {
    ExperimentConfig config;
    config.kernel = KernelType::DOT_PRODUCT;
    config.dtype = dtype;
    config.size = size;
    config.iterations = size < 10000 ? 1000 : 100;
    config.warmup = 10;
    
    ExperimentResult result = run_single_experiment(config);
    
    file << "DOT_PRODUCT," 
         << (dtype == DataType::FLOAT32 ? "float32" : "float64") << ","
         << size << ","
         << result.scalar_time << ","
         << result.simd_time << ","
         << result.speedup << ","
         << result.gflops << ","
         << result.bandwidth_gbs << "\n";
    
    file.flush();
}

void ExperimentRunner::baseline_elementwise_multiply(std::ofstream& file, DataType dtype, size_t size) {
    ExperimentConfig config;
    config.kernel = KernelType::ELEMENTWISE_MULTIPLY;
    config.dtype = dtype;
    config.size = size;
    config.iterations = size < 10000 ? 1000 : 100;
    config.warmup = 10;
    
    ExperimentResult result = run_single_experiment(config);
    
    file << "ELEMENTWISE_MULTIPLY," 
         << (dtype == DataType::FLOAT32 ? "float32" : "float64") << ","
         << size << ","
         << result.scalar_time << ","
         << result.simd_time << ","
         << result.speedup << ","
         << result.gflops << ","
         << result.bandwidth_gbs << "\n";
    
    file.flush();
}

void ExperimentRunner::baseline_stencil(std::ofstream& file, DataType dtype, size_t size) {
    ExperimentConfig config;
    config.kernel = KernelType::STENCIL_3POINT;
    config.dtype = dtype;
    config.size = size;
    config.iterations = size < 10000 ? 1000 : 100;
    config.warmup = 10;
    
    ExperimentResult result = run_single_experiment(config);
    
    file << "STENCIL_3POINT," 
         << (dtype == DataType::FLOAT32 ? "float32" : "float64") << ","
         << size << ","
         << result.scalar_time << ","
         << result.simd_time << ","
         << result.speedup << ","
         << result.gflops << ","
         << result.bandwidth_gbs << "\n";
    
    file.flush();
}

void ExperimentRunner::run_locality_sweep(const std::string& output_csv) {
    std::cout << "Running locality sweep experiment..." << std::endl;
    
    std::ofstream file(output_csv);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_csv << std::endl;
        return;
    }
    
    file << "Kernel,DataType,Size,ScalarTime,SIMDTime,Speedup,GFLOPs,BandwidthGBs,CacheLevel\n";
    
    std::vector<size_t> sizes = generate_sizes_for_cache_levels();
    DataType dtype = DataType::FLOAT32; // Focus on float32 for locality study
    KernelType kernel = KernelType::SAXPY; // Use SAXPY as representative kernel
    
    for (size_t size : sizes) {
        ExperimentConfig config;
        config.kernel = kernel;
        config.dtype = dtype;
        config.size = size;
        
        // Adjust iterations based on problem size to get measurable times
        if (size <= 16384) config.iterations = 1000;
        else if (size <= 131072) config.iterations = 500;
        else if (size <= 1048576) config.iterations = 100;
        else config.iterations = 50;
        
        config.warmup = 10;
        
        ExperimentResult result = run_single_experiment(config);
        
        // Determine cache level based on size
        std::string cache_level;
        if (size <= 32768) cache_level = "L1";
        else if (size <= 262144) cache_level = "L2";
        else if (size <= 8388608) cache_level = "L3";
        else cache_level = "DRAM";
        
        file << "SAXPY,float32,"
             << size << ","
             << result.scalar_time << ","
             << result.simd_time << ","
             << result.speedup << ","
             << result.gflops << ","
             << result.bandwidth_gbs << ","
             << cache_level << "\n";
        
        file.flush();
        std::cout << "  Size: " << size << " (" << cache_level << ") - Speedup: " << result.speedup 
                  << " GFLOPS: " << result.gflops << std::endl;
    }
    
    file.close();
    std::cout << "Locality sweep completed. Results saved to " << output_csv << std::endl;
}

void ExperimentRunner::run_alignment_study(const std::string& output_csv) {
    std::cout << "Running alignment study experiment..." << std::endl;
    
    std::ofstream file(output_csv);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_csv << std::endl;
        return;
    }
    
    file << "Kernel,DataType,Size,Alignment,HasTail,ScalarTime,SIMDTime,Speedup,GFLOPs,BandwidthGBs\n";
    
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536};
    DataType dtype = DataType::FLOAT32;
    KernelType kernel = KernelType::SAXPY;
    
    // Test different alignment scenarios
    for (size_t size : sizes) {
        // Test aligned vs misaligned
        for (bool aligned : {true, false}) {
            // Test with and without vector tail (sizes not multiple of 8 for float32 with 256-bit AVX)
            for (bool has_tail : {true, false}) {
                size_t test_size = size;
                if (has_tail) {
                    test_size = size + 3; // Make size not multiple of 8
                }
                
                ExperimentConfig config;
                config.kernel = kernel;
                config.dtype = dtype;
                config.size = test_size;
                config.aligned = aligned;
                config.iterations = test_size < 10000 ? 1000 : 100;
                config.warmup = 10;
                
                ExperimentResult result = run_single_experiment(config);
                
                file << "SAXPY,float32,"
                     << test_size << ","
                     << (aligned ? "aligned" : "misaligned") << ","
                     << (has_tail ? "yes" : "no") << ","
                     << result.scalar_time << ","
                     << result.simd_time << ","
                     << result.speedup << ","
                     << result.gflops << ","
                     << result.bandwidth_gbs << "\n";
                
                file.flush();
                
                std::cout << "  Size: " << test_size 
                          << " Aligned: " << (aligned ? "yes" : "no")
                          << " Tail: " << (has_tail ? "yes" : "no")
                          << " - Speedup: " << result.speedup << std::endl;
            }
        }
    }
    
    file.close();
    std::cout << "Alignment study completed. Results saved to " << output_csv << std::endl;
}

void ExperimentRunner::run_stride_study(const std::string& output_csv) {
    std::cout << "Running stride study experiment..." << std::endl;
    
    std::ofstream file(output_csv);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_csv << std::endl;
        return;
    }
    
    file << "Kernel,DataType,Size,Stride,ScalarTime,SIMDTime,Speedup,GFLOPs,BandwidthGBs,Efficiency\n";
    
    std::vector<size_t> strides = {1, 2, 4, 8, 16};
    std::vector<size_t> sizes = {8192, 32768, 131072};
    DataType dtype = DataType::FLOAT32;
    KernelType kernel = KernelType::SAXPY;
    
    for (size_t size : sizes) {
        for (size_t stride : strides) {
            ExperimentConfig config;
            config.kernel = kernel;
            config.dtype = dtype;
            config.size = size;
            config.stride = stride;
            config.iterations = size < 10000 ? 1000 : 100;
            config.warmup = 10;
            
            ExperimentResult result = run_single_experiment(config);
            
            // Calculate efficiency relative to unit stride
            double efficiency = (stride == 1) ? 1.0 : (result.gflops / result.gflops);
            // Note: For proper efficiency, we'd need to run unit stride as reference
            // This is simplified - in practice you'd compare against stride=1 baseline
            
            file << "SAXPY,float32,"
                 << size << ","
                 << stride << ","
                 << result.scalar_time << ","
                 << result.simd_time << ","
                 << result.speedup << ","
                 << result.gflops << ","
                 << result.bandwidth_gbs << ","
                 << efficiency << "\n";
            
            file.flush();
            
            std::cout << "  Size: " << size 
                      << " Stride: " << stride
                      << " - Speedup: " << result.speedup 
                      << " GFLOPS: " << result.gflops << std::endl;
        }
    }
    
    file.close();
    std::cout << "Stride study completed. Results saved to " << output_csv << std::endl;
}

void ExperimentRunner::run_datatype_comparison(const std::string& output_csv) {
    std::cout << "Running data type comparison experiment..." << std::endl;
    
    std::ofstream file(output_csv);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_csv << std::endl;
        return;
    }
    
    file << "Kernel,DataType,Size,VectorWidth,ScalarTime,SIMDTime,Speedup,GFLOPs,BandwidthGBs,ArithmeticIntensity\n";
    
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144};
    std::vector<DataType> dtypes = {DataType::FLOAT32, DataType::FLOAT64, DataType::INT32};
    KernelType kernel = KernelType::SAXPY;
    
    for (DataType dtype : dtypes) {
        for (size_t size : sizes) {
            ExperimentConfig config;
            config.kernel = kernel;
            config.dtype = dtype;
            config.size = size;
            config.iterations = size < 10000 ? 1000 : 100;
            config.warmup = 10;
            
            ExperimentResult result = run_single_experiment(config);
            
            // Calculate vector width (lanes) based on data type and typical SIMD width
            size_t vector_width = 0;
            if (dtype == DataType::FLOAT32) vector_width = 8;  // 256-bit AVX = 8 floats
            else if (dtype == DataType::FLOAT64) vector_width = 4; // 256-bit AVX = 4 doubles
            else if (dtype == DataType::INT32) vector_width = 8;   // 256-bit AVX = 8 int32
            
            // Calculate arithmetic intensity (FLOPs/byte)
            double arithmetic_intensity = 0.0;
            if (config.kernel == KernelType::SAXPY) {
                // 2 FLOPs per element, 24 bytes per element (3 arrays * 4/8 bytes)
                size_t bytes_per_element = (dtype == DataType::FLOAT64) ? 8 : 4;
                arithmetic_intensity = 2.0 / (3.0 * bytes_per_element);
            }
            
            file << "SAXPY,"
                 << (dtype == DataType::FLOAT32 ? "float32" : 
                     dtype == DataType::FLOAT64 ? "float64" : "int32") << ","
                 << size << ","
                 << vector_width << ","
                 << result.scalar_time << ","
                 << result.simd_time << ","
                 << result.speedup << ","
                 << result.gflops << ","
                 << result.bandwidth_gbs << ","
                 << arithmetic_intensity << "\n";
            
            file.flush();
            
            std::cout << "  DataType: " 
                      << (dtype == DataType::FLOAT32 ? "float32" : 
                          dtype == DataType::FLOAT64 ? "float64" : "int32")
                      << " Size: " << size
                      << " - Speedup: " << result.speedup 
                      << " GFLOPS: " << result.gflops << std::endl;
        }
    }
    
    file.close();
    std::cout << "Data type comparison completed. Results saved to " << output_csv << std::endl;
}