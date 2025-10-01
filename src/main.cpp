#include "experiments.h"
#include <iostream>
#include <string>

void print_usage() {
    std::cout << "SIMD Profiling Experiments\n";
    std::cout << "Usage:\n";
    std::cout << "  simd_profiling [experiment] [output_csv]\n";
    std::cout << "\nExperiments:\n";
    std::cout << "  baseline    - Baseline scalar vs SIMD comparison\n";
    std::cout << "  locality    - Cache locality sweep\n";
    std::cout << "  alignment   - Alignment and tail handling study\n";
    std::cout << "  stride      - Stride access patterns\n";
    std::cout << "  datatype    - Data type comparison\n";
    std::cout << "  all         - Run all experiments\n";
    std::cout << "\nExamples:\n";
    std::cout << "  simd_profiling baseline data/baseline.csv\n";
    std::cout << "  simd_profiling all\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string experiment = argv[1];
    std::string output_csv;
    
    if (argc >= 3) {
        output_csv = argv[2];
    } else {
        // Default output file
        output_csv = "data/" + experiment + ".csv";
    }
    
    ExperimentRunner runner;
    
    try {
        if (experiment == "baseline") {
            runner.run_baseline_comparison(output_csv);
        } else if (experiment == "locality") {
            runner.run_locality_sweep(output_csv);
        } else if (experiment == "alignment") {
            runner.run_alignment_study(output_csv);
        } else if (experiment == "stride") {
            runner.run_stride_study(output_csv);
        } else if (experiment == "datatype") {
            runner.run_datatype_comparison(output_csv);
        } else if (experiment == "all") {
            std::cout << "Running all experiments...\n";
            runner.run_baseline_comparison("data/baseline.csv");
            runner.run_locality_sweep("data/locality.csv");
            runner.run_alignment_study("data/alignment.csv");
            runner.run_stride_study("data/stride.csv");
            runner.run_datatype_comparison("data/datatype.csv");
            std::cout << "All experiments completed.\n";
        } else {
            std::cerr << "Unknown experiment: " << experiment << std::endl;
            print_usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error running experiment: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}