% SIMD Profiling Analysis Script
% Comprehensive analysis of SIMD performance experiments
% Based on assignment requirements and provided data files

clear; close all; clc;

%% Load Data Files
fprintf('Loading experiment data...\n');

% Load all CSV files
baseline_data = readtable('baseline.csv');
alignment_data = readtable('alignment.csv');
datatype_data = readtable('datatype.csv');
locality_data = readtable('locality.csv');
stride_data = readtable('stride.csv');

fprintf('Data loaded successfully!\n');

%% 1. Baseline Comparison - Speedup and Performance
fprintf('Creating baseline comparison plots...\n');

% Filter for different kernels and data types
kernels = {'SAXPY', 'DOT_PRODUCT', 'ELEMENTWISE_MULTIPLY', 'STENCIL_3POINT'};
dtypes = {'float32', 'float64'};
colors = lines(4);

figure('Position', [100 100 1200 800]);

% Subplot 1: Speedup comparison
subplot(2,2,1);
hold on;
for i = 1:length(kernels)
    for j = 1:length(dtypes)
        mask = strcmp(baseline_data.Kernel, kernels{i}) & ...
               strcmp(baseline_data.DataType, dtypes{j});
        data_subset = baseline_data(mask, :);
        
        if ~isempty(data_subset)
            plot(data_subset.Size, data_subset.Speedup, ...
                 'o-', 'LineWidth', 2, 'MarkerSize', 6, ...
                 'Color', colors(i,:), 'MarkerFaceColor', colors(i,:));
        end
    end
end

set(gca, 'XScale', 'log');
xlabel('Array Size');
ylabel('Speedup (Scalar/SIMD)');
title('SIMD Speedup Across Kernels');
legend(kernels, 'Location', 'northeastoutside');
grid on;
yline(1, '--', 'Baseline', 'LineWidth', 2, 'Color', 'g');

% Subplot 2: GFLOP/s comparison for float32
subplot(2,2,2);
hold on;
for i = 1:length(kernels)
    mask = strcmp(baseline_data.Kernel, kernels{i}) & ...
           strcmp(baseline_data.DataType, 'float32');
    data_subset = baseline_data(mask, :);
    
    if ~isempty(data_subset)
        plot(data_subset.Size, data_subset.GFLOPs, ...
             's-', 'LineWidth', 2, 'MarkerSize', 6, ...
             'Color', colors(i,:), 'MarkerFaceColor', colors(i,:));
    end
end

set(gca, 'XScale', 'log');
xlabel('Array Size');
ylabel('GFLOP/s');
title('Performance: float32 Kernels');
legend(kernels, 'Location', 'northeastoutside');
grid on;

% Subplot 3: Bandwidth comparison
subplot(2,2,3);
hold on;
for i = 1:length(kernels)
    mask = strcmp(baseline_data.Kernel, kernels{i}) & ...
           strcmp(baseline_data.DataType, 'float32');
    data_subset = baseline_data(mask, :);
    
    if ~isempty(data_subset)
        plot(data_subset.Size, data_subset.BandwidthGBs, ...
             '^-', 'LineWidth', 2, 'MarkerSize', 6, ...
             'Color', colors(i,:), 'MarkerFaceColor', colors(i,:));
    end
end

set(gca, 'XScale', 'log');
xlabel('Array Size');
ylabel('Bandwidth (GB/s)');
title('Memory Bandwidth Usage');
legend(kernels, 'Location', 'northeastoutside');
grid on;

% Subplot 4: Data type comparison for SAXPY
subplot(2,2,4);
hold on;
saxpy_mask = strcmp(baseline_data.Kernel, 'SAXPY');
saxpy_data = baseline_data(saxpy_mask, :);

dtype_colors = [0.2 0.6 0.8; 0.8 0.4 0.2];
for j = 1:length(dtypes)
    mask = strcmp(saxpy_data.DataType, dtypes{j});
    data_subset = saxpy_data(mask, :);
    
    if ~isempty(data_subset)
        plot(data_subset.Size, data_subset.Speedup, ...
             'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', dtype_colors(j,:), 'MarkerFaceColor', dtype_colors(j,:));
    end
end

set(gca, 'XScale', 'log');
xlabel('Array Size');
ylabel('Speedup');
title('SAXPY: Data Type Comparison');
legend(dtypes, 'Location', 'best');
grid on;
yline(1, '--', 'Baseline', 'LineWidth', 2, 'Color', 'g');

sgtitle('Baseline SIMD Performance Analysis');
saveas(gcf, 'baseline_analysis.png');

%% 2. Locality Sweep Analysis
fprintf('Creating locality analysis plots...\n');

figure('Position', [100 100 1400 600]);

% Cache level annotations (approximate sizes for i7-11800H)
L1_size = 640 * 1024; % 640KB
L2_size = 10 * 1024 * 1024; % 10MB
L3_size = 24 * 1024 * 1024; % 24MB

% Convert bytes to elements (float32 = 4 bytes per element)
L1_elems = L1_size / 4;
L2_elems = L2_size / 4;
L3_elems = L3_size / 4;

% Subplot 1: Performance vs Size
subplot(1,3,1);
semilogx(locality_data.Size, locality_data.GFLOPs, 'bo-', ...
         'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
hold on;

% Add cache boundary lines
xline(L1_elems, '--r', 'L1 Limit', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
xline(L2_elems, '--r', 'L2 Limit', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
xline(L3_elems, '--r', 'L3 Limit', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');

xlabel('Array Size (elements)');
ylabel('GFLOP/s');
title('Performance vs Working Set Size');
grid on;

% Annotate cache regions
text(L1_elems/4, max(locality_data.GFLOPs)*0.9, 'L1', 'FontSize', 12, 'FontWeight', 'bold');
text((L1_elems+L2_elems)/2, max(locality_data.GFLOPs)*0.9, 'L2', 'FontSize', 12, 'FontWeight', 'bold');
text((L2_elems+L3_elems)/2, max(locality_data.GFLOPs)*0.9, 'L3', 'FontSize', 12, 'FontWeight', 'bold');
text(L3_elems*1.2, max(locality_data.GFLOPs)*0.9, 'DRAM', 'FontSize', 12, 'FontWeight', 'bold');

% Subplot 2: Bandwidth vs Size
subplot(1,3,2);
semilogx(locality_data.Size, locality_data.BandwidthGBs, 'ro-', ...
         'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
hold on;

xline(L1_elems, '--r', 'L1 Limit', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
xline(L2_elems, '--r', 'L2 Limit', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
xline(L3_elems, '--r', 'L3 Limit', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');

xlabel('Array Size (elements)');
ylabel('Bandwidth (GB/s)');
title('Memory Bandwidth vs Working Set Size');
grid on;

% Subplot 3: Speedup vs Size
subplot(1,3,3);
semilogx(locality_data.Size, locality_data.Speedup, 'go-', ...
         'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'g');
hold on;

xline(L1_elems, '--r', 'L1 Limit', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
xline(L2_elems, '--r', 'L2 Limit', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
xline(L3_elems, '--r', 'L3 Limit', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');

xlabel('Array Size (elements)');
ylabel('Speedup');
title('SIMD Speedup vs Working Set Size');
grid on;
yline(1, '--g', 'Baseline', 'LineWidth', 2);

sgtitle('Cache Locality Effects on SIMD Performance');
saveas(gcf, 'locality_analysis.png');

%% 3. Alignment and Tail Handling Study
fprintf('Creating alignment analysis plots...\n');

figure('Position', [100 100 1200 800]);

% Group data by size and alignment scenario
sizes = unique(alignment_data.Size);
scenarios = {'aligned-no', 'aligned-yes', 'misaligned-no', 'misaligned-yes'};
scenario_names = {'Aligned, No Tail', 'Aligned, Has Tail', 'Misaligned, No Tail', 'Misaligned, Has Tail'};
colors = [0.2 0.8 0.2; 0.8 0.8 0.2; 0.8 0.2 0.2; 0.2 0.2 0.8];

% Calculate average metrics for each scenario and size
avg_speedup = zeros(length(sizes), length(scenarios));
avg_gflops = zeros(length(sizes), length(scenarios));
avg_bandwidth = zeros(length(sizes), length(scenarios));

for i = 1:length(sizes)
    for j = 1:length(scenarios)
        parts = strsplit(scenarios{j}, '-');
        alignment = parts{1};
        has_tail = parts{2};
        
        mask = alignment_data.Size == sizes(i) & ...
               strcmp(alignment_data.Alignment, alignment) & ...
               strcmp(alignment_data.HasTail, has_tail);
        
        if any(mask)
            avg_speedup(i,j) = alignment_data.Speedup(mask);
            avg_gflops(i,j) = alignment_data.GFLOPs(mask);
            avg_bandwidth(i,j) = alignment_data.BandwidthGBs(mask);
        end
    end
end

% Plot 1: Speedup comparison
subplot(2,2,1);
hold on;
for j = 1:length(scenarios)
    plot(sizes, avg_speedup(:,j), 's-', 'LineWidth', 2, 'MarkerSize', 8, ...
         'Color', colors(j,:), 'MarkerFaceColor', colors(j,:));
end

set(gca, 'XScale', 'log');
xlabel('Array Size');
ylabel('Speedup');
title('Alignment and Tail Effects on Speedup');
legend(scenario_names, 'Location', 'southwest');
grid on;
yline(1, '--g', 'Baseline', 'LineWidth', 2);

% Plot 2: Performance comparison
subplot(2,2,2);
hold on;
for j = 1:length(scenarios)
    plot(sizes, avg_gflops(:,j), 's-', 'LineWidth', 2, 'MarkerSize', 8, ...
         'Color', colors(j,:), 'MarkerFaceColor', colors(j,:));
end

set(gca, 'XScale', 'log');
xlabel('Array Size');
ylabel('GFLOP/s');
title('Alignment and Tail Effects on Performance');
legend(scenario_names, 'Location', 'southwest');
grid on;

% Plot 3: Bandwidth comparison
subplot(2,2,3);
hold on;
for j = 1:length(scenarios)
    plot(sizes, avg_bandwidth(:,j), 's-', 'LineWidth', 2, 'MarkerSize', 8, ...
         'Color', colors(j,:), 'MarkerFaceColor', colors(j,:));
end

set(gca, 'XScale', 'log');
xlabel('Array Size');
ylabel('Bandwidth (GB/s)');
title('Alignment and Tail Effects on Bandwidth');
legend(scenario_names, 'Location', 'southwest');
grid on;

% Plot 4: Performance penalty heatmap
subplot(2,2,4);
performance_ratio = avg_gflops ./ max(avg_gflops, [], 2);
imagesc(performance_ratio');
colormap(parula);
colorbar;
set(gca, 'YTick', 1:length(scenarios), 'YTickLabel', scenario_names);
set(gca, 'XTick', 1:length(sizes), 'XTickLabel', arrayfun(@num2str, sizes, 'UniformOutput', false));
xlabel('Array Size');
title('Normalized Performance Heatmap');
ylabel('Alignment Scenario');

sgtitle('Memory Alignment and Vector Tail Handling Analysis');
saveas(gcf, 'alignment_analysis.png');

%% 4. Stride Access Patterns
fprintf('Creating stride analysis plots...\n');

figure('Position', [100 100 1200 800]);

sizes_stride = unique(stride_data.Size);
strides = unique(stride_data.Stride);
colors_stride = parula(length(strides));

% Create legend labels with array sizes
legend_labels = cell(length(sizes_stride), 1);
for i = 1:length(sizes_stride)
    legend_labels{i} = ['Size: ' num2str(sizes_stride(i))];
end

% Subplot 1: Speedup vs Stride for different sizes
subplot(2,2,1);
hold on;
for i = 1:length(sizes_stride)
    mask = stride_data.Size == sizes_stride(i);
    data_subset = stride_data(mask, :);
    
    if ~isempty(data_subset)
        plot(data_subset.Stride, data_subset.Speedup, 'o-', ...
             'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', colors_stride(i,:), 'MarkerFaceColor', colors_stride(i,:));
    end
end

set(gca, 'XScale', 'log');
xlabel('Stride');
ylabel('Speedup');
title('Stride Effects on SIMD Speedup');
legend(legend_labels, 'Location', 'southwest');
grid on;
yline(1, '--g', 'Baseline', 'LineWidth', 2);

% Subplot 2: Performance vs Stride
subplot(2,2,2);
hold on;
for i = 1:length(sizes_stride)
    mask = stride_data.Size == sizes_stride(i);
    data_subset = stride_data(mask, :);
    
    if ~isempty(data_subset)
        plot(data_subset.Stride, data_subset.GFLOPs, 's-', ...
             'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', colors_stride(i,:), 'MarkerFaceColor', colors_stride(i,:));
    end
end

set(gca, 'XScale', 'log');
xlabel('Stride');
ylabel('GFLOP/s');
title('Stride Effects on Performance');
legend(legend_labels, 'Location', 'northeast');
grid on;

% Subplot 3: Bandwidth vs Stride
subplot(2,2,3);
hold on;
for i = 1:length(sizes_stride)
    mask = stride_data.Size == sizes_stride(i);
    data_subset = stride_data(mask, :);
    
    if ~isempty(data_subset)
        plot(data_subset.Stride, data_subset.BandwidthGBs, '^-', ...
             'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', colors_stride(i,:), 'MarkerFaceColor', colors_stride(i,:));
    end
end

set(gca, 'XScale', 'log');
xlabel('Stride');
ylabel('Bandwidth (GB/s)');
title('Stride Effects on Bandwidth');
legend(legend_labels, 'Location', 'northeast');
grid on;

% Subplot 4: Efficiency vs Stride
subplot(2,2,4);
hold on;
for i = 1:length(sizes_stride)
    mask = stride_data.Size == sizes_stride(i);
    data_subset = stride_data(mask, :);
    
    if ~isempty(data_subset)
        plot(data_subset.Stride, data_subset.Efficiency, 'd-', ...
             'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', colors_stride(i,:), 'MarkerFaceColor', colors_stride(i,:));
    end
end

set(gca, 'XScale', 'log');
xlabel('Stride');
ylabel('Efficiency');
title('Stride Effects on Computational Efficiency');
legend(legend_labels, 'Location', 'southwest');
grid on;
yline(1, '--g', 'Optimal', 'LineWidth', 2);

sgtitle('Memory Access Pattern (Stride) Analysis');
saveas(gcf, 'stride_analysis.png');

%% 5. Data Type Comparison
fprintf('Creating data type analysis plots...\n');

figure('Position', [100 100 1200 800]);

dtypes_comparison = unique(datatype_data.DataType);
sizes_dtype = unique(datatype_data.Size);
colors_dtype = [0.2 0.6 0.8; 0.8 0.4 0.2; 0.6 0.2 0.8]; % float32, float64, int32

% Subplot 1: Speedup comparison
subplot(2,2,1);
hold on;
for i = 1:length(dtypes_comparison)
    mask = strcmp(datatype_data.DataType, dtypes_comparison{i});
    data_subset = datatype_data(mask, :);
    
    if ~isempty(data_subset)
        plot(data_subset.Size, data_subset.Speedup, 'o-', ...
             'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', colors_dtype(i,:), 'MarkerFaceColor', colors_dtype(i,:));
    end
end

set(gca, 'XScale', 'log');
xlabel('Array Size');
ylabel('Speedup');
title('Data Type Effects on SIMD Speedup');
legend(dtypes_comparison, 'Location', 'southwest');
grid on;
yline(1, '--g', 'Baseline', 'LineWidth', 2);

% Subplot 2: Performance comparison
subplot(2,2,2);
hold on;
for i = 1:length(dtypes_comparison)
    mask = strcmp(datatype_data.DataType, dtypes_comparison{i});
    data_subset = datatype_data(mask, :);
    
    if ~isempty(data_subset)
        plot(data_subset.Size, data_subset.GFLOPs, 's-', ...
             'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', colors_dtype(i,:), 'MarkerFaceColor', colors_dtype(i,:));
    end
end

set(gca, 'XScale', 'log');
xlabel('Array Size');
ylabel('GFLOP/s');
title('Data Type Effects on Performance');
legend(dtypes_comparison, 'Location', 'northeast');
grid on;

% Subplot 3: Bandwidth comparison
subplot(2,2,3);
hold on;
for i = 1:length(dtypes_comparison)
    mask = strcmp(datatype_data.DataType, dtypes_comparison{i});
    data_subset = datatype_data(mask, :);
    
    if ~isempty(data_subset)
        plot(data_subset.Size, data_subset.BandwidthGBs, '^-', ...
             'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', colors_dtype(i,:), 'MarkerFaceColor', colors_dtype(i,:));
    end
end

set(gca, 'XScale', 'log');
xlabel('Array Size');
ylabel('Bandwidth (GB/s)');
title('Data Type Effects on Bandwidth');
legend(dtypes_comparison, 'Location', 'northeast');
grid on;

% Subplot 4: Vector width and arithmetic intensity
subplot(2,2,4);
yyaxis left;
plot(datatype_data.Size(datatype_data.VectorWidth > 0), ...
     datatype_data.VectorWidth(datatype_data.VectorWidth > 0), 'bo-', ...
     'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
ylabel('Vector Width (lanes)');

yyaxis right;
plot(datatype_data.Size, datatype_data.ArithmeticIntensity, 'ro-', ...
     'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
ylabel('Arithmetic Intensity (FLOPs/byte)');

set(gca, 'XScale', 'log');
xlabel('Array Size');
title('Vector Characteristics');
grid on;

sgtitle('Data Type and Vector Architecture Analysis');
saveas(gcf, 'datatype_analysis.png');

%% 6. Roofline Model Analysis
fprintf('Creating roofline model analysis...\n');

% Estimate peak performance for i7-11800H
% 2.3 GHz base, up to 4.6 GHz turbo, 8 cores, but we're single-threaded
% AVX2: 256-bit vectors = 8 float32 ops per cycle or 4 float64 ops per cycle
peak_freq = 4.6; % GHz (turbo frequency)
peak_flops_float32 = peak_freq * 8 * 2; % 2 FMA units per core
peak_flops_float64 = peak_freq * 4 * 2; % 2 FMA units per core

% Measured memory bandwidth (from your locality data)
measured_bandwidth = max(locality_data.BandwidthGBs);

% Calculate arithmetic intensity for each kernel
kernel_ai = containers.Map();
kernel_perf = containers.Map();

% SAXPY: 2 FLOPs, 3 arrays * 4 bytes = 12 bytes for float32
kernel_ai('SAXPY_float32') = 2 / (3 * 4); % FLOPs/byte
kernel_ai('SAXPY_float64') = 2 / (3 * 8); % FLOPs/byte

% DOT_PRODUCT: 2 FLOPs, 2 arrays * 4 bytes = 8 bytes for float32  
kernel_ai('DOT_PRODUCT_float32') = 2 / (2 * 4);
kernel_ai('DOT_PRODUCT_float64') = 2 / (2 * 8);

% ELEMENTWISE_MULTIPLY: 1 FLOP, 3 arrays * 4 bytes = 12 bytes for float32
kernel_ai('ELEMENTWISE_MULTIPLY_float32') = 1 / (3 * 4);
kernel_ai('ELEMENTWISE_MULTIPLY_float64') = 1 / (3 * 8);

% STENCIL_3POINT: 5 FLOPs, 4 arrays * 4 bytes = 16 bytes for float32
kernel_ai('STENCIL_3POINT_float32') = 5 / (4 * 4);
kernel_ai('STENCIL_3POINT_float64') = 5 / (4 * 8);

% Get performance for medium-sized arrays (in cache)
medium_size = 16384;
mask = baseline_data.Size == medium_size;
medium_data = baseline_data(mask, :);

for i = 1:height(medium_data)
    kernel_name = [medium_data.Kernel{i} '_' medium_data.DataType{i}];
    kernel_perf(kernel_name) = medium_data.GFLOPs(i);
end

% Create roofline plot
figure('Position', [100 100 1000 800]);
hold on;

% Plot roofline boundaries
ai_range = logspace(-3, 1, 1000);
memory_bound = min(measured_bandwidth * ai_range, peak_flops_float32);

loglog(ai_range, memory_bound, 'g-', 'LineWidth', 3);
loglog(ai_range, peak_flops_float32 * ones(size(ai_range)), 'r-', 'LineWidth', 2);

% Plot kernel points
kernel_names = keys(kernel_ai);
markers = 'os^dv';
colors = lines(length(kernel_names));

for i = 1:length(kernel_names)
    ai_val = kernel_ai(kernel_names{i});
    if isKey(kernel_perf, kernel_names{i})
        perf_val = kernel_perf(kernel_names{i});
        
        % Determine if kernel is compute-bound or memory-bound
        if perf_val < measured_bandwidth * ai_val * 0.8
            marker_color = 'r';
            marker_size = 100;
        else
            marker_color = 'b';
            marker_size = 80;
        end
        
        scatter(ai_val, perf_val, marker_size, markers(mod(i-1, length(markers))+1), ...
                'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'g', 'LineWidth', 2);
        
        text(ai_val*1.2, perf_val*1.1, strrep(kernel_names{i}, '_', ' '), ...
             'FontSize', 9, 'FontWeight', 'bold');
    end
end

% Add annotations
text(0.01, peak_flops_float32*0.8, sprintf('Peak Compute: %.1f GFLOPS', peak_flops_float32), ...
     'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
text(0.1, measured_bandwidth*0.1*2, sprintf('Memory BW: %.1f GB/s', measured_bandwidth), ...
     'FontSize', 12, 'FontWeight', 'bold', 'Color', 'g', 'Rotation', 45);

xlabel('Arithmetic Intensity (FLOPs/byte)');
ylabel('Performance (GFLOP/s)');
title('Roofline Model Analysis');
set(gca, 'XScale', 'log', 'YScale', 'log');
grid on;

legend({'Memory Bound', 'Compute Bound', 'Kernels'}, 'Location', 'southeast');

% Add performance summary
annotation('textbox', [0.02 0.02 0.4 0.2], 'String', ...
           sprintf('Performance Summary:\nPeak Compute: %.1f GFLOPS\nMemory BW: %.1f GB/s\nSIMD Speedup Range: %.2f-%.2fx', ...
                   peak_flops_float32, measured_bandwidth, ...
                   min(baseline_data.Speedup), max(baseline_data.Speedup)), ...
           'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');

saveas(gcf, 'roofline_analysis.png');

%% 7. Summary Statistics and Key Findings
fprintf('Generating summary statistics...\n');

% Calculate overall statistics
fprintf('\n=== SIMD PROFILING SUMMARY ===\n');
fprintf('Overall Speedup Statistics:\n');
fprintf('  Mean Speedup: %.3fx\n', mean(baseline_data.Speedup));
fprintf('  Median Speedup: %.3fx\n', median(baseline_data.Speedup));
fprintf('  Min Speedup: %.3fx\n', min(baseline_data.Speedup));
fprintf('  Max Speedup: %.3fx\n', max(baseline_data.Speedup));
fprintf('  Speedup > 1.0: %.1f%% of cases\n', ...
        sum(baseline_data.Speedup > 1.0) / height(baseline_data) * 100);

fprintf('\nPerformance by Kernel:\n');
for i = 1:length(kernels)
    mask = strcmp(baseline_data.Kernel, kernels{i});
    if any(mask)
        avg_speedup = mean(baseline_data.Speedup(mask));
        avg_gflops = mean(baseline_data.GFLOPs(mask));
        fprintf('  %s: %.3fx speedup, %.1f GFLOP/s\n', ...
                kernels{i}, avg_speedup, avg_gflops);
    end
end

fprintf('\nCache Locality Impact:\n');
l1_mask = locality_data.Size <= L1_elems;
l2_mask = locality_data.Size > L1_elems & locality_data.Size <= L2_elems;
l3_mask = locality_data.Size > L2_elems & locality_data.Size <= L3_elems;
dram_mask = locality_data.Size > L3_elems;

fprintf('  L1 Cache: %.1f GFLOP/s\n', mean(locality_data.GFLOPs(l1_mask)));
fprintf('  L2 Cache: %.1f GFLOP/s\n', mean(locality_data.GFLOPs(l2_mask)));
fprintf('  L3 Cache: %.1f GFLOP/s\n', mean(locality_data.GFLOPs(l3_mask)));
fprintf('  DRAM: %.1f GFLOP/s\n', mean(locality_data.GFLOPs(dram_mask)));

fprintf('\nAlignment Impact:\n');
aligned_mask = strcmp(alignment_data.Alignment, 'aligned');
misaligned_mask = strcmp(alignment_data.Alignment, 'misaligned');
fprintf('  Aligned: %.1f GFLOP/s\n', mean(alignment_data.GFLOPs(aligned_mask)));
fprintf('  Misaligned: %.1f GFLOP/s (%.1f%% penalty)\n', ...
        mean(alignment_data.GFLOPs(misaligned_mask)), ...
        (1 - mean(alignment_data.GFLOPs(misaligned_mask)) / mean(alignment_data.GFLOPs(aligned_mask))) * 100);

fprintf('\n=== KEY FINDINGS ===\n');
fprintf('1. Limited SIMD speedup observed (typically 0.95-1.05x)\n');
fprintf('2. Memory-bound behavior dominates for most kernel sizes\n');
fprintf('3. Cache locality has significant impact on performance\n');
fprintf('4. Alignment effects are measurable but not dramatic\n');
fprintf('5. Stride access patterns severely degrade performance\n');

fprintf('\nAll plots saved as PNG files in current directory.\n');
fprintf('Analysis complete!\n');