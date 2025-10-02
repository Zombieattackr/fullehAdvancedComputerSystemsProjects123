function analyze_simd_experiments()
    % SIMD Profiling Data Analysis
    % Analyzes multiple CSV files for statistical analysis and creates plots
    
    close all;
    clc;
    
    % Set better color scheme for dark background
    colors = get_plot_colors();
    
    % Create output directory for plots
    if ~exist('plots', 'dir')
        mkdir('plots');
    end
    
    % Analyze all experiment types
    analyze_baseline_comparison(colors);
    analyze_locality_sweep(colors);
    analyze_alignment_study(colors);
    analyze_stride_study(colors);
    analyze_datatype_comparison(colors);
    
    fprintf('Analysis complete! Plots saved to ./plots/\n');
end

function colors = get_plot_colors()
    % Define a color scheme that works well on dark backgrounds
    colors = struct();
    colors.blue = [0.2, 0.6, 1.0];
    colors.red = [1.0, 0.4, 0.4];
    colors.green = [0.4, 0.8, 0.4];
    colors.orange = [1.0, 0.6, 0.2];
    colors.purple = [0.7, 0.4, 0.9];
    colors.cyan = [0.3, 0.8, 0.8];
    colors.yellow = [1.0, 0.8, 0.2];
    colors.gray = [0.6, 0.6, 0.6];
    colors.white = [1.0, 1.0, 1.0];
    colors.black = [0.1, 0.1, 0.1];
end

function analyze_baseline_comparison(colors)
    fprintf('Analyzing baseline comparison...\n');
    
    % Load all baseline CSV files
    files = {'baseline1.csv'}; % Start with just one file for testing
    all_data = load_multiple_files(files);
    
    if isempty(all_data)
        fprintf('No baseline files found. Skipping.\n');
        return;
    end
    
    % Convert categorical arrays to cell arrays of strings
    if iscategorical(all_data.Kernel)
        kernels = cellstr(unique(all_data.Kernel));
    else
        kernels = unique(all_data.Kernel);
    end
    
    if iscategorical(all_data.DataType)
        dtypes = cellstr(unique(all_data.DataType));
    else
        dtypes = unique(all_data.DataType);
    end
    
    sizes = unique(all_data.Size);
    
    % Create speedup plot
    figure('Position', [100, 100, 1200, 800]);
    
    for k = 1:length(kernels)
        kernel = kernels{k};
        
        for d = 1:length(dtypes)
            dtype = dtypes{d};
            
            % Get data for this kernel and data type
            if iscategorical(all_data.Kernel)
                kernel_mask = strcmp(cellstr(all_data.Kernel), kernel);
            else
                kernel_mask = strcmp(all_data.Kernel, kernel);
            end
            
            if iscategorical(all_data.DataType)
                dtype_mask = strcmp(cellstr(all_data.DataType), dtype);
            else
                dtype_mask = strcmp(all_data.DataType, dtype);
            end
            
            mask = kernel_mask & dtype_mask;
            sub_data = all_data(mask, :);
            
            if height(sub_data) == 0
                continue;
            end
            
            % Calculate statistics across runs
            unique_sizes = unique(sub_data.Size);
            median_speedup = zeros(size(unique_sizes));
            q25_speedup = zeros(size(unique_sizes));
            q75_speedup = zeros(size(unique_sizes));
            
            for s = 1:length(unique_sizes)
                size_mask = sub_data.Size == unique_sizes(s);
                speedups = sub_data.Speedup(size_mask);
                median_speedup(s) = median(speedups);
                q25_speedup(s) = quantile(speedups, 0.25);
                q75_speedup(s) = quantile(speedups, 0.75);
            end
            
            % Plot with error bars
            subplot(2, 2, d);
            color_idx = find(strcmp(kernels, kernel));
            color_list = [colors.blue; colors.red; colors.green; colors.orange; colors.purple];
            
            if color_idx <= length(color_list)
                line_color = color_list(color_idx, :);
            else
                line_color = colors.gray;
            end
            
            errorbar(unique_sizes, median_speedup, ...
                    median_speedup - q25_speedup, q75_speedup - median_speedup, ...
                    'o-', 'Color', line_color, 'LineWidth', 2, 'MarkerSize', 6, ...
                    'MarkerFaceColor', line_color);
            hold on;
        end
        
        title(sprintf('%s Speedup', dtype), 'Color', colors.white, 'FontSize', 12);
        xlabel('Array Size', 'Color', colors.white);
        ylabel('Speedup (Scalar/SIMD)', 'Color', colors.white);
        set(gca, 'XScale', 'log', 'Color', colors.black, 'GridColor', colors.gray);
        grid on;
        legend(kernels, 'TextColor', colors.white, 'Location', 'best');
    end
    
    sgtitle('Baseline Comparison: Scalar vs SIMD Speedup', 'Color', colors.white, 'FontSize', 14);
    save_plot('baseline_speedup.png');
    
    % Create GFLOPS comparison plot
    figure('Position', [100, 100, 1200, 800]);
    
    for k = 1:length(kernels)
        kernel = kernels{k};
        
        for d = 1:length(dtypes)
            dtype = dtypes{d};
            
            if iscategorical(all_data.Kernel)
                kernel_mask = strcmp(cellstr(all_data.Kernel), kernel);
            else
                kernel_mask = strcmp(all_data.Kernel, kernel);
            end
            
            if iscategorical(all_data.DataType)
                dtype_mask = strcmp(cellstr(all_data.DataType), dtype);
            else
                dtype_mask = strcmp(all_data.DataType, dtype);
            end
            
            mask = kernel_mask & dtype_mask;
            sub_data = all_data(mask, :);
            
            if height(sub_data) == 0
                continue;
            end
            
            unique_sizes = unique(sub_data.Size);
            median_gflops = zeros(size(unique_sizes));
            
            for s = 1:length(unique_sizes)
                size_mask = sub_data.Size == unique_sizes(s);
                gflops = sub_data.GFLOPs(size_mask);
                median_gflops(s) = median(gflops);
            end
            
            subplot(2, 2, d);
            color_idx = find(strcmp(kernels, kernel));
            color_list = [colors.blue; colors.red; colors.green; colors.orange; colors.purple];
            
            if color_idx <= length(color_list)
                line_color = color_list(color_idx, :);
            else
                line_color = colors.gray;
            end
            
            semilogx(unique_sizes, median_gflops, 'o-', 'Color', line_color, ...
                    'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', line_color);
            hold on;
        end
        
        title(sprintf('%s Performance', dtype), 'Color', colors.white, 'FontSize', 12);
        xlabel('Array Size', 'Color', colors.white);
        ylabel('GFLOPS', 'Color', colors.white);
        set(gca, 'Color', colors.black, 'GridColor', colors.gray);
        grid on;
        legend(kernels, 'TextColor', colors.white, 'Location', 'best');
    end
    
    sgtitle('Baseline Comparison: SIMD Performance (GFLOPS)', 'Color', colors.white, 'FontSize', 14);
    save_plot('baseline_gflops.png');
end

function analyze_locality_sweep(colors)
    fprintf('Analyzing locality sweep...\n');
    
    files = {'locality1.csv'}; % Start with one file
    all_data = load_multiple_files(files);
    
    if isempty(all_data)
        fprintf('No locality files found. Skipping.\n');
        return;
    end
    
    figure('Position', [100, 100, 1000, 800]);
    
    % Plot GFLOPS vs size with cache level annotations
    unique_sizes = unique(all_data.Size);
    median_gflops = zeros(size(unique_sizes));
    q25_gflops = zeros(size(unique_sizes));
    q75_gflops = zeros(size(unique_sizes));
    
    for s = 1:length(unique_sizes)
        size_mask = all_data.Size == unique_sizes(s);
        gflops = all_data.GFLOPs(size_mask);
        median_gflops(s) = median(gflops);
        q25_gflops(s) = quantile(gflops, 0.25);
        q75_gflops(s) = quantile(gflops, 0.75);
    end
    
    errorbar(unique_sizes, median_gflops, ...
             median_gflops - q25_gflops, q75_gflops - median_gflops, ...
             'o-', 'Color', colors.blue, 'LineWidth', 2, 'MarkerSize', 6, ...
             'MarkerFaceColor', colors.blue);
    
    % Annotate cache levels (adjust based on your CPU specs)
    cache_boundaries = [32*1024, 256*1024, 8*1024*1024]; % L1, L2, L3 boundaries
    cache_labels = {'L1', 'L2', 'L3', 'DRAM'};
    
    y_lim = ylim;
    for i = 1:length(cache_boundaries)
        line([cache_boundaries(i), cache_boundaries(i)], y_lim, ...
             'Color', colors.red, 'LineStyle', '--', 'LineWidth', 1);
        text(cache_boundaries(i)*0.8, y_lim(2)*0.9, cache_labels{i}, ...
             'Color', colors.white, 'FontSize', 10, 'FontWeight', 'bold');
    end
    text(cache_boundaries(end)*1.2, y_lim(2)*0.9, cache_labels{end}, ...
         'Color', colors.white, 'FontSize', 10, 'FontWeight', 'bold');
    
    xlabel('Array Size (elements)', 'Color', colors.white);
    ylabel('GFLOPS', 'Color', colors.white);
    title('Locality Sweep: Performance Across Cache Hierarchy', 'Color', colors.white, 'FontSize', 14);
    set(gca, 'XScale', 'log', 'Color', colors.black, 'GridColor', colors.gray);
    grid on;
    
    save_plot('locality_sweep.png');
    
    % Plot speedup
    figure('Position', [100, 100, 1000, 600]);
    
    median_speedup = zeros(size(unique_sizes));
    q25_speedup = zeros(size(unique_sizes));
    q75_speedup = zeros(size(unique_sizes));
    
    for s = 1:length(unique_sizes)
        size_mask = all_data.Size == unique_sizes(s);
        speedups = all_data.Speedup(size_mask);
        median_speedup(s) = median(speedups);
        q25_speedup(s) = quantile(speedups, 0.25);
        q75_speedup(s) = quantile(speedups, 0.75);
    end
    
    errorbar(unique_sizes, median_speedup, ...
             median_speedup - q25_speedup, q75_speedup - median_speedup, ...
             'o-', 'Color', colors.green, 'LineWidth', 2, 'MarkerSize', 6, ...
             'MarkerFaceColor', colors.green);
    
    xlabel('Array Size (elements)', 'Color', colors.white);
    ylabel('Speedup (Scalar/SIMD)', 'Color', colors.white);
    title('Locality Sweep: SIMD Speedup Across Cache Hierarchy', 'Color', colors.white, 'FontSize', 14);
    set(gca, 'XScale', 'log', 'Color', colors.black, 'GridColor', colors.gray);
    grid on;
    
    save_plot('locality_speedup.png');
end

function analyze_alignment_study(colors)
    fprintf('Analyzing alignment study...\n');
    
    files = {'alignment1.csv'}; % Start with one file
    all_data = load_multiple_files(files);
    
    if isempty(all_data)
        fprintf('No alignment files found. Skipping.\n');
        return;
    end
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Convert categorical data if needed
    if iscategorical(all_data.Alignment)
        alignments = cellstr(unique(all_data.Alignment));
    else
        alignments = unique(all_data.Alignment);
    end
    
    if iscategorical(all_data.HasTail)
        tails = cellstr(unique(all_data.HasTail));
    else
        tails = unique(all_data.HasTail);
    end
    
    sizes = unique(all_data.Size);
    
    for a = 1:length(alignments)
        alignment = alignments{a};
        
        for t = 1:length(tails)
            tail = tails{t};
            
            if iscategorical(all_data.Alignment)
                align_mask = strcmp(cellstr(all_data.Alignment), alignment);
            else
                align_mask = strcmp(all_data.Alignment, alignment);
            end
            
            if iscategorical(all_data.HasTail)
                tail_mask = strcmp(cellstr(all_data.HasTail), tail);
            else
                tail_mask = strcmp(all_data.HasTail, tail);
            end
            
            mask = align_mask & tail_mask;
            sub_data = all_data(mask, :);
            
            if height(sub_data) == 0
                continue;
            end
            
            median_speedup = zeros(size(sizes));
            
            for s = 1:length(sizes)
                size_mask = sub_data.Size == sizes(s);
                speedups = sub_data.Speedup(size_mask);
                median_speedup(s) = median(speedups);
            end
            
            subplot(2, 2, (a-1)*2 + t);
            
            color_idx = (a-1)*length(tails) + t;
            color_list = [colors.blue; colors.red; colors.green; colors.orange];
            
            if color_idx <= length(color_list)
                line_color = color_list(color_idx, :);
            else
                line_color = colors.gray;
            end
            
            semilogx(sizes, median_speedup, 'o-', 'Color', line_color, ...
                    'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', line_color);
            
            title(sprintf('%s, Tail: %s', alignment, tail), 'Color', colors.white, 'FontSize', 10);
            xlabel('Array Size', 'Color', colors.white);
            ylabel('Speedup', 'Color', colors.white);
            set(gca, 'Color', colors.black, 'GridColor', colors.gray);
            grid on;
            ylim([0.9, 1.1]);
        end
    end
    
    sgtitle('Alignment Study: Impact on SIMD Speedup', 'Color', colors.white, 'FontSize', 14);
    save_plot('alignment_study.png');
end

function analyze_stride_study(colors)
    fprintf('Analyzing stride study...\n');
    
    files = {'stride1.csv'}; % Start with one file
    all_data = load_multiple_files(files);
    
    if isempty(all_data)
        fprintf('No stride files found. Skipping.\n');
        return;
    end
    
    figure('Position', [100, 100, 1200, 800]);
    
    strides = unique(all_data.Stride);
    sizes = unique(all_data.Size);
    
    for s_idx = 1:length(sizes)
        size_val = sizes(s_idx);
        
        median_gflops = zeros(size(strides));
        q25_gflops = zeros(size(strides));
        q75_gflops = zeros(size(strides));
        
        for st = 1:length(strides)
            stride_val = strides(st);
            mask = all_data.Size == size_val & all_data.Stride == stride_val;
            sub_data = all_data(mask, :);
            
            if height(sub_data) > 0
                gflops = sub_data.GFLOPs;
                median_gflops(st) = median(gflops);
                q25_gflops(st) = quantile(gflops, 0.25);
                q75_gflops(st) = quantile(gflops, 0.75);
            end
        end
        
        subplot(2, 2, s_idx);
        errorbar(strides, median_gflops, ...
                 median_gflops - q25_gflops, q75_gflops - median_gflops, ...
                 'o-', 'Color', colors.purple, 'LineWidth', 2, 'MarkerSize', 6, ...
                 'MarkerFaceColor', colors.purple);
        
        title(sprintf('Size: %d', size_val), 'Color', colors.white, 'FontSize', 10);
        xlabel('Stride', 'Color', colors.white);
        ylabel('GFLOPS', 'Color', colors.white);
        set(gca, 'Color', colors.black, 'GridColor', colors.gray, 'XScale', 'log');
        grid on;
    end
    
    sgtitle('Stride Study: Impact on SIMD Performance', 'Color', colors.white, 'FontSize', 14);
    save_plot('stride_study.png');
end

function analyze_datatype_comparison(colors)
    fprintf('Analyzing data type comparison...\n');
    
    files = {'datatype1.csv'}; % Start with one file
    all_data = load_multiple_files(files);
    
    if isempty(all_data)
        fprintf('No datatype files found. Skipping.\n');
        return;
    end
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Convert categorical data if needed
    if iscategorical(all_data.DataType)
        dtypes = cellstr(unique(all_data.DataType));
    else
        dtypes = unique(all_data.DataType);
    end
    
    sizes = unique(all_data.Size);
    
    for d = 1:length(dtypes)
        dtype = dtypes{d};
        
        if iscategorical(all_data.DataType)
            mask = strcmp(cellstr(all_data.DataType), dtype);
        else
            mask = strcmp(all_data.DataType, dtype);
        end
        
        sub_data = all_data(mask, :);
        
        if height(sub_data) == 0
            continue;
        end
        
        median_speedup = zeros(size(sizes));
        q25_speedup = zeros(size(sizes));
        q75_speedup = zeros(size(sizes));
        
        for s = 1:length(sizes)
            size_mask = sub_data.Size == sizes(s);
            speedups = sub_data.Speedup(size_mask);
            median_speedup(s) = median(speedups);
            q25_speedup(s) = quantile(speedups, 0.25);
            q75_speedup(s) = quantile(speedups, 0.75);
        end
        
        color_list = [colors.blue; colors.red; colors.green];
        if d <= size(color_list, 1)
            line_color = color_list(d, :);
        else
            line_color = colors.gray;
        end
        
        errorbar(sizes, median_speedup, ...
                 median_speedup - q25_speedup, q75_speedup - median_speedup, ...
                 'o-', 'Color', line_color, 'LineWidth', 2, 'MarkerSize', 6, ...
                 'MarkerFaceColor', line_color);
        hold on;
    end
    
    xlabel('Array Size', 'Color', colors.white);
    ylabel('Speedup (Scalar/SIMD)', 'Color', colors.white);
    title('Data Type Comparison: SIMD Speedup', 'Color', colors.white, 'FontSize', 14);
    set(gca, 'XScale', 'log', 'Color', colors.black, 'GridColor', colors.gray);
    grid on;
    legend(dtypes, 'TextColor', colors.white, 'Location', 'best');
    
    save_plot('datatype_comparison.png');
end

function all_data = load_multiple_files(filenames)
    % Load multiple CSV files and combine them
    all_data = table();
    
    for i = 1:length(filenames)
        filename = filenames{i};
        
        if exist(filename, 'file')
            try
                data = readtable(filename);
                all_data = [all_data; data];
                fprintf('Loaded: %s (%d rows)\n', filename, height(data));
            catch ME
                fprintf('Warning: Could not load %s - %s\n', filename, ME.message);
            end
        else
            fprintf('Warning: File not found - %s\n', filename);
        end
    end
end

function save_plot(filename)
    % Save plot with proper styling for dark background
    set(gcf, 'Color', [0.1, 0.1, 0.1]);
    set(gca, 'XColor', [0.8, 0.8, 0.8], 'YColor', [0.8, 0.8, 0.8]);
    
    saveas(gcf, fullfile('plots', filename));
    fprintf('Saved: %s\n', fullfile('plots', filename));
end

% Run the analysis
analyze_simd_experiments();