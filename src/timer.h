#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

class HighResolutionTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double>;

    static TimePoint now() { return Clock::now(); }
    static double elapsedSeconds(const TimePoint& start, const TimePoint& end) {
        return std::chrono::duration_cast<Duration>(end - start).count();
    }
};

class MeasurementStats {
private:
    std::vector<double> measurements;
    double mean_, stddev_, min_, max_;

public:
    void add(double value) { measurements.push_back(value); }
    
    void compute() {
        if (measurements.empty()) return;
        
        // Remove outliers using median absolute deviation
        if (measurements.size() >= 5) {
            removeOutliers();
        }
        
        mean_ = 0.0;
        for (double m : measurements) mean_ += m;
        mean_ /= measurements.size();
        
        stddev_ = 0.0;
        for (double m : measurements) stddev_ += (m - mean_) * (m - mean_);
        stddev_ = std::sqrt(stddev_ / measurements.size());
        
        min_ = *std::min_element(measurements.begin(), measurements.end());
        max_ = *std::max_element(measurements.begin(), measurements.end());
    }
    
    double mean() const { return mean_; }
    double stddev() const { return stddev_; }
    double min() const { return min_; }
    double max() const { return max_; }
    size_t count() const { return measurements.size(); }
    
private:
    void removeOutliers() {
        // Sort and compute median
        std::vector<double> sorted = measurements;
        std::sort(sorted.begin(), sorted.end());
        double median = sorted[sorted.size() / 2];
        
        // Compute MAD
        std::vector<double> deviations;
        for (double m : measurements) {
            deviations.push_back(std::abs(m - median));
        }
        std::sort(deviations.begin(), deviations.end());
        double mad = deviations[deviations.size() / 2];
        
        // Remove outliers (beyond 3 MAD)
        double threshold = 3.0 * 1.4826 * mad; // 1.4826 constant for normal distribution
        measurements.erase(
            std::remove_if(measurements.begin(), measurements.end(),
                [median, threshold](double x) { return std::abs(x - median) > threshold; }),
            measurements.end()
        );
    }
};

#endif