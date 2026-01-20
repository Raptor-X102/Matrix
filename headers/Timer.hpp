#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>

class Timer {
public:
    Timer() = default;

    void start() { start_ = std::chrono::high_resolution_clock::now(); }

    // Returns time in seconds
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return static_cast<double>(duration.count()) / 1e6; // in seconds
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

#endif
