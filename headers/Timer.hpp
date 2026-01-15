#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        std::cerr << "Time consumed: " << duration.count() << " us" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

#endif
