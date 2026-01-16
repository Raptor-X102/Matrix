#include "../headers/Matrix.hpp"
#include "../headers/Timer.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <chrono>

using namespace std;

template<typename T>
void benchmark_multiplication(int size) {
    cout << "\n--- Benchmarking Matrix Multiplication (" << size << "x" << size << ", Type: " << typeid(T).name() << ") ---\n";

    cout << "Generating matrices A and B...\n";
    Matrix<T> A = Matrix<T>::Generate_matrix(size, size, static_cast<T>(-10), static_cast<T>(10), 10);
    Matrix<T> B = Matrix<T>::Generate_matrix(size, size, static_cast<T>(-10), static_cast<T>(10), 10);
    Matrix<T> C(size, size);

    Timer timer;

    cout << "Testing Basic Algorithm...\n";
    timer.start();
    C = multiply_basic(A, B);
    double time_basic = timer.elapsed();
    cout << "Time (Basic): " << fixed << setprecision(6) << time_basic << " seconds\n";

    #ifdef __AVX__
    cout << "Testing AVX Algorithm...\n";
    timer.start();
    C = multiply_avx(A, B);
    double time_avx = timer.elapsed();
    cout << "Time (AVX): " << fixed << setprecision(6) << time_avx << " seconds\n";

    if (time_basic > 0) {
        double speedup = time_basic / time_avx;
        cout << "Speedup (Basic / AVX): " << fixed << setprecision(2) << speedup << "x\n";
    }
    #else
    cout << "AVX algorithm not compiled in. Skipping AVX benchmark.\n";
    #endif

    cout << "Testing Default Operator* (typically Optimal)...\n";
    timer.start();
    C = A * B;
    double time_default = timer.elapsed();
    cout << "Time (Default *): " << fixed << setprecision(6) << time_default << " seconds\n";

    cout << "--- Benchmark Complete for size " << size << "x" << size << " ---\n\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <data_type> <size>\n";
        cerr << "Data types: int, double\n";
        cerr << "Example: " << argv[0] << " double 500\n";
        return 1;
    }

    string type_str = argv[1];
    int size = stoi(argv[2]);

    if (size <= 0) {
        cerr << "Error: Size must be positive.\n";
        return 1;
    }

        if (size > 2000) {
            cerr << "Warning: Size " << size << " is very large. This may take a significant amount of time and memory.\n";
        }

    try {
        if (type_str == "int") {
            benchmark_multiplication<int>(size);
        } else if (type_str == "double") {
            benchmark_multiplication<double>(size);
        } else {
            cerr << "Error: Unsupported data type '" << type_str << "'. Use 'int' or 'double'.\n";
            return 1;
        }
    } catch (const exception& e) {
        cerr << "Exception occurred: " << e.what() << endl;
        return 1;
    }

    return 0;
}
