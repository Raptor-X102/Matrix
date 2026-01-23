#include "Matrix.hpp"
#include "Timer.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <iomanip> // For formatting output

using namespace std;

template<typename T> void run_arithmetic_tests(int size) {
    cout << "\n--- Testing Arithmetic Operations for size " << size << "x" << size
         << " (Type: " << typeid(T).name() << ") ---\n";

    Matrix<T> A = Matrix<T>::Generate_matrix(size,
                                             size,
                                             static_cast<T>(-5),
                                             static_cast<T>(5),
                                             10);
    Matrix<T> B = Matrix<T>::Generate_matrix(size,
                                             size,
                                             static_cast<T>(-3),
                                             static_cast<T>(3),
                                             10);
    T scalar = static_cast<T>(2);

    cout << "Initial Matrices:\n";
    cout << "Matrix A:\n";
    A.print(6);
    cout << "Matrix B:\n";
    B.print(6);
    cout << "Scalar: " << scalar << "\n\n";

// Helper macro to reduce repetition
#define MEASURE_AND_PRINT(OP_DESC, OP_EXPR, RESULT_LABEL)                               \
    {                                                                                   \
        Timer t;                                                                        \
        t.start();                                                                      \
        auto result = OP_EXPR;                                                          \
        double elapsed_time = t.elapsed();                                              \
        cout << "Testing " << OP_DESC << "... Time: " << fixed << setprecision(6)       \
             << elapsed_time << " seconds\n";                                           \
        cout << RESULT_LABEL << ":\n";                                                  \
        result.print(6);                                                                \
        cout << endl;                                                                   \
    }

    MEASURE_AND_PRINT("Matrix Addition (A + B)", (A + B), "Result A + B")
    MEASURE_AND_PRINT("Matrix Subtraction (A - B)", (A - B), "Result A - B")
    MEASURE_AND_PRINT("Matrix Multiplication (A * B)", (A * B), "Result A * B")
    MEASURE_AND_PRINT("Matrix + Scalar (A + scalar)", (A + scalar), "Result A + scalar")
    MEASURE_AND_PRINT("Matrix - Scalar (A - scalar)", (A - scalar), "Result A - scalar")
    MEASURE_AND_PRINT("Matrix * Scalar (A * scalar)", (A * scalar), "Result A * scalar")
    MEASURE_AND_PRINT("Scalar * Matrix (scalar * A)", (scalar * A), "Result scalar * A")

#undef MEASURE_AND_PRINT

    cout << "--- Test Complete for size " << size << "x" << size << " ---\n\n";
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <data_type> <size>\n";
        cerr << "Data types: int, double\n";
        cerr << "Example: " << argv[0] << " double 4\n";
        cerr << "Note: Output will be printed only for matrices up to 6x6.\n";
        return 1;
    }

    string type_str = argv[1];
    int size = stoi(argv[2]);

    if (size <= 0) {
        cerr << "Error: Size must be positive.\n";
        return 1;
    }

    if (size > 20) { // Just a warning, you can adjust this limit
        cerr << "Warning: Size " << size
             << " is large. Consider reducing for better performance timing.\n";
    }

    try {
        if (type_str == "int") {
            run_arithmetic_tests<int>(size);
        } else if (type_str == "double") {
            run_arithmetic_tests<double>(size);
        } else {
            cerr << "Error: Unsupported data type '" << type_str
                 << "'. Use 'int' or 'double'.\n";
            return 1;
        }
    } catch (const exception &e) {
        cerr << "Exception occurred: " << e.what() << endl;
        return 1;
    }

    return 0;
}
