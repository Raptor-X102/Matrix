#include <iostream>
#include <chrono>
#include <iomanip>
#include <functional>
#include <complex>

#include "Matrix.hpp"

#ifdef TIME_MEASURE
#include "Timer.hpp"
#endif

// Общая шаблонная функция для double и complex
template <typename T>
void test_inverse_with_timer_and_print(const std::string &method_name, 
                                       const Matrix<T> &original_matrix, 
                                       std::function<Matrix<T>(const Matrix<T>&)> inverse_func) {
    std::cout << "Testing " << method_name << "...\n";

    if (original_matrix.get_rows() <= 5 && original_matrix.get_cols() <= 5) {
        std::cout << "Original matrix:\n";
        original_matrix.precise_print(6);
        std::cout << "\n";
    }

    Matrix<T> result_matrix(1, 1);
    
    double elapsed_time = 0.0;
    
    {
#ifdef TIME_MEASURE
        Timer timer;
        result_matrix = inverse_func(original_matrix);
#else
        auto start_time = std::chrono::high_resolution_clock::now();
        result_matrix = inverse_func(original_matrix);
        auto end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
#endif
    }
    
    std::cout << "Time: " << std::fixed << std::setprecision(6) 
              << elapsed_time << " seconds\n";

    if (result_matrix.get_rows() <= 5 && result_matrix.get_cols() <= 5) {
        std::cout << "Inverse matrix (" << method_name << "):\n";
        result_matrix.precise_print(6);
        std::cout << "\n";
    }

    std::cout << method_name << ": Inverse computation completed.\n";
    std::cout << "\n";
}

// Перегруженная версия для int, которая возвращает Matrix<double>
void test_inverse_int(const std::string &method_name, 
                      const Matrix<int> &original_matrix,
                      std::function<Matrix<double>(const Matrix<int>&)> inverse_func) {
    std::cout << "Testing " << method_name << "...\n";

    if (original_matrix.get_rows() <= 5 && original_matrix.get_cols() <= 5) {
        std::cout << "Original matrix:\n";
        original_matrix.print();
        std::cout << "\n";
    }

    Matrix<double> result_matrix(1, 1);
    
    double elapsed_time = 0.0;
    
    {
#ifdef TIME_MEASURE
        Timer timer;
        result_matrix = inverse_func(original_matrix);
#else
        auto start_time = std::chrono::high_resolution_clock::now();
        result_matrix = inverse_func(original_matrix);
        auto end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
#endif
    }
    
    std::cout << "Time: " << std::fixed << std::setprecision(6) 
              << elapsed_time << " seconds\n";

    if (result_matrix.get_rows() <= 5 && result_matrix.get_cols() <= 5) {
        std::cout << "Inverse matrix (" << method_name << "):\n";
        result_matrix.precise_print(6);
        std::cout << "\n";
    }

    // Проверка корректности
    if (original_matrix.get_rows() <= 5 && original_matrix.get_rows() == original_matrix.get_cols()) {
        try {
            Matrix<double> original_double(original_matrix.get_rows(), original_matrix.get_cols());
            for (int i = 0; i < original_matrix.get_rows(); ++i) {
                for (int j = 0; j < original_matrix.get_cols(); ++j) {
                    original_double(i, j) = static_cast<double>(original_matrix(i, j));
                }
            }
            
            auto identity_check = original_double * result_matrix;
            std::cout << "Verification (A * A⁻¹):\n";
            identity_check.precise_print(6);
            
            double max_error = 0.0;
            for (int i = 0; i < identity_check.get_rows(); ++i) {
                for (int j = 0; j < identity_check.get_cols(); ++j) {
                    double expected = (i == j) ? 1.0 : 0.0;
                    double error = std::abs(identity_check(i, j) - expected);
                    if (error > max_error) {
                        max_error = error;
                    }
                }
            }
            std::cout << "Max error: " << std::scientific << std::setprecision(2) 
                      << max_error << "\n";
        } catch (const std::exception& e) {
            std::cout << "Verification failed: " << e.what() << "\n";
        }
    }
    
    std::cout << method_name << ": Inverse computation completed.\n";
    std::cout << "\n";
}

int main() {
    int n;
    std::string type_choice;
    std::cout << "Enter matrix size: ";
    std::cin >> n;
    std::cout << "Choose type (int/double/complex): ";
    std::cin >> type_choice;

    if (type_choice == "int") {
        std::cout << "Generating random " << n << "x" << n << " integer matrix...\n";
        auto matrix = Matrix<int>::Generate_matrix(n, n, -10, 10);

        // Для целочисленных матриц используем перегруженную функцию
        std::function<Matrix<double>(const Matrix<int>&)> inverse_lu = 
            [](const Matrix<int>& m) { return m.inverse(); };
        std::function<Matrix<double>(const Matrix<int>&)> inverse_gauss = 
            [](const Matrix<int>& m) { return m.inverse_gauss_jordan(); };
        
        test_inverse_int("LU Decomposition Method", matrix, inverse_lu);
        test_inverse_int("Gauss-Jordan Method", matrix, inverse_gauss);
        
    } else if (type_choice == "double") {
        std::cout << "Generating random " << n << "x" << n << " double matrix...\n";
        auto matrix = Matrix<double>::Generate_matrix(n, n, -10.0, 10.0);

        auto inverse_lu = [](const Matrix<double>& m) { return m.inverse(); };
        auto inverse_gauss = [](const Matrix<double>& m) { 
            return m.inverse_gauss_jordan();
        };

        test_inverse_with_timer_and_print<double>("LU Decomposition Method", matrix, inverse_lu);
        test_inverse_with_timer_and_print<double>("Gauss-Jordan Method", matrix, inverse_gauss);
        
    } else if (type_choice == "complex") {
        std::cout << "Generating random " << n << "x" << n << " complex matrix...\n";
        
        Matrix<std::complex<double>> matrix(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double real_part = static_cast<double>(rand()) / RAND_MAX * 20.0 - 10.0;
                double imag_part = static_cast<double>(rand()) / RAND_MAX * 20.0 - 10.0;
                matrix(i, j) = std::complex<double>(real_part, imag_part);
            }
        }

        auto inverse_lu = [](const Matrix<std::complex<double>>& m) { return m.inverse(); };
        auto inverse_gauss = [](const Matrix<std::complex<double>>& m) { 
            return m.inverse_gauss_jordan();
        };

        test_inverse_with_timer_and_print<std::complex<double>>("LU Decomposition Method", matrix, inverse_lu);
        test_inverse_with_timer_and_print<std::complex<double>>("Gauss-Jordan Method", matrix, inverse_gauss);
        
    } else {
        std::cerr << "Error: Unsupported type '" << type_choice << "'. Use 'int', 'double', or 'complex'.\n";
        return 1;
    }

    return 0;
}
