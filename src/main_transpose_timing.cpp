// main_transpose_timing.cpp
#include <iostream>
#include <complex>
#include <chrono>
#include <iomanip>
#include <random>
#include <type_traits>
#include <string>

#include "Matrix.hpp"

// Вспомогательная функция для форматирования времени
std::string format_duration(double seconds) {
    if (seconds < 1e-6) {
        return std::to_string(seconds * 1e9) + " ns";
    } else if (seconds < 1e-3) {
        return std::to_string(seconds * 1e6) + " μs";
    } else if (seconds < 1.0) {
        return std::to_string(seconds * 1e3) + " ms";
    } else {
        return std::to_string(seconds) + " s";
    }
}

// Функция для получения значения элемента как double (для скалярных типов)
template<typename T>
double get_element_value(const T& elem) {
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return std::abs(elem);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return std::abs(elem);
    } else if constexpr (std::is_arithmetic_v<T>) {
        return static_cast<double>(elem);
    } else {
        // Для нескалярных типов (например, Matrix) возвращаем 0
        // Эта версия не должна вызываться для матриц
        return 0.0;
    }
}

// Функция для проверки корректности транспонирования (для скалярных типов)
template<typename T>
bool verify_transpose_scalar(const Matrix<T>& original, const Matrix<T>& transposed) {
    static_assert(!detail::is_matrix_v<T>, 
                  "verify_transpose_scalar should only be used for scalar types");
    
    if (original.get_rows() != transposed.get_cols() || 
        original.get_cols() != transposed.get_rows()) {
        return false;
    }
    
    const double eps = 1e-10;
    
    for (int i = 0; i < original.get_rows(); ++i) {
        for (int j = 0; j < original.get_cols(); ++j) {
            double orig_val = get_element_value(original(i, j));
            double trans_val = get_element_value(transposed(j, i));
            
            if (std::abs(orig_val - trans_val) > eps) {
                std::cout << "  Mismatch at [" << i << "," << j << "]: " 
                          << orig_val << " != " << trans_val << "\n";
                return false;
            }
        }
    }
    return true;
}

// Функция для проверки корректности транспонирования (для блочных матриц)
template<typename T>
bool verify_transpose_block(const Matrix<Matrix<T>>& original, 
                           const Matrix<Matrix<T>>& transposed) {
    if (original.get_rows() != transposed.get_cols() || 
        original.get_cols() != transposed.get_rows()) {
        return false;
    }
    
    const double eps = 1e-10;
    
    for (int i = 0; i < original.get_rows(); ++i) {
        for (int j = 0; j < original.get_cols(); ++j) {
            const auto& orig_block = original(i, j);
            const auto& trans_block = transposed(j, i);
            
            // Проверяем размеры блоков
            if (orig_block.get_rows() != trans_block.get_rows() ||
                orig_block.get_cols() != trans_block.get_cols()) {
                std::cout << "  Block size mismatch at [" << i << "," << j << "]\n";
                return false;
            }
            
            // Проверяем содержимое блоков
            for (int bi = 0; bi < orig_block.get_rows(); ++bi) {
                for (int bj = 0; bj < orig_block.get_cols(); ++bj) {
                    double orig_val = get_element_value(orig_block(bi, bj));
                    double trans_val = get_element_value(trans_block(bi, bj));
                    
                    if (std::abs(orig_val - trans_val) > eps) {
                        std::cout << "  Mismatch at block [" << i << "," << j 
                                  << "] element [" << bi << "," << bj << "]: " 
                                  << orig_val << " != " << trans_val << "\n";
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// Обертка для выбора правильной функции проверки
template<typename MatType>
bool verify_transpose(const MatType& original, const MatType& transposed) {
    using ValueType = typename MatType::value_type;
    
    if constexpr (detail::is_matrix_v<ValueType>) {
        return verify_transpose_block<typename ValueType::value_type>(original, transposed);
    } else {
        return verify_transpose_scalar<ValueType>(original, transposed);
    }
}

// Функция для проверки блочной матрицы на симметрию
template<typename T>
bool check_block_matrix_symmetry(const Matrix<Matrix<T>>& matrix) {
    if (matrix.get_rows() != matrix.get_cols()) {
        return false;
    }
    
    const double eps = 1e-10;
    
    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int j = i + 1; j < matrix.get_cols(); ++j) {
            const auto& block1 = matrix(i, j);
            const auto& block2 = matrix(j, i);
            
            // Проверяем размеры блоков
            if (block1.get_rows() != block2.get_rows() ||
                block1.get_cols() != block2.get_cols()) {
                return false;
            }
            
            // Проверяем содержимое блоков
            for (int bi = 0; bi < block1.get_rows(); ++bi) {
                for (int bj = 0; bj < block1.get_cols(); ++bj) {
                    double val1 = get_element_value(block1(bi, bj));
                    double val2 = get_element_value(block2(bi, bj));
                    
                    if (std::abs(val1 - val2) > eps) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

template<typename MatType>
void test_transpose_with_timing(int rows, int cols) {
    using ValueType = typename MatType::value_type;
    
    std::cout << "========================================\n";
    std::cout << "Testing TRANSPOSE for " << rows << "x" << cols << " matrix\n";
    
    if constexpr (detail::is_matrix_v<ValueType>) {
        std::cout << "Type: Matrix<Matrix<" 
                  << typeid(typename ValueType::value_type).name() << ">>\n";
    } else {
        std::cout << "Type: Matrix<" << typeid(ValueType).name() << ">\n";
    }
    std::cout << "========================================\n";

    try {
        // 1. Генерация матрицы
        std::cout << "\n1. GENERATING MATRIX...\n";
        MatType A;
        
        if constexpr (detail::is_matrix_v<ValueType>) {
            // Для блочной матрицы
            int block_rows, block_cols;
            std::cout << "   Enter block rows: ";
            std::cin >> block_rows;
            std::cout << "   Enter block cols: ";
            std::cin >> block_cols;
            
            // Создаем диагональную блочную матрицу для простоты
            A = MatType::BlockIdentity(rows, cols, block_rows, block_cols);
            
            // Добавляем некоторые недиагональные элементы
            using InnerType = typename ValueType::value_type;
            ValueType small_block = ValueType::Zero(block_rows, block_cols);
            if (block_rows > 0 && block_cols > 0) {
                small_block(0, 0) = static_cast<InnerType>(0.1);
            }
            
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    if (i != j && std::abs(i - j) == 1) {
                        A(i, j) = small_block;
                    }
                }
            }
        } else {
            // Для скалярных матриц генерируем случайные значения
            ValueType min_val, max_val;
            
            if constexpr (std::is_same_v<ValueType, int>) {
                min_val = -10;
                max_val = 10;
            } else if constexpr (std::is_same_v<ValueType, double>) {
                min_val = -5.0;
                max_val = 5.0;
            } else if constexpr (std::is_same_v<ValueType, float>) {
                min_val = -5.0f;
                max_val = 5.0f;
            } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
                min_val = std::complex<double>(-2.0, -2.0);
                max_val = std::complex<double>(2.0, 2.0);
            } else {
                min_val = static_cast<ValueType>(-5);
                max_val = static_cast<ValueType>(5);
            }
            
            A = MatType::Generate_matrix(rows, cols, min_val, max_val, 5, ValueType{1});
        }
        
        std::cout << "   ✓ Matrix generated\n";
        
        // 2. Показываем исходную матрицу
        std::cout << "\n2. ORIGINAL MATRIX A:\n";
        if (rows <= 5 && cols <= 5) {
            if constexpr (detail::is_matrix_v<ValueType>) {
                A.detailed_print();
            } else {
                A.precise_print(4);
            }
        } else {
            std::cout << "   [Matrix " << rows << "x" << cols;
            if constexpr (detail::is_matrix_v<ValueType>) {
                if (rows > 0 && cols > 0) {
                    std::cout << " with blocks " << A(0, 0).get_rows() << "x" 
                              << A(0, 0).get_cols();
                }
            }
            std::cout << "]\n";
        }
        
        // 3. Транспонирование с замером времени
        std::cout << "\n3. TRANSPOSING MATRIX...\n";
        
        // Многократное выполнение для точного замера
        const int iterations = (rows * cols < 10000) ? 1000 : 10;
        double total_time = 0.0;
        MatType A_transposed;
        
        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            A_transposed = A.transpose();
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }
        
        double avg_time = total_time / iterations;
        std::cout << "   ✓ Transpose completed\n";
        std::cout << "   Average time over " << iterations << " iterations: " 
                  << std::fixed << std::setprecision(9) << avg_time << " s (" 
                  << format_duration(avg_time) << ")\n";
        
        // 4. Показываем транспонированную матрицу
        std::cout << "\n4. TRANSPOSED MATRIX A^T:\n";
        if (cols <= 5 && rows <= 5) {
            if constexpr (detail::is_matrix_v<ValueType>) {
                A_transposed.detailed_print();
            } else {
                A_transposed.precise_print(4);
            }
        } else {
            std::cout << "   [Matrix " << cols << "x" << rows;
            if constexpr (detail::is_matrix_v<ValueType>) {
                if (cols > 0 && rows > 0) {
                    std::cout << " with blocks " << A_transposed(0, 0).get_rows() << "x" 
                              << A_transposed(0, 0).get_cols();
                }
            }
            std::cout << "]\n";
        }
        
        // 5. Проверка корректности
        std::cout << "\n5. VERIFYING TRANSPOSE CORRECTNESS...\n";
        
        // Проверка 1: (A^T)^T == A
        std::cout << "   a) Checking (A^T)^T == A...\n";
        auto start = std::chrono::high_resolution_clock::now();
        auto A_double_transposed = A_transposed.transpose();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        
        bool correct = verify_transpose(A, A_double_transposed);
        if (correct) {
            std::cout << "      ✓ (A^T)^T == A (verification took " 
                      << elapsed.count() << " s)\n";
        } else {
            std::cout << "      ✗ (A^T)^T != A\n";
        }
        
        // Проверка 2: Симметрия для симметричных матриц (если квадратная)
        if (rows == cols) {
            std::cout << "   b) Checking symmetry property...\n";
            
            // Создаем симметричную матрицу
            MatType symmetric = A;
            if constexpr (detail::is_matrix_v<ValueType>) {
                // Для блочных матриц делаем симметричной
                for (int i = 0; i < rows; ++i) {
                    for (int j = i + 1; j < cols; ++j) {
                        symmetric(j, i) = symmetric(i, j);
                    }
                }
                
                bool symmetric_correct = check_block_matrix_symmetry<typename ValueType::value_type>(symmetric);
                
                if (symmetric_correct) {
                    std::cout << "      ✓ Symmetric matrix property holds\n";
                } else {
                    std::cout << "      ✗ Symmetric matrix property violated\n";
                }
            } else {
                // Для скалярных матриц
                for (int i = 0; i < rows; ++i) {
                    for (int j = i + 1; j < cols; ++j) {
                        symmetric(j, i) = symmetric(i, j);
                    }
                }
                
                auto sym_transposed = symmetric.transpose();
                bool symmetric_correct = verify_transpose(symmetric, sym_transposed);
                
                if (symmetric_correct) {
                    std::cout << "      ✓ Symmetric matrix property holds\n";
                } else {
                    std::cout << "      ✗ Symmetric matrix property violated\n";
                }
            }
        }
        
        // 6. Дополнительная информация
        std::cout << "\n6. ADDITIONAL INFORMATION:\n";
        std::cout << "   Original matrix dimensions: " << rows << "x" << cols << "\n";
        std::cout << "   Transposed matrix dimensions: " << cols << "x" << rows << "\n";
        if constexpr (detail::is_matrix_v<ValueType>) {
            if (rows > 0 && cols > 0) {
                std::cout << "   Block size: " << A(0, 0).get_rows() << "x" 
                          << A(0, 0).get_cols() << "\n";
                std::cout << "   Total elements (including blocks): " 
                          << rows * cols * A(0, 0).get_rows() * A(0, 0).get_cols() << "\n";
            }
        } else {
            std::cout << "   Total elements: " << rows * cols << "\n";
        }
        
        std::cout << "\n========================================\n";
        std::cout << "✓ TRANSPOSE TEST COMPLETED\n";
        
    } catch (const std::exception& e) {
        std::cout << "\n========================================\n";
        std::cout << "✗ TEST FAILED!\n";
        std::cout << "Error: " << e.what() << "\n";
    }
}

int main() {
    std::cout << "MATRIX TRANSPOSE TIMING TEST\n";
    std::cout << "=============================\n";
    
    while (true) {
        int rows, cols;
        std::string type_choice;
        
        std::cout << "\nEnter matrix rows (0 to exit): ";
        std::cin >> rows;
        
        if (rows == 0) {
            std::cout << "Goodbye!\n";
            break;
        }
        
        std::cout << "Enter matrix cols: ";
        std::cin >> cols;
        
        if (rows < 1 || cols < 1) {
            std::cout << "Invalid dimensions!\n";
            continue;
        }
        
        std::cout << "\nChoose matrix type:\n";
        std::cout << "  1. int\n";
        std::cout << "  2. double\n";
        std::cout << "  3. float\n";
        std::cout << "  4. complex<double>\n";
        std::cout << "  5. matrix of matrices (block matrix)\n";
        std::cout << "Choice (1-5): ";
        std::cin >> type_choice;
        
        try {
            if (type_choice == "1") {
                test_transpose_with_timing<Matrix<int>>(rows, cols);
            } else if (type_choice == "2") {
                test_transpose_with_timing<Matrix<double>>(rows, cols);
            } else if (type_choice == "3") {
                test_transpose_with_timing<Matrix<float>>(rows, cols);
            } else if (type_choice == "4") {
                test_transpose_with_timing<Matrix<std::complex<double>>>(rows, cols);
            } else if (type_choice == "5") {
                std::string inner_type;
                std::cout << "\nChoose inner matrix type:\n";
                std::cout << "  1. int blocks\n";
                std::cout << "  2. double blocks\n";
                std::cout << "  3. float blocks\n";
                std::cout << "  4. complex<double> blocks\n";
                std::cout << "Choice (1-4): ";
                std::cin >> inner_type;
                
                if (inner_type == "1") {
                    test_transpose_with_timing<Matrix<Matrix<int>>>(rows, cols);
                } else if (inner_type == "2") {
                    test_transpose_with_timing<Matrix<Matrix<double>>>(rows, cols);
                } else if (inner_type == "3") {
                    test_transpose_with_timing<Matrix<Matrix<float>>>(rows, cols);
                } else if (inner_type == "4") {
                    test_transpose_with_timing<Matrix<Matrix<std::complex<double>>>>(rows, cols);
                } else {
                    std::cout << "Invalid choice\n";
                }
            } else {
                std::cout << "Invalid choice\n";
            }
        } catch (const std::exception& e) {
            std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        }
        
        std::cout << "\nPress Enter to continue...";
        std::cin.ignore();
        std::cin.get();
    }
    
    return 0;
}
