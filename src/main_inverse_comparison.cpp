#include <iostream>
#include <complex>
#include <chrono>
#include <iomanip>
#include <type_traits>

#include "Matrix.hpp"

// Вспомогательная функция для проверки
template<typename MatType>
double check_matrix_identity(const MatType& M) {
    double max_error = 0.0;
    using ValueType = typename MatType::value_type;
    
    for (int i = 0; i < M.get_rows(); ++i) {
        for (int j = 0; j < M.get_cols(); ++j) {
            if constexpr (detail::is_matrix_v<ValueType>) {
                auto& block = M(i, j);
                for (int bi = 0; bi < block.get_rows(); ++bi) {
                    for (int bj = 0; bj < block.get_cols(); ++bj) {
                        double expected = (i == j && bi == bj) ? 1.0 : 0.0;
                        double actual = 0.0;
                        
                        // Безопасное преобразование типа
                        if constexpr (std::is_same_v<typename ValueType::value_type, std::complex<double>>) {
                            actual = std::abs(block(bi, bj));
                        } else if constexpr (std::is_same_v<typename ValueType::value_type, std::complex<float>>) {
                            actual = std::abs(block(bi, bj));
                        } else {
                            actual = static_cast<double>(block(bi, bj));
                        }
                        
                        double error = std::abs(actual - expected);
                        if (error > max_error) max_error = error;
                    }
                }
            } else {
                double expected = (i == j) ? 1.0 : 0.0;
                double actual = 0.0;
                
                // Безопасное преобразование типа
                if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
                    actual = std::abs(M(i, j));
                } else if constexpr (std::is_same_v<ValueType, std::complex<float>>) {
                    actual = std::abs(M(i, j));
                } else {
                    actual = static_cast<double>(M(i, j));
                }
                
                double error = std::abs(actual - expected);
                if (error > max_error) max_error = error;
            }
        }
    }
    
    return max_error;
}

template <typename MatType>
void test_inverse_matrix(int n) {
    using ValueType = typename MatType::value_type;

    std::cout << "========================================\n";
    std::cout << "Testing INVERSE matrix for " << n << "x" << n << " matrix\n";
    
    if constexpr (detail::is_matrix_v<ValueType>) {
        std::cout << "Type: Matrix<Matrix<" << typeid(typename ValueType::value_type).name() << ">>\n";
    } else {
        std::cout << "Type: Matrix<" << typeid(ValueType).name() << ">\n";
    }
    std::cout << "========================================\n";

    try {
        // 1. Генерация ГАРАНТИРОВАННО обратимой матрицы
        std::cout << "\n1. GENERATING INVERTIBLE MATRIX...\n";
        MatType A;
        
        if constexpr (detail::is_matrix_v<ValueType>) {
            int block_size;
            std::cout << "   Enter block size (e.g., 2, 3): ";
            std::cin >> block_size;
            
            using InnerType = typename ValueType::value_type;
            
            // Создаем диагонально-доминирующую блочную матрицу
            A = MatType::BlockMatrix(n, n, block_size, block_size);
            
            // Заполняем как блочно-диагональную с небольшими возмущениями
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) {
                        // Диагональный блок: диагонально-доминирующая матрица
                        ValueType block = ValueType::Identity(block_size, block_size);
                        
                        // Делаем диагонально-доминирующей
                        for (int bi = 0; bi < block_size; ++bi) {
                            double diag_val = 1.0 + (bi + 1) * 0.5;
                            
                            if constexpr (std::is_same_v<InnerType, std::complex<double>>) {
                                block(bi, bi) = std::complex<double>(diag_val, 0.0);
                            } else {
                                block(bi, bi) = static_cast<InnerType>(diag_val);
                            }
                            
                            // Небольшие недиагональные элементы
                            for (int bj = 0; bj < block_size; ++bj) {
                                if (bi != bj) {
                                    double val = 0.1 / (abs(bi - bj) + 1);
                                    if constexpr (std::is_same_v<InnerType, std::complex<double>>) {
                                        block(bi, bj) = std::complex<double>(val, 0.0);
                                    } else {
                                        block(bi, bj) = static_cast<InnerType>(val);
                                    }
                                }
                            }
                        }
                        A(i, j) = block;
                    } else if (abs(i - j) == 1) {
                        // Блоки рядом с диагональю: маленькие значения
                        ValueType block = ValueType::Zero(block_size, block_size);
                        if constexpr (std::is_same_v<InnerType, std::complex<double>>) {
                            block(0, 0) = std::complex<double>(0.05, 0.0);
                        } else {
                            block(0, 0) = static_cast<InnerType>(0.05);
                        }
                        A(i, j) = block;
                    } else {
                        // Остальные блоки: нулевые
                        A(i, j) = ValueType::Zero(block_size, block_size);
                    }
                }
            }
        } else {
            // Для скалярных матриц используем существующий генератор
            ValueType min_val, max_val;
            
            if constexpr (std::is_same_v<ValueType, int>) {
                min_val = -3;
                max_val = 3;
            } else if constexpr (std::is_same_v<ValueType, double>) {
                min_val = -2.0;
                max_val = 2.0;
            } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
                min_val = std::complex<double>(-1.0, -1.0);
                max_val = std::complex<double>(1.0, 1.0);
            } else {
                min_val = static_cast<ValueType>(-2);
                max_val = static_cast<ValueType>(2);
            }
            
            // Генерируем несколько раз, пока не получим обратимую
            int attempts = 0;
            do {
                A = MatType::Generate_matrix(n, n, min_val, max_val, 10, static_cast<ValueType>(1));
                attempts++;
                
                // Проверяем определитель
                auto det_opt = A.det();
                if (det_opt && !MatType::is_zero(*det_opt)) {
                    break;
                }
                
                if (attempts > 5) {
                    std::cout << "   ⚠ Could not generate invertible matrix after " << attempts << " attempts\n";
                    // Используем заведомо обратимую
                    A = MatType::Identity(n, n);
                    for (int i = 0; i < n; ++i) {
                        A(i, i) = static_cast<ValueType>(2.0);
                        if (i > 0) A(i, i-1) = static_cast<ValueType>(0.1);
                        if (i < n-1) A(i, i+1) = static_cast<ValueType>(0.1);
                    }
                    break;
                }
            } while (true);
            
            std::cout << "   Generated after " << attempts << " attempt(s)\n";
        }
        
        std::cout << "   ✓ Matrix generated\n";

        // 2. Показываем исходную матрицу
        std::cout << "\n2. ORIGINAL MATRIX A:\n";
        if (n <= 4) {
            if constexpr (detail::is_matrix_v<ValueType>) {
                A.detailed_print();
            } else {
                A.precise_print(4);
            }
        } else {
            std::cout << "   [Matrix " << n << "x" << n;
            if constexpr (detail::is_matrix_v<ValueType>) {
                if (A.get_rows() > 0 && A.get_cols() > 0) {
                    std::cout << " with blocks " << A(0, 0).get_rows() << "x" << A(0, 0).get_cols();
                }
            }
            std::cout << "]\n";
        }

        // 3. Проверяем определитель
        std::cout << "\n3. CHECKING DETERMINANT...\n";
        auto det_opt = A.det();
        if (det_opt) {
            std::cout << "   det(A) ";
            if constexpr (detail::is_matrix_v<ValueType>) {
                std::cout << "is " << (*det_opt).get_rows() << "x" << (*det_opt).get_cols() << " matrix\n";
                if ((*det_opt).get_rows() <= 3) {
                    std::cout << "   ";
                    (*det_opt).precise_print(4);
                }
            } else {
                std::cout << "= " << *det_opt << "\n";
            }
            
            if (MatType::is_zero(*det_opt)) {
                std::cout << "   ⚠ WARNING: Matrix appears to be singular!\n";
            } else {
                std::cout << "   ✓ Matrix is invertible (det ≠ 0)\n";
            }
        } else {
            std::cout << "   ⚠ Could not compute determinant\n";
        }

        // 4. Вычисляем обратную матрицу
        std::cout << "\n4. COMPUTING INVERSE MATRIX A⁻¹...\n";
        auto inv_start = std::chrono::high_resolution_clock::now();
        
        auto A_inv = A.inverse();
        
        auto inv_end = std::chrono::high_resolution_clock::now();
        double inv_time = std::chrono::duration<double>(inv_end - inv_start).count();
        
        std::cout << "   ✓ Inverse computed in " << std::fixed << std::setprecision(6) 
                  << inv_time << " seconds\n";

        // 5. Показываем обратную матрицу
        std::cout << "\n5. INVERSE MATRIX A⁻¹:\n";
        if (n <= 4) {
            if constexpr (detail::is_matrix_v<ValueType>) {
                A_inv.detailed_print();
            } else {
                A_inv.precise_print(6);
            }
        }

        // 6. Проверяем умножение
        std::cout << "\n6. VERIFICATION:\n";
        
        double max_error1 = 0.0, max_error2 = 0.0;
        
        std::cout << "   a) A × A⁻¹:\n";
        try {
            auto P1 = A * A_inv;
            if (n <= 3) {
                if constexpr (detail::is_matrix_v<ValueType>) {
                    P1.detailed_print();
                } else {
                    P1.precise_print(6);
                }
            }
            
            // Проверяем ошибку
            max_error1 = check_matrix_identity(P1);
            std::cout << "      Max error: " << std::scientific << std::setprecision(2) 
                      << max_error1 << "\n";
            if (max_error1 < 1e-6) {
                std::cout << "      ✓ A × A⁻¹ ≈ I\n";
            } else {
                std::cout << "      ✗ A × A⁻¹ ≠ I (error too large)\n";
            }
        } catch (const std::exception& e) {
            std::cout << "      ✗ Failed: " << e.what() << "\n";
            max_error1 = 1e10; // Большая ошибка
        }
        
        std::cout << "   b) A⁻¹ × A:\n";
        try {
            auto P2 = A_inv * A;
            if (n <= 3) {
                if constexpr (detail::is_matrix_v<ValueType>) {
                    P2.detailed_print();
                } else {
                    P2.precise_print(6);
                }
            }
            
            max_error2 = check_matrix_identity(P2);
            std::cout << "      Max error: " << std::scientific << std::setprecision(2) 
                      << max_error2 << "\n";
            if (max_error2 < 1e-6) {
                std::cout << "      ✓ A⁻¹ × A ≈ I\n";
            } else {
                std::cout << "      ✗ A⁻¹ × A ≠ I (error too large)\n";
            }
        } catch (const std::exception& e) {
            std::cout << "      ✗ Failed: " << e.what() << "\n";
            max_error2 = 1e10; // Большая ошибка
        }

        std::cout << "\n========================================\n";
        if (max_error1 < 1e-6 && max_error2 < 1e-6) {
            std::cout << "✓ TEST PASSED!\n";
        } else {
            std::cout << "⚠ TEST HAS ISSUES (errors too large)\n";
        }

    } catch (const std::exception& e) {
        std::cout << "\n========================================\n";
        std::cout << "✗ TEST FAILED!\n";
        std::cout << "Error: " << e.what() << "\n";
        std::cout << "========================================\n";
    }
}

int main() {
    std::cout << "MATRIX INVERSE TEST\n";
    std::cout << "===================\n";
    
    while (true) {
        int n;
        std::string type_choice;
        
        std::cout << "\nEnter matrix size (0 to exit): ";
        std::cin >> n;
        
        if (n == 0) {
            std::cout << "Goodbye!\n";
            break;
        }
        
        if (n < 1) {
            std::cout << "Invalid size!\n";
            continue;
        }
        
        std::cout << "Choose matrix type:\n";
        std::cout << "  1. int (will be cast to double for inverse)\n";
        std::cout << "  2. double\n";
        std::cout << "  3. complex\n";
        std::cout << "  4. matrix of matrices (block matrix)\n";
        std::cout << "Choice (1-4): ";
        std::cin >> type_choice;
        
        try {
            if (type_choice == "1") {
                test_inverse_matrix<Matrix<int>>(n);
            } else if (type_choice == "2") {
                test_inverse_matrix<Matrix<double>>(n);
            } else if (type_choice == "3") {
                test_inverse_matrix<Matrix<std::complex<double>>>(n);
            } else if (type_choice == "4") {
                std::string inner_type;
                std::cout << "Choose inner matrix type:\n";
                std::cout << "  1. int (blocks)\n";
                std::cout << "  2. double (blocks)\n";
                std::cout << "  3. complex (blocks)\n";
                std::cout << "Choice (1-3): ";
                std::cin >> inner_type;
                
                if (inner_type == "1") {
                    test_inverse_matrix<Matrix<Matrix<int>>>(n);
                } else if (inner_type == "2") {
                    test_inverse_matrix<Matrix<Matrix<double>>>(n);
                } else if (inner_type == "3") {
                    test_inverse_matrix<Matrix<Matrix<std::complex<double>>>>(n);
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
