#include <iostream>
#include <complex>
#include <chrono>
#include <iomanip>
#include <type_traits>

#include "Matrix.hpp"

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
        // 1. Генерация матрицы
        std::cout << "\n1. GENERATING MATRIX...\n";
        MatType A;
        
        if constexpr (detail::is_matrix_v<ValueType>) {
            // Для блочной матрицы
            int block_size;
            std::cout << "   Enter block size (e.g., 2, 3): ";
            std::cin >> block_size;
            
            using InnerType = typename ValueType::value_type;
            
            // Создаем почти единичную блочную матрицу
            ValueType identity_block = ValueType::Identity(block_size, block_size);
            ValueType small_block = ValueType::Identity(block_size, block_size);
            
            // Добавляем небольшой шум
            if (block_size > 1) {
                small_block(0, block_size-1) = static_cast<InnerType>(0.1);
                small_block(block_size-1, 0) = static_cast<InnerType>(-0.1);
            }
            
            A = MatType::Generate_matrix(n, n, identity_block, small_block, 3, identity_block);
        } else {
            // Для скалярной матрицы
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
            
            A = MatType::Generate_matrix(n, n, min_val, max_val, 10, static_cast<ValueType>(1));
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
            std::cout << " - too large to display]\n";
        }

        // 3. Вычисляем определитель (опционально)
        std::cout << "\n3. COMPUTING DETERMINANT (optional)...\n";
        auto det_start = std::chrono::high_resolution_clock::now();
        auto det_opt = A.det();
        auto det_end = std::chrono::high_resolution_clock::now();
        double det_time = std::chrono::duration<double>(det_end - det_start).count();
        
        if (det_opt) {
            std::cout << "   ✓ Determinant computed in " << std::fixed << std::setprecision(6) 
                      << det_time << " seconds\n";
            if (n <= 3) {
                std::cout << "   det(A) = ";
                if constexpr (detail::is_matrix_v<ValueType>) {
                    if ((*det_opt).get_rows() <= 3 && (*det_opt).get_cols() <= 3) {
                        std::cout << "\n";
                        (*det_opt).precise_print(4);
                    } else {
                        std::cout << "[Matrix " << (*det_opt).get_rows() << "x" << (*det_opt).get_cols() << "]\n";
                    }
                } else {
                    std::cout << *det_opt << "\n";
                }
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
        } else {
            std::cout << "   [Inverse matrix " << A_inv.get_rows() << "x" << A_inv.get_cols();
            if constexpr (detail::is_matrix_v<ValueType>) {
                if (A_inv.get_rows() > 0 && A_inv.get_cols() > 0) {
                    std::cout << " with blocks " << A_inv(0, 0).get_rows() << "x" << A_inv(0, 0).get_cols();
                }
            }
            std::cout << " - too large to display]\n";
        }

        // 6. Проверяем умножение A × A⁻¹ ≈ I (для небольших матриц)
        if (n <= 3) {
            std::cout << "\n6. VERIFICATION: A × A⁻¹ (should be ≈ I):\n";
            try {
                auto product = A * A_inv;
                
                if constexpr (detail::is_matrix_v<ValueType>) {
                    product.detailed_print();
                } else {
                    product.precise_print(6);
                }
                
                // Проверяем ошибку
                double max_error = 0.0;
                for (int i = 0; i < product.get_rows(); ++i) {
                    for (int j = 0; j < product.get_cols(); ++j) {
                        if constexpr (detail::is_matrix_v<ValueType>) {
                            // Для блочной матрицы проверяем каждый блок
                            auto& block = product(i, j);
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
                            // Для скалярной матрицы
                            using ElemType = typename MatType::value_type;
                            ElemType actual_elem = product(i, j);
                            
                            double actual = 0.0;
                            double expected = (i == j) ? 1.0 : 0.0;
                            
                            // Безопасное преобразование типа
                            if constexpr (std::is_same_v<ElemType, std::complex<double>>) {
                                actual = std::abs(actual_elem);
                            } else if constexpr (std::is_same_v<ElemType, std::complex<float>>) {
                                actual = std::abs(actual_elem);
                            } else if constexpr (std::is_integral_v<ElemType>) {
                                actual = static_cast<double>(actual_elem);
                            } else {
                                actual = actual_elem;
                            }
                            
                            double error = std::abs(actual - expected);
                            if (error > max_error) max_error = error;
                        }
                    }
                }
                
                std::cout << "\n   Max error: " << std::scientific << std::setprecision(2) 
                          << max_error << "\n";
                if (max_error < 1e-6) {
                    std::cout << "   ✓ VERIFICATION SUCCESSFUL!\n";
                } else {
                    std::cout << "   ⚠ Verification error is large\n";
                }
                
            } catch (const std::exception& e) {
                std::cout << "   ✗ Verification failed: " << e.what() << "\n";
            }
        }

        std::cout << "\n========================================\n";
        std::cout << "TEST COMPLETED SUCCESSFULLY!\n";

    } catch (const std::exception& e) {
        std::cout << "\n========================================\n";
        std::cout << "TEST FAILED!\n";
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
