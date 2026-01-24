#include <iostream>
#include <complex>
#include <chrono>
#include <string>
#include <iomanip>
#include "Matrix.hpp"
#include "Vector.hpp"

template<typename T>
Matrix<T> create_positive_definite_matrix(int size) {
    // Создаем случайную матрицу
    Matrix<T> A = Matrix<T>::Generate_matrix(size, size, T(-5), T(5), 100, T(1));

    // Делаем ее положительно определенной: A = B * B^T
    Matrix<T> B = Matrix<T>::Generate_matrix(size, size, T(-2), T(2), 100, T(1));
    A = B * B.transpose();

    // Добавляем небольшой сдвиг к диагонали для гарантии невырожденности
    for (int i = 0; i < size; ++i) {
        A(i, i) = A(i, i) + T(1);
    }

    return A;
}

template<typename T>
Matrix<T> generate_test_matrix(int size, T min_val, T max_val) {
    if constexpr (detail::is_matrix_v<T>) {
        using ElementType = typename T::value_type;
        int block_rows = 2;
        int block_cols = 2;
        Matrix<T> result = Matrix<T>::BlockMatrix(size, size, block_rows, block_cols);
        
        // Для блочных матриц создаем положительно определенную матрицу
        // A = L * L^T, где L - нижнетреугольная матрица с ненулевой диагональю
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j <= i; ++j) {  // Только нижний треугольник
                Matrix<ElementType> block = Matrix<ElementType>::Generate_matrix(
                    block_rows, block_cols, 
                    static_cast<ElementType>(min_val), 
                    static_cast<ElementType>(max_val)
                );
                
                // Для диагональных блоков делаем их положительно определенными
                if (i == j) {
                    // Создаем нижнетреугольную матрицу L
                    Matrix<ElementType> L = Matrix<ElementType>::Zero(block_rows, block_cols);
                    for (int k = 0; k < block_rows; ++k) {
                        for (int m = 0; m <= k; ++m) {
                            L(k, m) = Matrix<ElementType>::generate_random(
                                static_cast<ElementType>(min_val/2), 
                                static_cast<ElementType>(max_val/2)
                            );
                        }
                        // Гарантируем, что диагональные элементы не нулевые
                        L(k, k) = L(k, k) + ElementType(1);
                    }
                    // A = L * L^T - гарантированно положительно определенная
                    block = L * L.transpose();
                }
                result(i, j) = block;
                
                // Симметричное отражение для верхнего треугольника
                if (i != j) {
                    result(j, i) = block.transpose();
                }
            }
        }
        return result;
    } else {
        // Для скалярных типов создаем положительно определенную матрицу
        if constexpr (std::is_arithmetic_v<T> && !detail::is_complex_v<T>) {
            // Создаем нижнетреугольную матрицу L
            Matrix<T> L = Matrix<T>::Zero(size, size);
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j <= i; ++j) {
                    L(i, j) = Matrix<T>::generate_random(min_val/2, max_val/2);
                }
                // Гарантируем, что диагональные элементы не нулевые
                L(i, i) = L(i, i) + T(1);
            }
            // A = L * L^T - гарантированно положительно определенная
            return L * L.transpose();
        } else {
            // Для комплексных чисел оставляем как было
            return Matrix<T>::Generate_matrix(size, size, min_val, max_val, 100, T(1));
        }
    }
}

template<typename T>
std::string type_name() {
    if constexpr (std::is_same_v<T, int>) return "int";
    else if constexpr (std::is_same_v<T, float>) return "float";
    else if constexpr (std::is_same_v<T, double>) return "double";
    else if constexpr (std::is_same_v<T, std::complex<float>>) return "complex<float>";
    else if constexpr (std::is_same_v<T, std::complex<double>>) return "complex<double>";
    else if constexpr (detail::is_matrix_v<T>) return "Matrix<" + type_name<typename T::value_type>() + ">";
    else return "unknown";
}

template<typename T>
void print_matrix_info(const Matrix<T>& A, const std::string& name) {
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << name << " (" << type_name<T>() << " " 
              << A.get_rows() << "×" << A.get_cols() << ")\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    
    if (A.get_rows() <= 6 && A.get_cols() <= 6) {
        std::cout << "Matrix:\n";
        A.detailed_print();
    } else {
        std::cout << "Matrix (first 6×6 block):\n";
        A.print(6);
    }
}

template<typename T>
void test_sqrt_for_type(const Matrix<T>& A) {
    using ResultType = typename Matrix<T>::template sqrt_return_type<T>;
    
    print_matrix_info(A, "INPUT");
    
    std::cout << "\n";
    std::cout << "1. CHECKING IF SQUARE ROOT EXISTS\n";
    
    bool has_sqrt = A.has_square_root();
    std::cout << "   Has square root: " << (has_sqrt ? "YES" : "NO") << "\n";
    
    if (!has_sqrt) {
        std::cout << "\n   Skipping computation (no square root exists)\n";
        return;
    }
    
    std::cout << "\n";
    std::cout << "2. COMPUTING SQUARE ROOT (type: " << type_name<ResultType>() << ")\n";
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto B = A.sqrt();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);
        
        std::cout << "   Computation time: " << std::fixed << std::setprecision(6) 
                  << duration.count() << " seconds\n";
        
        std::cout << "   Computation completed (may be approximate)\n";
        
        // Проверяем, не содержит ли результат NaN/Inf
        bool has_invalid = false;
        for (int i = 0; i < B.get_rows() && !has_invalid; ++i) {
            for (int j = 0; j < B.get_cols() && !has_invalid; ++j) {
                if constexpr (std::is_floating_point_v<ResultType> || 
                              detail::is_complex_v<ResultType>) {
                    using std::isnan;
                    using std::isinf;
                    if constexpr (detail::is_complex_v<ResultType>) {
                        if (isnan(std::real(B(i, j))) || isnan(std::imag(B(i, j))) ||
                            isinf(std::real(B(i, j))) || isinf(std::imag(B(i, j)))) {
                            has_invalid = true;
                        }
                    } else {
                        if (isnan(B(i, j)) || isinf(B(i, j))) {
                            has_invalid = true;
                        }
                    }
                }
            }
        }
        
        if (has_invalid) {
            std::cout << "\n   WARNING: Result contains NaN or Inf values\n";
            // Но все равно показываем матрицу
        }
        
        if (B.get_rows() <= 6 && B.get_cols() <= 6) {
            std::cout << "\n   Result matrix:\n";
            B.detailed_print();
        }
        
        std::cout << "\n";
        std::cout << "3. VERIFICATION: (sqrt(A))² should equal A\n";
        
        try {
            auto ver_start = std::chrono::high_resolution_clock::now();
            auto C = B * B;
            auto ver_end = std::chrono::high_resolution_clock::now();
            auto ver_duration = std::chrono::duration<double>(ver_end - ver_start);
            
            std::cout << "   Verification time: " << std::fixed << std::setprecision(6)
                      << ver_duration.count() << " seconds\n";
            
            auto D = C - A;
            auto error = D.frobenius_norm();
            
            std::cout << "\n   Frobenius norm of difference: ";
            
            if constexpr (detail::is_matrix_v<decltype(error)>) {
                using ErrorType = decltype(error);
                using ElemType = typename ErrorType::value_type;
                ElemType error_norm = error.frobenius_norm();
                
                if constexpr (detail::is_complex_v<ElemType>) {
                    std::cout << std::abs(error_norm);
                } else {
                    std::cout << error_norm;
                }
            } else if constexpr (detail::is_complex_v<decltype(error)>) {
                std::cout << std::abs(error);
            } else {
                std::cout << error;
            }
            std::cout << "\n";
            
            bool verification_passed = false;
            
            if constexpr (detail::is_matrix_v<decltype(error)>) {
                using ErrorType = decltype(error);
                using ElemType = typename ErrorType::value_type;
                ElemType error_norm = error.frobenius_norm();
                
                if constexpr (detail::is_complex_v<ElemType>) {
                    verification_passed = std::abs(error_norm) < 1e-4;  // Более либерально
                } else {
                    verification_passed = error_norm < 1e-4;
                }
            } else if constexpr (detail::is_complex_v<decltype(error)>) {
                verification_passed = std::abs(error) < 1e-4;
            } else {
                verification_passed = error < 1e-4;
            }
            
            if (verification_passed) {
                std::cout << "\n   VERIFICATION PASSED: (sqrt(A))² ≈ A\n";
            } else {
                std::cout << "\n   VERIFICATION FAILED or APPROXIMATE: Error too large\n";
                std::cout << "   (Result may be an approximate square root)\n";
            }
        }
        catch (const std::exception& e) {
            std::cout << "\n   Verification failed: " << e.what() << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cout << "\n   ERROR: " << e.what() << "\n";
    }
    catch (...) {
        std::cout << "\n   ERROR: Unknown error occurred\n";
    }
    
    std::cout << "\n";
}

int main() {
    std::cout << "MATRIX SQUARE ROOT TESTER\n\n";
    
    int size;
    std::cout << "Enter matrix size (n for n×n matrix): ";
    std::cin >> size;
    
    if (size <= 0) {
        std::cerr << "Error: Size must be positive\n";
        return 1;
    }
    
    std::cout << "\nSelect matrix element type:\n";
    std::cout << "1. int\n";
    std::cout << "2. float\n";
    std::cout << "3. double\n";
    std::cout << "4. complex<float>\n";
    std::cout << "5. complex<double>\n";
    std::cout << "6. Matrix<int>\n";
    std::cout << "7. Matrix<double>\n";
    std::cout << "8. FIXED TEST: Simple block-diagonal matrix\n";
    std::cout << "Choice [1-8]: ";
    
    int choice;
    std::cin >> choice;
    
    std::cout << "\nTEST STARTING\n";
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    try {
        switch (choice) {
            case 1: {
                Matrix<int> A = generate_test_matrix(size, -10, 10);
                test_sqrt_for_type(A);
                break;
            }
            case 2: {
                Matrix<float> A = generate_test_matrix(size, -10.0f, 10.0f);
                test_sqrt_for_type(A);
                break;
            }
            case 3: {
                Matrix<double> A = generate_test_matrix(size, -10.0, 10.0);
                test_sqrt_for_type(A);
                break;
            }
            case 4: {
                using Complex = std::complex<float>;
                Matrix<Complex> A = generate_test_matrix(
                    size, 
                    Complex(-5.0f, -5.0f), 
                    Complex(5.0f, 5.0f)
                );
                test_sqrt_for_type(A);
                break;
            }
            case 5: {
                using Complex = std::complex<double>;
                Matrix<Complex> A = generate_test_matrix(
                    size, 
                    Complex(-5.0, -5.0), 
                    Complex(5.0, 5.0)
                );
                test_sqrt_for_type(A);
                break;
            }
            case 6: {
                using BlockType = Matrix<int>;
                int block_rows;
                int block_cols;
                std::cout << "Enter block rows: ";
                std::cin >> block_rows;
                std::cout << "Enter block cols: ";
                std::cin >> block_cols;
                
                if (block_rows <= 0 || block_cols <= 0) {
                    std::cerr << "Error: Block dimensions must be positive\n";
                    return 1;
                }
                
                Matrix<BlockType> A = Matrix<BlockType>::BlockMatrix(size, size, block_rows, block_cols);
                for (int i = 0; i < size; ++i) {
                    for (int j = 0; j < size; ++j) {
                        A(i, j) = Matrix<int>::Generate_matrix(block_rows, block_cols, -5, 5);
                    }
                }
                test_sqrt_for_type(A);
                break;
            }
            case 7: {
                using BlockType = Matrix<double>;
                int block_rows;
                int block_cols;
                std::cout << "Enter block rows: ";
                std::cin >> block_rows;
                std::cout << "Enter block cols: ";
                std::cin >> block_cols;
                
                if (block_rows <= 0 || block_cols <= 0) {
                    std::cerr << "Error: Block dimensions must be positive\n";
                    return 1;
                }
                
                Matrix<BlockType> A = Matrix<BlockType>::BlockMatrix(size, size, block_rows, block_cols);
                for (int i = 0; i < size; ++i) {
                    for (int j = 0; j < size; ++j) {
                        A(i, j) = Matrix<double>::Generate_matrix(block_rows, block_cols, -5.0, 5.0);
                    }
                }
                test_sqrt_for_type(A);
                break;
            }
            case 8: {
                // FIXED TEST: Simple block-diagonal matrix
                using BlockType = Matrix<double>;
                
                std::cout << "\n=== FIXED TEST: Simple Block-Diagonal Matrix ===\n";
                
                // Блок A: положительно определенная матрица 2x2 с ИЗВЕСТНЫМ квадратным корнем
                BlockType A(2, 2);
                A(0, 0) = 25.0;  // Простые числа для легкого вычисления
                A(0, 1) = 0.0;
                A(1, 0) = 0.0;
                A(1, 1) = 16.0;
                
                std::cout << "\nBlock A (diagonal for simplicity):\n";
                A.detailed_print();
                
                // Известный квадратный корень блока A (диагональная матрица)
                BlockType sqrtA(2, 2);
                sqrtA(0, 0) = 5.0;  // sqrt(25)
                sqrtA(0, 1) = 0.0;
                sqrtA(1, 0) = 0.0;
                sqrtA(1, 1) = 4.0;  // sqrt(16)
                
                std::cout << "\nKnown sqrt of A (should be diagonal [5, 0; 0, 4]):\n";
                sqrtA.detailed_print();
                
                // Проверяем: sqrtA * sqrtA должно равняться A
                BlockType checkA = sqrtA * sqrtA;
                std::cout << "\nVerification: sqrtA * sqrtA =\n";
                checkA.detailed_print();
                
                double errorA = (checkA - A).frobenius_norm();
                std::cout << "Error: " << errorA << " (should be 0)\n";
                
                // Блок B: другая диагональная матрица 2x2
                BlockType B(2, 2);
                B(0, 0) = 9.0;
                B(0, 1) = 0.0;
                B(1, 0) = 0.0;
                B(1, 1) = 36.0;
                
                std::cout << "\n\nBlock B (diagonal for simplicity):\n";
                B.detailed_print();
                
                // Известный квадратный корень блока B
                BlockType sqrtB(2, 2);
                sqrtB(0, 0) = 3.0;   // sqrt(9)
                sqrtB(0, 1) = 0.0;
                sqrtB(1, 0) = 0.0;
                sqrtB(1, 1) = 6.0;   // sqrt(36)
                
                std::cout << "\nKnown sqrt of B (should be diagonal [3, 0; 0, 6]):\n";
                sqrtB.detailed_print();
                
                // Проверяем: sqrtB * sqrtB должно равняться B
                BlockType checkB = sqrtB * sqrtB;
                std::cout << "\nVerification: sqrtB * sqrtB =\n";
                checkB.detailed_print();
                
                double errorB = (checkB - B).frobenius_norm();
                std::cout << "Error: " << errorB << " (should be 0)\n";
                
                // Создаем блочно-диагональную матрицу 2x2
                Matrix<BlockType> M(2, 2);
                
                // Инициализируем правильно с нулевыми блоками
                BlockType zeroBlock = BlockType::Zero(2, 2);
                
                M(0, 0) = A;
                M(0, 1) = zeroBlock;
                M(1, 0) = zeroBlock;
                M(1, 1) = B;
                
                std::cout << "\n\nBlock-diagonal matrix M (2x2 blocks of 2x2 doubles):\n";
                // Используем простой вывод для избежания проблем с форматом
                std::cout << "M(0,0):\n";
                M(0, 0).detailed_print();
                std::cout << "M(0,1):\n";
                M(0, 1).detailed_print();
                std::cout << "M(1,0):\n";
                M(1, 0).detailed_print();
                std::cout << "M(1,1):\n";
                M(1, 1).detailed_print();
                
                // Известный квадратный корень всей матрицы
                Matrix<BlockType> expectedSqrt(2, 2);
                expectedSqrt(0, 0) = sqrtA;
                expectedSqrt(0, 1) = zeroBlock;
                expectedSqrt(1, 0) = zeroBlock;
                expectedSqrt(1, 1) = sqrtB;
                
                std::cout << "\n\nExpected sqrt of M (diagonal blocks only):\n";
                std::cout << "expectedSqrt(0,0):\n";
                expectedSqrt(0, 0).detailed_print();
                std::cout << "expectedSqrt(1,1):\n";
                expectedSqrt(1, 1).detailed_print();
                
                // Тестируем нашу реализацию
                std::cout << "\n\n=== Testing our implementation ===\n";
                try {
                    std::cout << "Calling M.sqrt()...\n";
                    Matrix<BlockType> computedSqrt = M.sqrt();
                    
                    std::cout << "\nComputed sqrt of M:\n";
                    std::cout << "computedSqrt(0,0):\n";
                    computedSqrt(0, 0).detailed_print();
                    std::cout << "computedSqrt(0,1):\n";
                    computedSqrt(0, 1).detailed_print();
                    std::cout << "computedSqrt(1,0):\n";
                    computedSqrt(1, 0).detailed_print();
                    std::cout << "computedSqrt(1,1):\n";
                    computedSqrt(1, 1).detailed_print();
                    
                    // Проверяем результат
                    std::cout << "\nVerification: computedSqrt * computedSqrt:\n";
                    try {
                        Matrix<BlockType> verification = computedSqrt * computedSqrt;
                        
                        std::cout << "Verification(0,0):\n";
                        verification(0, 0).detailed_print();
                        std::cout << "Should be:\n";
                        M(0, 0).detailed_print();
                        
                        std::cout << "\nVerification(1,1):\n";
                        verification(1, 1).detailed_print();
                        std::cout << "Should be:\n";
                        M(1, 1).detailed_print();
                        
                        double totalError = (verification - M).frobenius_norm();
                        std::cout << "\nTotal error: " << totalError << "\n";
                        
                        if (totalError < 1e-6) {
                            std::cout << "\nSUCCESS: Square root computed correctly!\n";
                        } else {
                            std::cout << "\nWARNING: Large error in computed square root\n";
                        }
                        
                        // Сравниваем с ожидаемым результатом
                        double diffFromExpected = (computedSqrt - expectedSqrt).frobenius_norm();
                        std::cout << "Difference from expected result: " << diffFromExpected << "\n";
                        
                    } catch (const std::exception& e) {
                        std::cout << "\nERROR in verification: " << e.what() << "\n";
                    }
                    
                } catch (const std::exception& e) {
                    std::cout << "\nERROR in sqrt computation: " << e.what() << "\n";
                    
                    // Попробуем вычислить вручную через диагональные блоки
                    std::cout << "\nTrying manual computation (diagonal blocks only):\n";
                    Matrix<BlockType> manualSqrt(2, 2);
                    
                    try {
                        std::cout << "Computing sqrt of block A...\n";
                        manualSqrt(0, 0) = A.sqrt();
                        std::cout << "Result:\n";
                        manualSqrt(0, 0).detailed_print();
                    } catch (const std::exception& e) {
                        std::cout << "Failed to compute sqrt of A: " << e.what() << "\n";
                    }
                    
                    try {
                        std::cout << "\nComputing sqrt of block B...\n";
                        manualSqrt(1, 1) = B.sqrt();
                        std::cout << "Result:\n";
                        manualSqrt(1, 1).detailed_print();
                    } catch (const std::exception& e) {
                        std::cout << "Failed to compute sqrt of B: " << e.what() << "\n";
                    }
                    
                    manualSqrt(0, 1) = zeroBlock;
                    manualSqrt(1, 0) = zeroBlock;
                }
                break;
            }
            default:
                std::cerr << "Error: Invalid choice\n";
                return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << "\n";
        return 1;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration<double>(total_end - total_start);
    
    std::cout << "\nTEST COMPLETE\n";
    std::cout << "Total execution time: " << std::fixed << std::setprecision(3) 
              << total_duration.count() << " seconds\n";
    
    return 0;
}
