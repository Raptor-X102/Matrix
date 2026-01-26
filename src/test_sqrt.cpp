#include <iostream>
#include <complex>
#include <chrono>
#include <string>
#include <iomanip>
#include "Matrix.hpp"
#include "Vector.hpp"

template<typename T> Matrix<T> create_positive_definite_matrix(int size) {
    Matrix<T> A = Matrix<T>::Generate_matrix(size, size, T(-5), T(5), 100, T(1));
    Matrix<T> B = Matrix<T>::Generate_matrix(size, size, T(-2), T(2), 100, T(1));
    A = B * B.transpose();
    for (int i = 0; i < size; ++i) {
        A(i, i) = A(i, i) + T(1);
    }
    return A;
}

Matrix<Matrix<double>> create_non_diagonal_block_matrix_with_sqrt() {
    using BlockType = Matrix<double>;
    Matrix<BlockType> M(2, 2);
    BlockType A(2, 2);
    A(0, 0) = 5.0;
    A(0, 1) = 2.0;
    A(1, 0) = 2.0;
    A(1, 1) = 5.0;
    BlockType B(2, 2);
    B(0, 0) = 3.0;
    B(0, 1) = 1.0;
    B(1, 0) = 1.0;
    B(1, 1) = 4.0;
    BlockType C(2, 2);
    C(0, 0) = 2.0;
    C(0, 1) = 1.0;
    C(1, 0) = 1.0;
    C(1, 1) = 3.0;
    BlockType D(2, 2);
    D(0, 0) = 4.0;
    D(0, 1) = 1.0;
    D(1, 0) = 1.0;
    D(1, 1) = 5.0;
    M(0, 0) = A;
    M(0, 1) = B;
    M(1, 0) = B;
    M(1, 1) = D;
    return M;
}

template<typename T> Matrix<T> generate_test_matrix(int size, T min_val, T max_val) {
    if constexpr (detail::is_matrix_v<T>) {
        using ElementType = typename T::value_type;
        int block_rows = 2;
        int block_cols = 2;
        Matrix<T> result = Matrix<T>::BlockMatrix(size, size, block_rows, block_cols);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j <= i; ++j) {
                Matrix<ElementType> block = Matrix<ElementType>::Generate_matrix(
                    block_rows,
                    block_cols,
                    static_cast<ElementType>(min_val),
                    static_cast<ElementType>(max_val));
                if (i == j) {
                    Matrix<ElementType> L =
                        Matrix<ElementType>::Zero(block_rows, block_cols);
                    for (int k = 0; k < block_rows; ++k) {
                        for (int m = 0; m <= k; ++m) {
                            L(k, m) = Matrix<ElementType>::generate_random(
                                static_cast<ElementType>(min_val / 2),
                                static_cast<ElementType>(max_val / 2));
                        }
                        L(k, k) = L(k, k) + ElementType(1);
                    }
                    block = L * L.transpose();
                }
                result(i, j) = block;
                if (i != j) {
                    result(j, i) = block.transpose();
                }
            }
        }
        return result;
    } else {
        if constexpr (std::is_arithmetic_v<T> && !detail::is_complex_v<T>) {
            Matrix<T> L = Matrix<T>::Zero(size, size);
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j <= i; ++j) {
                    L(i, j) = Matrix<T>::generate_random(min_val / 2, max_val / 2);
                }
                L(i, i) = L(i, i) + T(1);
            }
            return L * L.transpose();
        } else {
            return Matrix<T>::Generate_matrix(size, size, min_val, max_val, 100, T(1));
        }
    }
}

template<typename T> std::string type_name() {
    if constexpr (std::is_same_v<T, int>)
        return "int";
    else if constexpr (std::is_same_v<T, float>)
        return "float";
    else if constexpr (std::is_same_v<T, double>)
        return "double";
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        return "complex<float>";
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        return "complex<double>";
    else if constexpr (detail::is_matrix_v<T>)
        return "Matrix<" + type_name<typename T::value_type>() + ">";
    else
        return "unknown";
}

template<typename T>
void print_matrix_info(const Matrix<T> &A, const std::string &name) {
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << name << " (" << type_name<T>() << " " << A.get_rows() << "×"
              << A.get_cols() << ")\n";
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
        std::cout << "\n   Skipping computation\n";
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
        std::cout << "   Result obtained (may be approximate)\n";
        print_matrix_info(B, "RESULT");
        std::cout << "\n";
        std::cout << "3. VERIFICATION\n";
        try {
            auto ver_start = std::chrono::high_resolution_clock::now();
            auto C = B * B;
            auto ver_end = std::chrono::high_resolution_clock::now();
            auto ver_duration = std::chrono::duration<double>(ver_end - ver_start);
            std::cout << "   Verification time: " << std::fixed << std::setprecision(6)
                      << ver_duration.count() << " seconds\n";
            auto D = C - A;
            auto error = D.frobenius_norm();
            std::cout << "   Error norm: ";
            if constexpr (detail::is_matrix_v<decltype(error)>) {
                std::cout << error.frobenius_norm();
            } else if constexpr (detail::is_complex_v<decltype(error)>) {
                std::cout << std::abs(error);
            } else {
                std::cout << error;
            }
            std::cout << "\n";
            bool ok = false;
            if constexpr (detail::is_matrix_v<decltype(error)>) {
                ok = error.frobenius_norm() < 1e-4;
            } else if constexpr (detail::is_complex_v<decltype(error)>) {
                ok = std::abs(error) < 1e-4;
            } else {
                ok = error < 1e-4;
            }
            if (ok) {
                std::cout << "   SUCCESS: Good approximation\n";
            } else {
                std::cout << "   WARNING: Large error (approximate result)\n";
            }
        }
        catch (const std::exception& e) {
            std::cout << "   Verification failed: " << e.what() << "\n";
            std::cout << "   Result may be invalid\n";
        }
    }
    catch (const std::exception& e) {
        std::cout << "\n   ERROR during sqrt(): " << e.what() << "\n";
        std::cout << "   Trying safe_sqrt()...\n";
        try {
            auto [approx, success] = A.safe_sqrt();
            std::cout << "   Got " << (success ? "successful" : "approximate") << " result from safe_sqrt()\n";
            if (approx.get_rows() > 0 && approx.get_cols() > 0) {
                print_matrix_info(approx, "SAFE SQRT APPROXIMATION");
                std::cout << "   Note: This is a diagonal approximation\n";
                std::cout << "   It may not be an accurate square root\n";
            }
        }
        catch (const std::exception& e2) {
            std::cout << "   safe_sqrt() also failed: " << e2.what() << "\n";
        }
    }
    catch (...) {
        std::cout << "\n   ERROR: Unknown error\n";
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
            Matrix<Complex> A =
                generate_test_matrix(size, Complex(-5.0f, -5.0f), Complex(5.0f, 5.0f));
            test_sqrt_for_type(A);
            break;
        }
        case 5: {
            using Complex = std::complex<double>;
            Matrix<Complex> A =
                generate_test_matrix(size, Complex(-5.0, -5.0), Complex(5.0, 5.0));
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
            Matrix<BlockType> A =
                Matrix<BlockType>::BlockMatrix(size, size, block_rows, block_cols);
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    A(i, j) =
                        Matrix<int>::Generate_matrix(block_rows, block_cols, -5, 5);
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
            Matrix<BlockType> A =
                Matrix<BlockType>::BlockMatrix(size, size, block_rows, block_cols);
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    A(i, j) = Matrix<double>::Generate_matrix(block_rows,
                                                              block_cols,
                                                              -5.0,
                                                              5.0);
                }
            }
            test_sqrt_for_type(A);
            break;
        }
        case 8: {
            using BlockType = Matrix<double>;
            std::cout << "\n=== FIXED TEST: Block Matrices ===\n";
            std::cout << "\n1. DIAGONAL BLOCK MATRIX:\n";
            BlockType A(2, 2);
            A(0, 0) = 25.0; A(0, 1) = 0.0;
            A(1, 0) = 0.0;  A(1, 1) = 16.0;
            BlockType B(2, 2);
            B(0, 0) = 9.0;  B(0, 1) = 0.0;
            B(1, 0) = 0.0;  B(1, 1) = 36.0;
            Matrix<BlockType> M_diag(2, 2);
            BlockType zeroBlock = BlockType::Zero(2, 2);
            M_diag(0, 0) = A;
            M_diag(0, 1) = zeroBlock;
            M_diag(1, 0) = zeroBlock;
            M_diag(1, 1) = B;
            std::cout << "\nDiagonal block matrix:\n";
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    std::cout << "Block (" << i << "," << j << "):\n";
                    M_diag(i, j).detailed_print();
                }
            }
            std::cout << "\nComputing sqrt...\n";
            try {
                auto sqrt_diag = M_diag.sqrt();
                std::cout << "Success!\n";
            }
            catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n";
            }
            std::cout << "\n\n2. NON-DIAGONAL BLOCK MATRIX WITH KNOWN SQRT:\n";
            Matrix<BlockType> C_known(2, 2);
            BlockType C11(2, 2);
            C11(0, 0) = 2.0; C11(0, 1) = 1.0;
            C11(1, 0) = 1.0; C11(1, 1) = 3.0;
            BlockType C12(2, 2);
            C12(0, 0) = 1.0; C12(0, 1) = 0.0;
            C12(1, 0) = 0.0; C12(1, 1) = 1.0;
            BlockType C21(2, 2);
            C21(0, 0) = 0.0; C21(0, 1) = 1.0;
            C21(1, 0) = 1.0; C21(1, 1) = 0.0;
            BlockType C22(2, 2);
            C22(0, 0) = 3.0; C22(0, 1) = 1.0;
            C22(1, 0) = 1.0; C22(1, 1) = 2.0;
            C_known(0, 0) = C11;
            C_known(0, 1) = C12;
            C_known(1, 0) = C21;
            C_known(1, 1) = C22;
            Matrix<BlockType> M_nondiag = C_known * C_known;
            std::cout << "\nNon-diagonal block matrix M = C^2:\n";
            std::cout << "\nMatrix C (known sqrt of M):\n";
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    std::cout << "Block (" << i << "," << j << "):\n";
                    C_known(i, j).detailed_print();
                }
            }
            std::cout << "\nMatrix M = C * C:\n";
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    std::cout << "Block (" << i << "," << j << "):\n";
                    M_nondiag(i, j).detailed_print();
                }
            }
            std::cout << "\nComputing sqrt(M)...\n";
            try {
                auto computed_sqrt = M_nondiag.sqrt();
                std::cout << "\nComputed sqrt:\n";
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        std::cout << "Block (" << i << "," << j << "):\n";
                        computed_sqrt(i, j).detailed_print();
                    }
                }
                auto verification = computed_sqrt * computed_sqrt;
                double error = (verification - M_nondiag).frobenius_norm();
                std::cout << "\nVerification error: " << error << "\n";
                if (error < 1e-6) {
                    std::cout << "\nSUCCESS!\n";
                } else {
                    std::cout << "\nFAILED: Large error\n";
                }
            }
            catch (const std::exception& e) {
                std::cout << "\nError computing sqrt: " << e.what() << "\n";
                std::cout << "Trying safe_sqrt...\n";
                auto [approx, success] = M_nondiag.safe_sqrt();
                std::cout << "safe_sqrt " << (success ? "succeeded" : "returned approximation") << "\n";
                if (approx.get_rows() > 0) {
                    std::cout << "\nResult from safe_sqrt:\n";
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            std::cout << "Block (" << i << "," << j << "):\n";
                            approx(i, j).detailed_print();
                        }
                    }
                }
            }
            break;
        }
        default:
            std::cerr << "Error: Invalid choice\n";
            return 1;
        }
    } catch (const std::exception &e) {
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
