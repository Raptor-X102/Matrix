#include "Matrix.hpp"
#include <iostream>
#include <complex>
#include <iomanip>

// Вспомогательная функция для печати типа
template<typename T> std::string type_name() {
    return typeid(T).name();
}

int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Testing mixed type matrix multiplications ===\n\n";

    // 1. Test int * double
    {
        std::cout << "1. int * double multiplication:\n";
        Matrix<int> A_int = Matrix<int>::Identity(3, 3);
        Matrix<double> B_double(3, 2);
        B_double = Matrix<double>::From_vector({{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}});

        std::cout << "Matrix A (3x3 int identity):\n";
        A_int.print();
        std::cout << "\nMatrix B (3x2 double):\n";
        B_double.print();

        auto result = A_int * B_double;
        std::cout << "\nResult A * B (3x2 double):\n";
        result.print();
        std::cout << std::endl;
    }

    // 2. Test double * complex
    {
        std::cout << "2. double * complex<double> multiplication:\n";
        Matrix<double> A_double(2, 2);
        A_double = Matrix<double>::From_vector({{1.5, 2.5}, {3.5, 4.5}});

        Matrix<std::complex<double>> B_complex(2, 2);
        B_complex = Matrix<std::complex<double>>::From_vector(
            {{std::complex<double>(1, 2), std::complex<double>(3, 4)},
             {std::complex<double>(5, 6), std::complex<double>(7, 8)}});

        std::cout << "Matrix A (2x2 double):\n";
        A_double.print();
        std::cout << "\nMatrix B (2x2 complex<double>):\n";
        B_complex.print();

        auto result = A_double * B_complex;
        std::cout << "\nResult A * B (2x2 complex<double>):\n";
        result.print();
        std::cout << std::endl;
    }

    // 3. Test complex * complex
    {
        std::cout << "3. complex<float> * complex<double> multiplication:\n";
        Matrix<std::complex<float>> A_cf(2, 2);
        A_cf = Matrix<std::complex<float>>::From_vector(
            {{std::complex<float>(1, 1), std::complex<float>(2, 2)},
             {std::complex<float>(3, 3), std::complex<float>(4, 4)}});

        Matrix<std::complex<double>> B_cd(2, 2);
        B_cd = Matrix<std::complex<double>>::From_vector(
            {{std::complex<double>(0.5, 0.5), std::complex<double>(1.0, 1.0)},
             {std::complex<double>(1.5, 1.5), std::complex<double>(2.0, 2.0)}});

        std::cout << "Matrix A (2x2 complex<float>):\n";
        A_cf.print();
        std::cout << "\nMatrix B (2x2 complex<double>):\n";
        B_cd.print();

        auto result = A_cf * B_cd;
        std::cout << "\nResult A * B (2x2 complex<double>):\n";
        result.print();
        std::cout << std::endl;
    }

    // 4. Test addition mixed types
    {
        std::cout << "4. int + double addition:\n";
        Matrix<int> A(2, 2);
        A = Matrix<int>::From_vector({{1, 2}, {3, 4}});

        Matrix<double> B(2, 2);
        B = Matrix<double>::From_vector({{0.5, 1.5}, {2.5, 3.5}});

        std::cout << "Matrix A (2x2 int):\n";
        A.print();
        std::cout << "\nMatrix B (2x2 double):\n";
        B.print();

        auto result = A + B;
        std::cout << "\nResult A + B (2x2 double):\n";
        result.print();
        std::cout << std::endl;
    }

    // 5. Test subtraction mixed types
    {
        std::cout << "5. complex<double> - float subtraction:\n";
        Matrix<std::complex<double>> A(2, 2);
        A = Matrix<std::complex<double>>::From_vector(
            {{std::complex<double>(5, 5), std::complex<double>(6, 6)},
             {std::complex<double>(7, 7), std::complex<double>(8, 8)}});

        Matrix<float> B(2, 2);
        B = Matrix<float>::From_vector({{1.5f, 2.5f}, {3.5f, 4.5f}});

        std::cout << "Matrix A (2x2 complex<double>):\n";
        A.print();
        std::cout << "\nMatrix B (2x2 float):\n";
        B.print();

        auto result = A - B;
        std::cout << "\nResult A - B (2x2 complex<double>):\n";
        result.print();
        std::cout << std::endl;
    }

    // 6. Test matrix * scalar (mixed types)
    {
        std::cout << "6. Matrix<int> * double scalar:\n";
        Matrix<int> int_mat = Matrix<int>::Identity(2, 2) * 3;
        double scalar = 2.5;

        std::cout << "Matrix (2x2 int * 3):\n";
        int_mat.print();
        std::cout << "\nScalar: " << scalar << " (double)\n";

        auto result = int_mat * scalar;
        std::cout << "\nResult Matrix * " << scalar << " (2x2 double):\n";
        result.print();
        std::cout << std::endl;
    }

    // 7. Test scalar * matrix (mixed types)
    {
        std::cout << "7. float scalar * Matrix<double>:\n";
        float scalar = 1.5f;
        Matrix<double> double_mat = Matrix<double>::Identity(2, 2) * 2.0;

        std::cout << "Scalar: " << scalar << " (float)\n";
        std::cout << "\nMatrix (2x2 double identity * 2.0):\n";
        double_mat.print();

        auto result = scalar * double_mat;
        std::cout << "\nResult " << scalar << " * Matrix (2x2 double):\n";
        result.print();
        std::cout << std::endl;
    }

    // 8. Test block matrix * scalar
    {
        std::cout << "8. Block matrix (int blocks) * double scalar:\n";

        Matrix<Matrix<int>> block_int_scalar(2, 2);
        block_int_scalar(0, 0) = Matrix<int>::Identity(2, 2);
        block_int_scalar(0, 1) = Matrix<int>::Identity(2, 2) * 2;
        block_int_scalar(1, 0) = Matrix<int>::Identity(2, 2) * 3;
        block_int_scalar(1, 1) = Matrix<int>::Identity(2, 2) * 4;

        double scalar = 1.5;

        std::cout << "Block Matrix (2x2 of 2x2 int blocks):\n";
        block_int_scalar.detailed_print();
        std::cout << "\nScalar: " << scalar << " (double)\n";

        auto result = block_int_scalar * scalar;

        std::cout << "\nResult Block Matrix * " << scalar
                  << " (Matrix<Matrix<double>>):\n";
        result.detailed_print();
        std::cout << std::endl;
    }

    // 9. Test block matrix with complex numbers
    {
        std::cout << "9. Block matrix with complex<double> blocks:\n";

        Matrix<Matrix<std::complex<double>>> block_complex(2, 2);

        block_complex(0, 0) = Matrix<std::complex<double>>::Identity(2, 2);
        block_complex(0, 1) = Matrix<std::complex<double>>::From_vector(
            {{std::complex<double>(1, 1), std::complex<double>(2, 2)},
             {std::complex<double>(3, 3), std::complex<double>(4, 4)}});

        block_complex(1, 0) = Matrix<std::complex<double>>::From_vector(
            {{std::complex<double>(5, 5), std::complex<double>(6, 6)},
             {std::complex<double>(7, 7), std::complex<double>(8, 8)}});

        block_complex(1, 1) = Matrix<std::complex<double>>::Zero(2, 2);

        std::complex<double> scalar(2.0, 1.0);

        std::cout << "Block Matrix (2x2 of 2x2 complex<double> blocks):\n";
        block_complex.detailed_print();
        std::cout << "\nScalar: (" << scalar.real() << ", " << scalar.imag()
                  << ") complex<double>\n";

        auto result = block_complex * scalar;

        std::cout
            << "\nResult Block Matrix * complex scalar (Matrix<Matrix<complex<double>>>):\n";
        result.detailed_print();
        std::cout << std::endl;
    }

    // 10. Test block matrix multiplication
    {
        std::cout << "10. Block matrix * block matrix multiplication:\n";

        Matrix<Matrix<double>> A_block = Matrix<Matrix<double>>::BlockZero(2, 2, 2, 2);
        Matrix<Matrix<double>> B_block = Matrix<Matrix<double>>::BlockZero(2, 2, 2, 2);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                A_block(i, j) = Matrix<double>::Identity(2, 2) * (i * 2 + j + 1);
                B_block(i, j) = Matrix<double>::Identity(2, 2) * (i * 2 + j + 1) * 0.5;
            }
        }

        std::cout << "Matrix A (2x2 of 2x2 double blocks):\n";
        A_block.detailed_print();
        std::cout << "\nMatrix B (2x2 of 2x2 double blocks):\n";
        B_block.detailed_print();

        auto result = A_block * B_block;
        std::cout << "\nResult A * B (2x2 of 2x2 double blocks):\n";
        result.detailed_print();
        std::cout << std::endl;
    }

    // 11. Test mixed block types: int blocks * double blocks
    {
        std::cout << "11. Block matrix (int blocks) * Block matrix (double blocks):\n";

        Matrix<Matrix<int>> A_int_blocks(2, 2);
        Matrix<Matrix<double>> B_double_blocks(2, 2);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                A_int_blocks(i, j) = Matrix<int>::Identity(2, 2) * (i * 2 + j + 1);
                B_double_blocks(i, j) =
                    Matrix<double>::Identity(2, 2) * (i * 2 + j + 1) * 0.1;
            }
        }

        std::cout << "Matrix A (2x2 of 2x2 int blocks):\n";
        A_int_blocks.detailed_print();
        std::cout << "\nMatrix B (2x2 of 2x2 double blocks):\n";
        B_double_blocks.detailed_print();

        auto result = A_int_blocks * B_double_blocks;
        std::cout << "\nResult A * B (Matrix<Matrix<double>>):\n";
        result.detailed_print();
        std::cout << std::endl;
    }

    // 12. Test block matrix addition
    {
        std::cout << "12. Block matrix addition (int + double):\n";

        Matrix<Matrix<int>> A_int_blocks(2, 2);
        Matrix<Matrix<double>> B_double_blocks(2, 2);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                A_int_blocks(i, j) = Matrix<int>::Identity(2, 2) * (i * 2 + j + 1);
                B_double_blocks(i, j) =
                    Matrix<double>::Identity(2, 2) * (i * 2 + j + 1) * 1.5;
            }
        }

        std::cout << "Matrix A (2x2 of 2x2 int blocks):\n";
        A_int_blocks.detailed_print();
        std::cout << "\nMatrix B (2x2 of 2x2 double blocks):\n";
        B_double_blocks.detailed_print();

        auto result = A_int_blocks + B_double_blocks;
        std::cout << "\nResult A + B (Matrix<Matrix<double>>):\n";
        result.detailed_print();
        std::cout << std::endl;
    }

    // 13. Test block matrix with complex block operations
    {
        std::cout << "13. Complex block matrix operations:\n";

        Matrix<Matrix<std::complex<double>>> A_complex_blocks(2, 2);
        Matrix<Matrix<std::complex<double>>> B_complex_blocks(2, 2);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                std::complex<double> val_a(i * 2 + j + 1, i * 2 + j + 1);
                std::complex<double> val_b(i * 2 + j + 1, -(i * 2 + j + 1));

                A_complex_blocks(i, j) =
                    Matrix<std::complex<double>>::Identity(2, 2) * val_a;
                B_complex_blocks(i, j) =
                    Matrix<std::complex<double>>::Identity(2, 2) * val_b;
            }
        }

        std::cout << "Matrix A (2x2 of 2x2 complex<double> blocks):\n";
        A_complex_blocks.detailed_print();
        std::cout << "\nMatrix B (2x2 of 2x2 complex<double> blocks):\n";
        B_complex_blocks.detailed_print();

        auto sum = A_complex_blocks + B_complex_blocks;
        std::cout << "\nResult A + B:\n";
        sum.detailed_print();

        auto product = A_complex_blocks * B_complex_blocks;
        std::cout << "\nResult A * B:\n";
        product.detailed_print();

        auto scaled = A_complex_blocks * std::complex<double>(2.0, 0.0);
        std::cout << "\nResult A * 2.0:\n";
        scaled.detailed_print();

        std::cout << std::endl;
    }

    /* // 14. Test inverse of block matrix
     {
         std::cout << "14. Inverse of simple block matrix:\n";

         // Создаем блочную диагональную матрицу для простоты
         Matrix<Matrix<double>> block_diag(2, 2);
         block_diag(0, 0) = Matrix<double>::From_vector({{2.0, 0.0}, {0.0, 2.0}});
         block_diag(0, 1) = Matrix<double>::Zero(2, 2);
         block_diag(1, 0) = Matrix<double>::Zero(2, 2);
         block_diag(1, 1) = Matrix<double>::From_vector({{3.0, 1.0}, {0.0, 3.0}});

         std::cout << "Original block matrix (2x2 of 2x2 double blocks):\n";
         block_diag.detailed_print();

         try {
             auto inverse = block_diag.inverse();
             std::cout << "\nInverse matrix:\n";
             inverse.detailed_print();

             // Проверяем, что A * A⁻¹ ≈ I
             auto identity_check = block_diag * inverse;
             std::cout << "\nCheck A * A⁻¹ (should be identity):\n";
             identity_check.detailed_print();
         } catch (const std::exception& e) {
             std::cout << "\nError computing inverse: " << e.what() << std::endl;
         }

         std::cout << std::endl;
     }*/

    // 16. Test multi-level block matrix (matrix of matrices of matrices)
    {
        std::cout << "16. Multi-level block matrix (Matrix<Matrix<Matrix<double>>>):\n";

        // Внутренний уровень: матрицы 2x2
        Matrix<Matrix<double>> inner_block1(1, 1);
        inner_block1(0, 0) = Matrix<double>::Identity(2, 2) * 2.0;

        Matrix<Matrix<double>> inner_block2(1, 1);
        inner_block2(0, 0) = Matrix<double>::Identity(2, 2) * 3.0;

        // Внешний уровень: матрица из матриц
        Matrix<Matrix<Matrix<double>>> multi_level(2, 1);
        multi_level(0, 0) = inner_block1;
        multi_level(1, 0) = inner_block2;

        std::cout << "Multi-level matrix structure:\n";
        std::cout << "Outer: 2x1 blocks of Matrix<Matrix<double>>\n";
        std::cout << "Middle: 1x1 blocks of Matrix<double>\n";
        std::cout << "Inner: 2x2 doubles\n\n";

        std::cout << "Detailed view:\n";
        multi_level.detailed_print();

        // Умножение на скаляр
        auto scaled = multi_level * 1.5;
        std::cout << "\nMulti-level matrix * 1.5:\n";
        scaled.detailed_print();

        std::cout << std::endl;
    }

    // 17. Test AVX optimization with mixed types that become float/double
    {
        std::cout << "17. AVX optimized mixed type (int * float -> float):\n";
        Matrix<int> A(4, 4);
        Matrix<float> B(4, 4);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                A(i, j) = i * 4 + j + 1;
                B(i, j) = (i * 4 + j + 1) * 0.5f;
            }
        }

        std::cout << "Matrix A (4x4 int):\n";
        A.print();
        std::cout << "\nMatrix B (4x4 float):\n";
        B.print();

        auto result = A * B;
        std::cout << "\nResult A * B (4x4 float, should use AVX):\n";
        result.print();
        std::cout << std::endl;
    }

    // 18. Test division of block matrix by scalar
    {
        std::cout << "18. Block matrix division by scalar:\n";

        Matrix<Matrix<double>> block_mat(2, 2);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                block_mat(i, j) = Matrix<double>::Identity(2, 2) * (i * 2 + j + 1);
            }
        }

        std::cout << "Original block matrix (2x2 of 2x2 double blocks):\n";
        block_mat.detailed_print();

        double divisor = 2.0;
        auto divided = block_mat / divisor;

        std::cout << "\nAfter division by " << divisor << ":\n";
        divided.detailed_print();

        std::cout << std::endl;
    }

    // 19. Test compound operations with block matrices
    {
        std::cout << "19. Compound operations with block matrices:\n";

        Matrix<Matrix<double>> A(2, 2);
        Matrix<Matrix<double>> B(2, 2);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                A(i, j) = Matrix<double>::Identity(2, 2) * (i * 2 + j + 1);
                B(i, j) = Matrix<double>::Identity(2, 2) * (i * 2 + j + 2);
            }
        }

        std::cout << "Matrix A (2x2 of 2x2 double blocks):\n";
        A.detailed_print();
        std::cout << "\nMatrix B (2x2 of 2x2 double blocks):\n";
        B.detailed_print();

        // Сложное выражение: 2*A + B/2 - A*0.5
        auto complex_expr = A * 2.0 + B / 2.0 - A * 0.5;

        std::cout << "\nExpression: 2*A + B/2 - A*0.5\n";
        std::cout << "Result:\n";
        complex_expr.detailed_print();

        std::cout << std::endl;
    }

    // 20. Stress test: large block matrix operations
    {
        std::cout << "20. Stress test with larger block matrices:\n";

        // Создаем блочную матрицу 3x3 с блоками 2x2
        Matrix<Matrix<double>> large_block(3, 3);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                large_block(i, j) = Matrix<double>::Identity(2, 2) * (i * 3 + j + 1);
            }
        }

        std::cout << "Large block matrix 3x3 of 2x2 blocks:\n";
        std::cout << "Total size: "
                  << large_block.get_rows() * large_block(0, 0).get_rows() << "x"
                  << large_block.get_cols() * large_block(0, 0).get_cols() << "\n";
        std::cout << "Showing first block only:\n";
        large_block(0, 0).print();

        // Проверяем несколько операций
        auto scaled = large_block * 3.0;
        auto shifted = large_block + large_block * 0.5;

        std::cout << "\nFirst block scaled by 3.0:\n";
        scaled(0, 0).print();

        std::cout << "\nFirst block shifted by 0.5*itself:\n";
        shifted(0, 0).print();

        std::cout << std::endl;
    }

    std::cout << "=== All mixed type multiplication tests completed ===\n";
    return 0;
}
