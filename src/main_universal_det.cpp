#include <iostream>
#include <complex>
#include <optional>

#include "Matrix.hpp"

#ifdef TIME_MEASURE
#include "Timer.hpp"
#endif

template<typename MatType> void test_det_with_timer_and_print(int n) {
    using ValueType = typename MatType::value_type;

    std::cout << "========================================\n";
    std::cout << "Testing determinant for " << n << "x" << n << " matrix\n";

    if constexpr (detail::is_matrix_v<ValueType>) {
        std::cout << "Type: Matrix<Matrix<"
                  << typeid(typename ValueType::value_type).name() << ">>\n";
    } else {
        std::cout << "Type: Matrix<" << typeid(ValueType).name() << ">\n";
    }
    std::cout << "========================================\n";

    // Генерация матрицы
    std::cout << "\n1. Generating matrix...\n";
    MatType matrix;

    try {
        if constexpr (detail::is_matrix_v<ValueType>) {
            // Для матрицы из матриц
            int sub_n;
            std::cout << "   Enter sub-matrix size: ";
            std::cin >> sub_n;

            using InnerType = typename ValueType::value_type;

            // Создаем простые диагональные матрицы для теста
            ValueType min_block =
                ValueType::Diagonal(sub_n, sub_n, static_cast<InnerType>(1));
            ValueType max_block =
                ValueType::Diagonal(sub_n, sub_n, static_cast<InnerType>(2));

            // Делаем min_block единичной матрицей для простоты
            min_block = ValueType::Identity(sub_n, sub_n);

            matrix = MatType::Generate_matrix(n,
                                              n,
                                              min_block,
                                              max_block,
                                              5,
                                              ValueType::Identity(sub_n, sub_n));
        } else {
            // Для скалярных типов
            ValueType min_val = static_cast<ValueType>(-2);
            ValueType max_val = static_cast<ValueType>(2);

            if constexpr (std::is_same_v<ValueType, int>) {
                min_val = -5;
                max_val = 5;
            }

            matrix = MatType::Generate_matrix(n,
                                              n,
                                              min_val,
                                              max_val,
                                              10,
                                              static_cast<ValueType>(1));
        }

        std::cout << "   ✓ Matrix generated successfully\n";
    } catch (const std::exception &e) {
        std::cout << "   ✗ Failed to generate matrix: " << e.what() << "\n";
        return;
    }

    // Показываем матрицу
    std::cout << "\n2. Generated matrix:\n";
    if (n <= 5) {
        if constexpr (detail::is_matrix_v<ValueType>) {
            matrix.detailed_print();
        } else {
            matrix.precise_print(4);
        }
    } else {
        std::cout << "   [Matrix too large to display]\n";
        // Показываем хотя бы размер
        std::cout << "   Size: " << matrix.get_rows() << "x" << matrix.get_cols();
        if constexpr (detail::is_matrix_v<ValueType>) {
            if (matrix.get_rows() > 0 && matrix.get_cols() > 0) {
                std::cout << ", block size: " << matrix(0, 0).get_rows() << "x"
                          << matrix(0, 0).get_cols();
            }
        }
        std::cout << "\n";
    }

    // Вычисляем определитель
    std::cout << "\n3. Computing determinant...\n";
    std::optional<ValueType> computed_det;

    try {
#ifdef TIME_MEASURE
        {
            Timer timer;
            computed_det = matrix.det();
        }
#else
        computed_det = matrix.det();
#endif
        std::cout << "   ✓ Determinant computation completed\n";
    } catch (const std::exception &e) {
        std::cout << "   ✗ Determinant computation failed: " << e.what() << "\n";
        return;
    }

    // Показываем результат
    std::cout << "\n4. Result:\n";
    if (computed_det) {
        if constexpr (detail::is_matrix_v<ValueType>) {
            std::cout << "   Determinant is a " << (*computed_det).get_rows() << "x"
                      << (*computed_det).get_cols() << " matrix:\n";
            if ((*computed_det).get_rows() <= 5 && (*computed_det).get_cols() <= 5) {
                (*computed_det).precise_print(6);
            } else {
                std::cout << "   [Matrix too large to display]\n";
            }
        } else {
            std::cout << "   Determinant = " << *computed_det << "\n";
        }

        // Дополнительная проверка для скалярных типов
        if constexpr (!detail::is_matrix_v<ValueType>) {
            std::cout << "   Absolute value: ";
            if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
                std::cout << std::abs(*computed_det) << "\n";
            } else {
                std::cout << std::abs(static_cast<double>(*computed_det)) << "\n";
            }
        } else {
            std::cout << "   ✗ Determinant computation returned nullopt\n";
        }
    }

    // Проверяем кеширование
    std::cout << "\n5. Checking determinant caching...\n";
    auto cached_det = matrix.get_determinant();
    if (cached_det) {
        std::cout << "   ✓ Determinant is cached\n";

        // Сравниваем с вычисленным
        if (computed_det) {
            bool equal = false;
            if constexpr (detail::is_matrix_v<ValueType>) {
                // Для матриц проверяем поэлементно
                if ((*computed_det).get_rows() == (*cached_det).get_rows()
                    && (*computed_det).get_cols() == (*cached_det).get_cols()) {
                    equal = true;
                    for (int i = 0; i < (*computed_det).get_rows() && equal; ++i) {
                        for (int j = 0; j < (*computed_det).get_cols() && equal; ++j) {
                            if (!Matrix<ValueType>::is_equal((*computed_det)(i, j),
                                                             (*cached_det)(i, j))) {
                                equal = false;
                            }
                        }
                    }
                }
            } else {
                equal = Matrix<ValueType>::is_equal(*computed_det, *cached_det);
            }

            if (equal) {
                std::cout << "   ✓ Cached value matches computed value\n";
            } else {
                std::cout << "   ✗ Cached value differs from computed value!\n";
            }
        }
    } else {
        std::cout << "   ✗ Determinant is not cached\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Test completed.\n";
}

int main() {
    std::cout << "MATRIX DETERMINANT TEST\n";
    std::cout << "=======================\n\n";

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

        std::cout << "Choose type:\n";
        std::cout << "  1. int\n";
        std::cout << "  2. double\n";
        std::cout << "  3. complex\n";
        std::cout << "  4. matrix of matrices\n";
        std::cout << "Choice: ";
        std::cin >> type_choice;

        try {
            if (type_choice == "1" || type_choice == "int") {
                test_det_with_timer_and_print<Matrix<int>>(n);
            } else if (type_choice == "2" || type_choice == "double") {
                test_det_with_timer_and_print<Matrix<double>>(n);
            } else if (type_choice == "3" || type_choice == "complex") {
                test_det_with_timer_and_print<Matrix<std::complex<double>>>(n);
            } else if (type_choice == "4" || type_choice == "matrix") {
                std::string inner_type;
                std::cout << "Choose inner matrix type (int/double/complex): ";
                std::cin >> inner_type;

                if (inner_type == "int") {
                    test_det_with_timer_and_print<Matrix<Matrix<int>>>(n);
                } else if (inner_type == "double") {
                    test_det_with_timer_and_print<Matrix<Matrix<double>>>(n);
                } else if (inner_type == "complex") {
                    test_det_with_timer_and_print<Matrix<Matrix<std::complex<double>>>>(
                        n);
                } else {
                    std::cout << "Unknown inner type\n";
                }
            } else {
                std::cout << "Unknown type choice\n";
            }
        } catch (const std::exception &e) {
            std::cout << "\n✗ Exception caught: " << e.what() << "\n";
        }
    }

    return 0;
}
