#include <iostream>
#include <complex>
#include <optional>

#include "Matrix.hpp"

#ifdef TIME_MEASURE
#include "Timer.hpp"
#endif

template <typename MatType>
void test_det_with_timer_and_print(int n) {
    using ValueType = typename MatType::value_type;

    std::cout << "Generating matrix " << n << "x" << n << "...\n";

    auto matrix = [&]() -> MatType {
        if constexpr (detail::is_matrix_v<ValueType>) {
            // Для матрицы из матриц
            int sub_n;
            std::cout << "Enter sub-matrix size: ";
            std::cin >> sub_n;
            
            // Создаем диагональные матрицы для min/max значений
            using InnerType = typename ValueType::value_type;
            
            // Приводим -10 и 10 к нужному типу
            InnerType min_val = static_cast<InnerType>(-10);
            InnerType max_val = static_cast<InnerType>(10);
            
            // Создаем диагональные матрицы
            auto min_matrix = ValueType::Diagonal(sub_n, sub_n, min_val);
            auto max_matrix = ValueType::Diagonal(sub_n, sub_n, max_val);
            
            return MatType::Generate_matrix(n, n, min_matrix, max_matrix);
        } else {
            // Для всех остальных типов (int, double, complex, etc.)
            ValueType min_val = static_cast<ValueType>(-10);
            ValueType max_val = static_cast<ValueType>(10);
            
            return MatType::Generate_matrix(n, n, min_val, max_val);
        }
    }();

    // В main_universal_det.cpp, в функции test_det_with_timer_and_print:

if (n <= 5) {
    std::cout << "Generated matrix:\n";
    if constexpr (detail::is_matrix_v<ValueType>) {
        // Для матриц матриц используем detailed_print
        matrix.detailed_print();
    } else {
        matrix.precise_print(6);
    }
    std::cout << "\n";
}

    std::optional<ValueType> computed_det;

#ifdef TIME_MEASURE
    {
        Timer timer;
        computed_det = matrix.det();
    }
#else
    computed_det = matrix.det();
#endif

    std::cout << "Computed determinant: ";
    if (computed_det) {
        if constexpr (detail::is_matrix_v<ValueType>) {
            if ((*computed_det).get_rows() <= 5 && (*computed_det).get_cols() <= 5) {
                (*computed_det).precise_print(6);
            } else {
                std::cout << "[Matrix too large to display]";
            }
        } else {
            std::cout << *computed_det;
        }
        std::cout << "\n";
    } else {
        std::cout << "(failed)\n";
    }

    if (computed_det) {
        std::cout << "Determinant: OK.\n";
    } else {
        std::cerr << "ERROR: Determinant computation failed.\n";
    }

    std::cout << "\n";
}

int main() {
    int n;
    std::string type_choice;
    std::string inner_type;

    std::cout << "Enter matrix size: ";
    std::cin >> n;
    std::cout << "Choose type (int/double/complex/matrix): ";
    std::cin >> type_choice;

    if (type_choice == "int") {
        test_det_with_timer_and_print<Matrix<int>>(n);
    } else if (type_choice == "double") {
        test_det_with_timer_and_print<Matrix<double>>(n);
    } else if (type_choice == "complex") {
        test_det_with_timer_and_print<Matrix<std::complex<double>>>(n);
    } else if (type_choice == "matrix") {
        std::cout << "Choose inner type (int/double/complex): ";
        std::cin >> inner_type;

        if (inner_type == "int") {
            test_det_with_timer_and_print<Matrix<Matrix<int>>>(n);
        } else if (inner_type == "double") {
            test_det_with_timer_and_print<Matrix<Matrix<double>>>(n);
        } else if (inner_type == "complex") {
            test_det_with_timer_and_print<Matrix<Matrix<std::complex<double>>>>(n);
        } else {
            std::cerr << "Unknown inner type.\n";
            return 1;
        }
    } else {
        std::cerr << "Unknown type.\n";
        return 1;
    }

    return 0;
}
