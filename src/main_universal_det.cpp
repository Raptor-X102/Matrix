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
        if constexpr (std::is_same_v<ValueType, int>) {
            // Для int используем целочисленные литералы
            return MatType::Generate_matrix(n, n, ValueType{-10}, ValueType{10});
        } else if constexpr (std::is_same_v<ValueType, double>) {
            // Для double используем double литералы
            return MatType::Generate_matrix(n, n, ValueType{-10.0}, ValueType{10.0});
        } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
            // Для complex
            return MatType::Generate_matrix(n, n, 
                ValueType{-10.0, -10.0}, 
                ValueType{10.0, 10.0});
        } else if constexpr (detail::is_matrix_v<ValueType>) {
            // Для матрицы из матриц
            int sub_n;
            std::cout << "Enter sub-matrix size: ";
            std::cin >> sub_n;
            
            using InnerType = typename ValueType::value_type;
            
            if constexpr (std::is_same_v<InnerType, int>) {
                // Внутренние матрицы int
                auto min_matrix = ValueType::Generate_matrix(sub_n, sub_n, -10, -10);
                auto max_matrix = ValueType::Generate_matrix(sub_n, sub_n, 10, 10);
                return MatType::Generate_matrix(n, n, min_matrix, max_matrix);
            } else if constexpr (std::is_same_v<InnerType, double>) {
                // Внутренние матрицы double
                auto min_matrix = ValueType::Generate_matrix(sub_n, sub_n, -10.0, -10.0);
                auto max_matrix = ValueType::Generate_matrix(sub_n, sub_n, 10.0, 10.0);
                return MatType::Generate_matrix(n, n, min_matrix, max_matrix);
            } else if constexpr (std::is_same_v<InnerType, std::complex<double>>) {
                // Внутренние матрицы complex
                auto min_matrix = ValueType::Generate_matrix(sub_n, sub_n, 
                    InnerType{-10.0, -10.0}, 
                    InnerType{-10.0, -10.0});
                auto max_matrix = ValueType::Generate_matrix(sub_n, sub_n,
                    InnerType{10.0, 10.0},
                    InnerType{10.0, 10.0});
                return MatType::Generate_matrix(n, n, min_matrix, max_matrix);
            } else {
                // Для других типов
                auto min_val = InnerType{-10};
                auto max_val = InnerType{10};
                auto min_matrix = ValueType::Generate_matrix(sub_n, sub_n, min_val, min_val);
                auto max_matrix = ValueType::Generate_matrix(sub_n, sub_n, max_val, max_val);
                return MatType::Generate_matrix(n, n, min_matrix, max_matrix);
            }
        } else {
            // Для любых других типов используем static_cast
            return MatType::Generate_matrix(n, n, 
                ValueType{static_cast<ValueType>(-10)}, 
                ValueType{static_cast<ValueType>(10)});
        }
    }();

    if (n <= 5) {
        std::cout << "Generated matrix:\n";
        matrix.precise_print(6);
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
