#include <iostream>
#include <optional>

#include "Matrix.hpp"
#include "Matrix_1d_array.hpp"

#ifdef TIME_MEASURE
#include "Timer.hpp"
#endif

template<typename MatType>
void test_det_with_timer_and_print(const std::string& name, int n) {
    std::cout << "Testing " << name << "...\n";

    auto matrix = MatType::Generate_matrix(n, n, 0.1, 2.0);

    // Вывод матрицы (если n не слишком большой)
    if (n <= 10) {
        std::cout << "Generated matrix:\n";
        matrix.precise_print(5);
        std::cout << "\n";
    }

    // Получаем эталонный определитель
    auto expected_det = matrix.get_determinant();

    std::cout << "Expected determinant: ";
    if (expected_det) {
        std::cout << *expected_det << "\n";
    } else {
        std::cout << "(not available)\n";
    }

    // Считаем определитель
    std::optional<typename MatType::value_type> computed_det;

    #ifdef TIME_MEASURE
    {
        Timer timer;  // Таймер начинает измерение только тут
        computed_det = matrix.det();
    }
    #else
    computed_det = matrix.det();
    #endif

    std::cout << "Computed determinant: ";
    if (computed_det) {
        std::cout << *computed_det << "\n";
    } else {
        std::cout << "(failed)\n";
    }

    if (expected_det && computed_det) {
        if (std::abs(*expected_det - *computed_det) > 1e-9) {
            std::cerr << "ERROR: " << name << " computed determinant differs from expected!\n";
            std::cerr << "Expected: " << *expected_det << ", Got: " << *computed_det << "\n";
        } else {
            std::cout << name << " determinant: OK.\n";
        }
    } else {
        std::cerr << "ERROR: Either expected or computed determinant is missing.\n";
    }

    std::cout << "\n";
}

int main() {
    int n;
    std::cin >> n;

    // === Запускаем обе реализации ===
    test_det_with_timer_and_print<Matrix<double>>("Matrix", n);
    test_det_with_timer_and_print<Matrix_1d<double>>("Matrix_1d", n);

    return 0;
}
