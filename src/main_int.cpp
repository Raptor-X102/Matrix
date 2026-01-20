#include <iostream>
#include <optional>

#include "Matrix.hpp"

#ifdef TIME_MEASURE
#include "Timer.hpp"
#endif

using ScalarType = int;

template<typename MatType>
void test_det_with_timer_and_print(const std::string &name, int n) {
    std::cout << "Testing " << name << "...\n";

    auto matrix = MatType::Generate_binary_matrix(n, n);

    if (n <= 10) {
        std::cout << "Generated matrix:\n";
        matrix.precise_print(5);
        std::cout << "\n";
    }

    auto expected_det = matrix.get_determinant();

    std::cout << "Expected determinant: ";
    if (expected_det) {
        std::cout << *expected_det << "\n";
    } else {
        std::cout << "(not available)\n";
    }

    std::optional<typename MatType::value_type> computed_det;

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
        std::cout << *computed_det << "\n";
    } else {
        std::cout << "(failed)\n";
    }

    if (expected_det && computed_det) {
        if (*expected_det != *computed_det) {
            std::cerr << "ERROR: " << name
                      << " computed determinant differs from expected!\n";
            std::cerr << "Expected: " << *expected_det << ", Got: " << *computed_det
                      << "\n";
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

#if defined(MATRIX_IMPL_CLASSIC)
    test_det_with_timer_and_print<Matrix<ScalarType>>("Matrix", n);
#else
    test_det_with_timer_and_print<Matrix<ScalarType>>("Matrix", n);
#endif

    return 0;
}
