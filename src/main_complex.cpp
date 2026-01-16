#include <iostream>
#include <complex>
#include <optional>

#include "Matrix.hpp"

#ifdef TIME_MEASURE
#include "Timer.hpp"
#endif

using ScalarType = std::complex<double>;

template <typename MatType>
void test_det_with_timer_and_print(const std::string &name, int n) {
    std::cout << "Testing " << name << " with complex numbers...\n";

    auto matrix = MatType::Generate_matrix(n, n, ScalarType{-2.0, -1.0}, ScalarType{2.0, 1.0});

    if (n <= 5) {
        std::cout << "Generated matrix:\n";
        matrix.precise_print(6);
        std::cout << "\n";
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

    if (computed_det) {
        std::cout << name << " determinant: OK.\n";
    } else {
        std::cerr << "ERROR: Determinant computation failed for complex matrix.\n";
    }

    std::cout << "\n";
}

int main() {
    int n;
    std::cin >> n;

    test_det_with_timer_and_print<Matrix<ScalarType>>("ComplexMatrix", n);

    return 0;
}
