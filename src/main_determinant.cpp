#include "Matrix.hpp"
#include <iostream>

int main() {
    int n;
    std::cin >> n;

    Matrix<double> matrix(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> matrix(i, j);
        }
    }

    std::cout << matrix.det() << std::endl;

    return 0;
}
