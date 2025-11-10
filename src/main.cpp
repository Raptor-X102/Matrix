#include "Matrix.hpp"
//#include "Matrix_1d_array.hpp"

int main() {

    int n;
    std::cin >> n;
    Matrix<double> matrix = Matrix<double>::Generate_matrix(n, n, 0.1, 2);
    matrix.precise_print();
    std::optional<double> det = matrix.get_determinant();
    std::cout << *det << "\n";

    #ifdef TIME_MEASURE
    {
        Timer timer;
    #endif

        det = matrix.det();

    #ifdef TIME_MEASURE
    }
    #endif

    if (!det) {

        DEBUG_PRINTF("ERROR: det() returned nullopt\n");
        return 1;
    }

    std::cout << *det;
    return 0;
}
