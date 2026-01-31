template<typename T> T Matrix<T>::det() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square for determinant");
    }
    
    auto result = det(0, 0, rows_);
    if (!result) {
        throw std::runtime_error("Determinant computation failed");
    }
    return *result;
}

template<typename T> std::optional<T> Matrix<T>::try_det() const {
    if (rows_ != cols_) {
        return std::nullopt;
    }
    return det(0, 0, rows_);
}

template<typename T> std::optional<T> Matrix<T>::det(int row, int col, int size) const {
    if (row < 0 || col < 0 || size <= 0 || row + size > rows_ || col + size > cols_) {
        throw std::invalid_argument("Invalid submatrix parameters for determinant");
    }

    if constexpr (detail::is_builtin_integral_v<T>) {
        return det_integer_algorithm(row, col, size);
    } else {
        static_assert(detail::has_division_v<T>,
                      "Numeric determinant requires operator/ for type T.");
        return det_numeric_impl(row, col, size);
    }
}

template<typename T>
std::optional<T> Matrix<T>::det_integer_algorithm(int row, int col, int size) const {
    if (size == 1) {
        return (*this)(row, col);
    }

    Matrix<T> work_matrix = Submatrix(*this, row, col, size, size);
    long long previous_pivot = 1;
    int row_swaps = 0;

    for (int step = 0; step < size - 1; ++step) {
        int pivot_row = -1;
        for (int i = step; i < size; ++i) {
            if (work_matrix(i, step) != T{0}) {
                pivot_row = i;
                break;
            }
        }

        if (pivot_row == -1) {
            return T{0};
        }

        if (pivot_row != step) {
            work_matrix.swap_rows(step, pivot_row);
            ++row_swaps;
        }

        long long current_pivot = static_cast<long long>(work_matrix(step, step));

        for (int i = step + 1; i < size; ++i) {
            for (int j = step + 1; j < size; ++j) {
                long long numerator =
                    static_cast<long long>(work_matrix(i, j)) * current_pivot
                    - static_cast<long long>(work_matrix(i, step))
                          * static_cast<long long>(work_matrix(step, j));

                if (step > 0 && previous_pivot != 0) {
                    long long new_val = numerator / previous_pivot;
                    work_matrix(i, j) = static_cast<T>(new_val);
                } else {
                    work_matrix(i, j) = static_cast<T>(numerator);
                }
            }
            work_matrix(i, step) = T{0};
        }

        previous_pivot = current_pivot;
    }

    long long final_det = static_cast<long long>(work_matrix(size - 1, size - 1));

    if (row_swaps % 2 != 0) {
        final_det = -final_det;
    }

    if (final_det > std::numeric_limits<T>::max()
        || final_det < std::numeric_limits<T>::min()) {
        throw std::overflow_error("Determinant overflow");
    }

    return static_cast<T>(final_det);
}

template<typename T>
template<typename U>
std::optional<T> Matrix<T>::det_numeric_impl(int row, int col, int size) const {
    int block_rows = 1, block_cols = 1;
    if constexpr (detail::is_matrix_v<T>) {
        if (size > 0) {
            block_rows = (*this)(row, col).get_rows();
            block_cols = (*this)(row, col).get_cols();
        }
    }

    if (size == 1) {
        return (*this)(row, col);
    }

    Matrix<T> matrix_cpy = Submatrix(*this, row, col, size, size);
    T determinant = identity_element<T>(block_rows, block_cols);
    int sign = 1;

    for (int j = 0; j < size; ++j) {
        std::optional<int> max_index_opt =
            matrix_cpy.template find_pivot_in_subcol<T>(j, j);
        if (!max_index_opt) {
            return zero_element<T>(block_rows, block_cols);
        }

        int max_index = *max_index_opt;

        if (max_index != j) {
            matrix_cpy.swap_rows(max_index, j);
            sign = -sign;
        }

        T pivot = matrix_cpy(j, j);

        if (is_element_zero(pivot)) {
            return zero_element<T>(block_rows, block_cols);
        }

        determinant = determinant * pivot;

        for (int i = j + 1; i < size; ++i) {
            T scalar = matrix_cpy(i, j) / pivot;

            for (int k = j + 1; k < size; ++k) {
                matrix_cpy(i, k) = matrix_cpy(i, k) - scalar * matrix_cpy(j, k);
            }
        }
    }

    if (sign == -1) {
        determinant = -determinant;
    }

    return determinant;
}
