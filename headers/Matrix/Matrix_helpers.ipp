template<typename T> void Matrix<T>::alloc_matrix_() {
    if (rows_ <= 0 || cols_ <= 0) {
        matrix_ = nullptr;
        return;
    }

    try {
        matrix_ = std::make_unique<std::unique_ptr<T[]>[]>(rows_);
        for (int i = 0; i < rows_; ++i) {
            matrix_[i] = std::make_unique<T[]>(cols_);
        }
    } catch (const std::bad_alloc &) {
        matrix_.reset();
        throw;
    }
}

template<typename T> void Matrix<T>::init_zero_() {
    for (int i = 0; i < rows_; i++)
        for (int j = 0; j < cols_; j++)
            matrix_[i][j] = T{};
}

template<typename T> std::optional<T> &Matrix<T>::get_determinant_() {
    return determinant_;
}

template<typename T> std::optional<T> Matrix<T>::get_determinant() const {
    return determinant_;
}

template<typename T> int Matrix<T>::get_rows() const {
    return rows_;
}
template<typename T> int Matrix<T>::get_cols() const {
    return cols_;
}
template<typename T> int Matrix<T>::get_min_dim() const {
    return min_dim_;
}

template<typename T> template<typename U> Matrix<U> Matrix<T>::cast_to() const {
    using NonRefU = std::remove_reference_t<U>;
    static_assert(!std::is_reference_v<NonRefU>, "Cannot cast to reference type");

    if constexpr (detail::is_matrix_v<T> && detail::is_matrix_v<NonRefU>) {
        Matrix<NonRefU> result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result(i, j) =
                    (*this)(i, j).template cast_to<typename NonRefU::value_type>();
            }
        }
        return result;
    } else if constexpr (detail::is_matrix_v<T> && !detail::is_matrix_v<NonRefU>) {
        throw std::runtime_error("Cannot cast block matrix to scalar matrix type");
    } else {
        Matrix<NonRefU> result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result(i, j) = static_cast<NonRefU>((*this)(i, j));
            }
        }
        return result;
    }
}

template<typename T>
template<typename U>
bool Matrix<T>::is_equal(const U &a, const U &b) {
    if constexpr (detail::is_matrix_v<U>) {
        if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) {
            return false;
        }
        for (int i = 0; i < a.get_rows(); ++i) {
            for (int j = 0; j < a.get_cols(); ++j) {
                if (!is_equal(a(i, j), b(i, j))) {
                    return false;
                }
            }
        }
        return true;
    } else {
        if constexpr (std::is_floating_point_v<U>) {
            constexpr double epsilon = 1e-12;
            if constexpr (std::is_same_v<U, float>) {
                constexpr float epsilon_f = 1e-6f;
                return std::abs(a - b) < epsilon_f;
            } else {
                return std::abs(a - b) < epsilon;
            }
        } else if constexpr (detail::is_complex_v<U>) {
            using std::abs;
            using RealType = typename U::value_type;
            constexpr RealType epsilon = static_cast<RealType>(1e-12);
            return abs(a - b) < epsilon;
        } else {
            return a == b;
        }
    }
}

template<typename T> template<typename U> bool Matrix<T>::is_zero(const U &value) {
    if constexpr (detail::is_matrix_v<U>) {
        auto norm_val = value.frobenius_norm();
        return is_zero(norm_val);
    } else if constexpr (detail::is_complex_v<U>) {
        using std::abs;
        constexpr double epsilon = 1e-12;
        if constexpr (std::is_same_v<typename U::value_type, float>) {
            constexpr float epsilon_f = 1e-6f;
            return abs(value) < epsilon_f;
        } else {
            return abs(value) < epsilon;
        }
    } else if constexpr (std::is_floating_point_v<U>) {
        constexpr double epsilon = 1e-12;
        if constexpr (std::is_same_v<U, float>) {
            constexpr float epsilon_f = 1e-6f;
            return std::abs(value) < epsilon_f;
        } else {
            return std::abs(value) < epsilon;
        }
    } else {
        return value == U{0};
    }
}

template<typename T> bool Matrix<T>::is_zero(int i, int j) const {
    return is_zero((*this)(i, j));
}

template<typename T>
template<typename U>
std::optional<int> Matrix<T>::find_pivot_in_subcol(int row, int col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        DEBUG_PRINTF("ERROR: index out of range\n");
        return std::nullopt;
    }

    if (rows_ == 0)
        return std::nullopt;

    int max_val_index = row;

    if constexpr (detail::has_abs_v<U> && !detail::is_matrix_v<U>) {
        using std::abs;
        auto max_abs = abs((*this)(row, col));

        for (int i = row + 1; i < rows_; ++i) {
            auto current_abs = abs((*this)(i, col));
            if (current_abs > max_abs) {
                max_val_index = i;
                max_abs = current_abs;
            }
        }
        return max_val_index;
    } else if constexpr (detail::is_matrix_v<U>) {
        double max_norm = compute_block_norm((*this)(row, col));

        for (int i = row + 1; i < rows_; ++i) {
            double current_norm = compute_block_norm((*this)(i, col));
            if (current_norm > max_norm) {
                max_val_index = i;
                max_norm = current_norm;
            }
        }
        return max_val_index;
    } else {
        for (int i = row; i < rows_; ++i) {
            if (!is_zero((*this)(i, col))) {
                return i;
            }
        }
        return std::nullopt;
    }
}

template<typename T> void Matrix<T>::swap_data(Matrix<T> &other) noexcept {
    using std::swap;
    swap(rows_, other.rows_);
    swap(cols_, other.cols_);
    swap(min_dim_, other.min_dim_);
    matrix_.swap(other.matrix_);
}

template<typename T> void swap(Matrix<T> &first, Matrix<T> &second) noexcept {
    first.swap_data(second);
}

template<typename T> void Matrix<T>::swap_rows(int i, int j) {
    if (i < 0 || i >= rows_ || j < 0 || j >= rows_) {
        throw std::out_of_range("Row index out of range in swap_rows");
    }

    if (i != j) {
        std::swap(matrix_[i], matrix_[j]);

        if (determinant_)
            determinant_ = -*determinant_;
    }
}

template<typename T> void Matrix<T>::multiply_row(int target_row, T scalar) {
    if (target_row < 0 || target_row >= rows_) {
        throw std::out_of_range("Row index out of range in multiply_row");
    }

    // there is no scalar null check here intentionally
    // user must keep in mind that
    // it would't be an equivalent transformation
    for (int j = 0; j < cols_; ++j)
        matrix_[target_row][j] = matrix_[target_row][j] * scalar;

    if (determinant_)
        determinant_ = scalar * *determinant_;
}

template<typename T>
void Matrix<T>::add_row_scaled(int target_row, int source_row, T scalar) {
    if (target_row < 0 || target_row >= rows_ || source_row < 0 || source_row >= rows_) {
        throw std::out_of_range("Row index out of range in add_row_scaled");
    }

    for (int j = 0; j < cols_; ++j)
        matrix_[target_row][j] =
            matrix_[target_row][j] + matrix_[source_row][j] * scalar;
}

template<typename T> void Matrix<T>::print() const {
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++)
            std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                      << std::defaultfloat << matrix_[i][j] << ' ';

        std::cout << "\n\n";
    }
}

template<typename T> void Matrix<T>::print(int max_size) const {
    if (rows_ <= max_size && cols_ <= max_size) {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++)
                std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                          << std::defaultfloat << matrix_[i][j] << ' ';
            std::cout << "\n\n";
        }
    } else {
        std::cout << "Skipped printing (dimensions " << rows_ << "x" << cols_
                  << " exceed " << max_size << "x" << max_size << ").\n";
    }
}

template<typename T> void Matrix<T>::precise_print(int precision) const {
    int field_width = precision + 8;
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if constexpr (detail::is_matrix_v<T>) {
                if (rows_ <= 5 && cols_ <= 5 && (*this)(i, j).get_rows() <= 3
                    && (*this)(i, j).get_cols() <= 3) {
                    std::cout << "[" << (*this)(i, j).get_rows() << "x"
                              << (*this)(i, j).get_cols() << "] ";
                } else {
                    std::cout << "[SubMatrix] ";
                }
            } else {
                std::cout << std::setw(field_width) << std::scientific
                          << std::setprecision(precision) << matrix_[i][j] << " ";
            }
        }
        std::cout << "\n";
    }
}

template<typename T> void Matrix<T>::detailed_print() const {
    if constexpr (detail::is_matrix_v<T>) {
        std::cout << "Block Matrix " << rows_ << "x" << cols_ << " of "
                  << (*this)(0, 0).get_rows() << "x" << (*this)(0, 0).get_cols()
                  << " blocks:\n";

        std::cout << std::fixed << std::setprecision(2);

        for (int i = 0; i < rows_; ++i) {
            for (int inner_row = 0; inner_row < (*this)(0, 0).get_rows(); ++inner_row) {
                std::cout << "  ";
                for (int j = 0; j < cols_; ++j) {
                    const auto &block = (*this)(i, j);

                    std::cout << "[";
                    for (int inner_col = 0; inner_col < block.get_cols(); ++inner_col) {
                        std::cout << std::setw(6) << block(inner_row, inner_col);
                        if (inner_col < block.get_cols() - 1)
                            std::cout << " ";
                    }
                    std::cout << "]";

                    if (j < cols_ - 1)
                        std::cout << "  ";
                }
                std::cout << "\n";
            }
            if (i < rows_ - 1) {
                std::cout << "\n";
            }
        }
        std::cout << std::defaultfloat;
    } else {
        print(10);
    }
}

template<typename T>
template<typename U>
bool Matrix<T>::is_element_zero(const U &elem) {
    using std::abs;

    if constexpr (detail::is_matrix_v<U>) {
        return compute_block_norm(elem) < Epsilon;
    } else {
        try {
            return abs(elem) < Epsilon;
        } catch (...) {
            try {
                return elem == U{};
            } catch (...) {
                return false;
            }
        }
    }
}

template<typename T>
template<typename U>
U Matrix<T>::identity_element(int rows, int cols) {
    if constexpr (detail::is_matrix_v<U>) {
        return U::Identity(rows, cols);
    } else {
        return U{1};
    }
}

template<typename T> template<typename U> U Matrix<T>::zero_element(int rows, int cols) {
    if constexpr (detail::is_matrix_v<U>) {
        return U::Zero(rows, cols);
    } else {
        return U{0};
    }
}

template<typename T>
template<typename ExampleType, typename ValueType>
auto Matrix<T>::create_scalar(const ExampleType &example, ValueType value) {
    if constexpr (detail::is_matrix_v<ExampleType>) {
        return value;
    } else {
        using CommonType =
            typename detail::matrix_common_type<ExampleType, ValueType>::type;
        return static_cast<CommonType>(value);
    }
}
