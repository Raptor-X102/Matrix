template<typename T>
template<typename ComputeType>
Matrix<ComputeType> Matrix<T>::inverse() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square");
    }

    if constexpr (std::is_integral_v<T> && !detail::is_matrix_v<T>) {
        using RealType = std::conditional_t<sizeof(T) >= 8, double, float>;
        Matrix<RealType> real_matrix = this->template cast_to<RealType>();
        return real_matrix.inverse();
    }

    constexpr bool is_block_matrix = detail::is_matrix_v<T>;
    constexpr bool use_abs = detail::has_abs_v<ComputeType>;

    return inverse_impl<ComputeType, is_block_matrix, use_abs>();
}

template<typename T>
template<typename ComputeType, bool IsBlockMatrix>
Matrix<ComputeType> Matrix<T>::create_augmented_matrix() const {
    int n = rows_;

    if constexpr (IsBlockMatrix) {
        using InnerType = typename T::value_type;
        int block_rows = (*this)(0, 0).get_rows();
        int block_cols = (*this)(0, 0).get_cols();

        Matrix<ComputeType> augmented(n, 2 * n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented(i, j) = (*this)(i, j);
            }

            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    augmented(i, n + j) = T::Identity(block_rows, block_cols);
                } else {
                    augmented(i, n + j) = T::Zero(block_rows, block_cols);
                }
            }
        }

        return augmented;
    } else {
        Matrix<ComputeType> augmented(n, 2 * n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented(i, j) = static_cast<ComputeType>((*this)(i, j));
            }
            augmented(i, n + i) = ComputeType(1);
        }

        return augmented;
    }
}

template<typename T>
template<typename ComputeType>
Matrix<ComputeType>
Matrix<T>::extract_inverse(const Matrix<ComputeType> &augmented) const {
    int n = augmented.get_rows();
    Matrix<ComputeType> inv(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inv(i, j) = augmented(i, n + j);
        }
    }

    return inv;
}

template<typename T>
template<typename ComputeType, bool IsBlockMatrix, bool UseAbs>
Matrix<ComputeType> Matrix<T>::inverse_impl() const {
    int n = rows_;

    if (n == 0) {
        return Matrix<ComputeType>();
    }

    Matrix<ComputeType> augmented =
        create_augmented_matrix<ComputeType, IsBlockMatrix>();

    for (int k = 0; k < n; ++k) {
        int pivot_row = -1;

        if constexpr (UseAbs) {
            double max_norm = 0.0;

            for (int i = k; i < n; ++i) {
                double norm = 0.0;

                if constexpr (IsBlockMatrix) {
                    norm = compute_block_norm(augmented(i, k));
                } else {
                    using std::abs;
                    norm = abs(augmented(i, k));
                }

                if (norm > max_norm) {
                    max_norm = norm;
                    pivot_row = i;
                }
            }

            if (max_norm < Epsilon) {
                throw std::runtime_error("Matrix is singular - all zero column");
            }
        } else {
            for (int i = k; i < n; ++i) {
                bool is_non_zero = false;

                if constexpr (IsBlockMatrix) {
                    is_non_zero = (compute_block_norm(augmented(i, k)) >= Epsilon);
                } else {
                    is_non_zero = !is_element_zero(augmented(i, k));
                }

                if (is_non_zero) {
                    pivot_row = i;
                    break;
                }
            }

            if (pivot_row == -1) {
                throw std::runtime_error("Matrix is singular - all zero column");
            }
        }

        if (pivot_row != k) {
            augmented.swap_rows(k, pivot_row);
        }

        try {
            normalize_row<ComputeType, IsBlockMatrix>(augmented, k);
        } catch (const std::exception &e) {
            std::string error = "Failed to normalize row: ";
            error += e.what();
            throw std::runtime_error(error);
        }

        eliminate_other_rows<ComputeType, IsBlockMatrix, UseAbs>(augmented, k);
    }

    return extract_inverse<ComputeType>(augmented);
}

template<typename T>
template<typename ComputeType, bool IsBlockMatrix>
void Matrix<T>::normalize_row(Matrix<ComputeType> &augmented, int row) const {
    int n = augmented.get_rows();
    ComputeType pivot = augmented(row, row);

    if (is_element_zero(pivot)) {
        throw std::runtime_error("Matrix is singular - zero pivot");
    }

    if constexpr (IsBlockMatrix) {
        ComputeType pivot_inv = pivot.inverse();
        for (int j = row; j < 2 * n; ++j) {
            augmented(row, j) = pivot_inv * augmented(row, j);
        }
    } else {
        for (int j = row; j < 2 * n; ++j) {
            augmented(row, j) = augmented(row, j) / pivot;
        }
    }
}

template<typename T>
template<typename ComputeType, bool IsBlockMatrix, bool UseAbs>
void Matrix<T>::eliminate_other_rows(Matrix<ComputeType> &augmented,
                                     int pivot_row) const {
    int n = augmented.get_rows();

    for (int i = 0; i < n; ++i) {
        if (i != pivot_row) {
            ComputeType factor;

            if constexpr (IsBlockMatrix) {
                factor =
                    augmented(i, pivot_row) * augmented(pivot_row, pivot_row).inverse();
            } else {
                factor = augmented(i, pivot_row) / augmented(pivot_row, pivot_row);
            }

            if (!is_element_zero(factor)) {
                for (int j = pivot_row; j < 2 * n; ++j) {
                    augmented(i, j) = augmented(i, j) - factor * augmented(pivot_row, j);
                }
            }
        }
    }
}
