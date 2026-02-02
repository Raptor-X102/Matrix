template<typename T>
template<typename ComputeType>
Matrix<ComputeType> Matrix<T>::inverse() const {
    if (min_dim_ == 0) {
        throw std::invalid_argument("Cannot invert empty matrix");
    }

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
    // caller checked if matrix is empty
    int n = rows_;

    if constexpr (IsBlockMatrix) {
        using InnerType = typename T::value_type;

        int block_rows = (*this)(0, 0).get_rows();
        int block_cols = (*this)(0, 0).get_cols();

        Matrix<ComputeType> augmented(n, 2 * n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                try {
                    augmented(i, j) = (*this)(i, j);
                } catch (...) {
                    throw std::runtime_error("Failed to copy block at position ("
                                             + std::to_string(i) + ", "
                                             + std::to_string(j) + ")");
                }
            }

            for (int j = 0; j < n; ++j) {
                try {
                    if (i == j) {
                        augmented(i, n + j) = T::Identity(block_rows, block_cols);
                    } else {
                        augmented(i, n + j) = T::Zero(block_rows, block_cols);
                    }
                } catch (...) {
                    throw std::runtime_error(
                        "Failed to create identity/zero block at position ("
                        + std::to_string(i) + ", " + std::to_string(n + j) + ")");
                }
            }
        }

        return augmented;
    } else {
        Matrix<ComputeType> augmented(n, 2 * n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                try {
                    augmented(i, j) = static_cast<ComputeType>((*this)(i, j));
                } catch (...) {
                    throw std::runtime_error("Failed to convert element at position ("
                                             + std::to_string(i) + ", "
                                             + std::to_string(j) + ")");
                }
            }
            try {
                augmented(i, n + i) = ComputeType(1);
            } catch (...) {
                throw std::runtime_error("Failed to set identity element at position ("
                                         + std::to_string(i) + ", "
                                         + std::to_string(n + i) + ")");
            }
        }

        return augmented;
    }
}

template<typename T>
template<typename ComputeType>
Matrix<ComputeType>
Matrix<T>::extract_inverse(const Matrix<ComputeType> &augmented) const {
    // caller checked if matrix is empty
    int n = augmented.get_rows();

    Matrix<ComputeType> inv(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            try {
                inv(i, j) = augmented(i, n + j);
            } catch (...) {
                throw std::runtime_error(
                    "Failed to extract inverse element at position (" + std::to_string(i)
                    + ", " + std::to_string(j) + ")");
            }
        }
    }

    return inv;
}

template<typename T>
template<typename ComputeType, bool IsBlockMatrix, bool UseAbs>
Matrix<ComputeType> Matrix<T>::inverse_impl() const {
    // caller checked if matrix is empty
    int n = rows_;

    Matrix<ComputeType> augmented;
    try {
        augmented = create_augmented_matrix<ComputeType, IsBlockMatrix>();
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to create augmented matrix: "
                                 + std::string(e.what()));
    }

    for (int k = 0; k < n; ++k) {
        int pivot_row = -1;

        try {
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
        } catch (const std::exception &e) {
            throw std::runtime_error("Failed to find pivot in column "
                                     + std::to_string(k) + ": " + e.what());
        }

        if (pivot_row != k) {
            try {
                augmented.swap_rows(k, pivot_row);
            } catch (...) {
                throw std::runtime_error("Failed to swap rows " + std::to_string(k)
                                         + " and " + std::to_string(pivot_row));
            }
        }

        try {
            normalize_row<ComputeType, IsBlockMatrix>(augmented, k);
        } catch (const std::exception &e) {
            throw std::runtime_error("Failed to normalize row " + std::to_string(k)
                                     + ": " + e.what());
        }

        try {
            eliminate_other_rows<ComputeType, IsBlockMatrix, UseAbs>(augmented, k);
        } catch (const std::exception &e) {
            throw std::runtime_error("Failed to eliminate rows with pivot at row "
                                     + std::to_string(k) + ": " + e.what());
        }
    }

    try {
        return extract_inverse<ComputeType>(augmented);
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to extract inverse: " + std::string(e.what()));
    }
}

template<typename T>
template<typename ComputeType, bool IsBlockMatrix>
void Matrix<T>::normalize_row(Matrix<ComputeType> &augmented, int row) const {
    int n = augmented.get_rows();

    if (row >= n) {
        throw std::out_of_range("Row index out of range in normalize_row");
    }

    ComputeType pivot;
    try {
        pivot = augmented(row, row);
    } catch (...) {
        throw std::runtime_error("Failed to access pivot element at ("
                                 + std::to_string(row) + ", " + std::to_string(row)
                                 + ")");
    }

    if (is_element_zero(pivot)) {
        throw std::runtime_error("Matrix is singular - zero pivot");
    }

    try {
        if constexpr (IsBlockMatrix) {
            ComputeType pivot_inv;
            try {
                pivot_inv = pivot.inverse();
            } catch (...) {
                throw std::runtime_error("Failed to invert pivot block");
            }

            for (int j = row; j < 2 * n; ++j) {
                try {
                    augmented(row, j) = pivot_inv * augmented(row, j);
                } catch (...) {
                    throw std::runtime_error("Failed to multiply row element at column "
                                             + std::to_string(j));
                }
            }
        } else {
            for (int j = row; j < 2 * n; ++j) {
                try {
                    augmented(row, j) = augmented(row, j) / pivot;
                } catch (...) {
                    throw std::runtime_error("Failed to divide row element at column "
                                             + std::to_string(j));
                }
            }
        }
    } catch (const std::exception &e) {
        throw std::runtime_error("Error in row normalization: " + std::string(e.what()));
    }
}

template<typename T>
template<typename ComputeType, bool IsBlockMatrix, bool UseAbs>
void Matrix<T>::eliminate_other_rows(Matrix<ComputeType> &augmented,
                                     int pivot_row) const {
    int n = augmented.get_rows();

    if (pivot_row >= n) {
        throw std::out_of_range("Pivot row index out of range");
    }

    for (int i = 0; i < n; ++i) {
        if (i != pivot_row) {
            try {
                ComputeType factor;

                if constexpr (IsBlockMatrix) {
                    try {
                        factor = augmented(i, pivot_row)
                                 * augmented(pivot_row, pivot_row).inverse();
                    } catch (...) {
                        throw std::runtime_error("Failed to compute factor for row "
                                                 + std::to_string(i));
                    }
                } else {
                    try {
                        factor =
                            augmented(i, pivot_row) / augmented(pivot_row, pivot_row);
                    } catch (...) {
                        throw std::runtime_error("Failed to compute factor for row "
                                                 + std::to_string(i));
                    }
                }

                if (!is_element_zero(factor)) {
                    for (int j = pivot_row; j < 2 * n; ++j) {
                        try {
                            augmented(i, j) =
                                augmented(i, j) - factor * augmented(pivot_row, j);
                        } catch (...) {
                            throw std::runtime_error(
                                "Failed to eliminate element at column "
                                + std::to_string(j) + " in row " + std::to_string(i));
                        }
                    }
                }
            } catch (const std::exception &e) {
                throw std::runtime_error("Error eliminating row " + std::to_string(i)
                                         + " with pivot row " + std::to_string(pivot_row)
                                         + ": " + e.what());
            }
        }
    }
}
