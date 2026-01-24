template<typename T> Matrix<T> Matrix<T>::transpose() const {
    if (rows_ == 0 || cols_ == 0) {
        return Matrix<T>(cols_, rows_);
    }

    Matrix<T> result(cols_, rows_);
    transpose_impl(result);

    result.determinant_ = determinant_;
    return result;
}

template<typename T> Matrix<T> &Matrix<T>::transpose_in_place() {
    if (rows_ != cols_) {
        throw std::invalid_argument("Cannot transpose non-square matrix in place");
    }

    for (int i = 0; i < rows_; ++i) {
        for (int j = i + 1; j < cols_; ++j) {
            std::swap(matrix_[i][j], matrix_[j][i]);
        }
    }

    determinant_.reset();
    return *this;
}

template<typename T> void Matrix<T>::transpose_impl(Matrix<T> &result) const {
    TransposeAlgorithm algorithm = select_transpose_algorithm();

    switch (algorithm) {
#ifdef __AVX__
    case TransposeAlgorithm::SIMD_BLOCKED:
        if constexpr (std::is_same_v<T, float>) {
            transpose_simd_blocked<float, 8>(result);
        } else if constexpr (std::is_same_v<T, double>) {
            transpose_simd_blocked<double, 4>(result);
        } else {
            transpose_blocked(result);
        }
        break;
#endif
    case TransposeAlgorithm::BLOCKED:
        transpose_blocked(result);
        break;
    case TransposeAlgorithm::SIMPLE:
    default:
        transpose_simple(result);
        break;
    }
}

template<typename T>
typename Matrix<T>::TransposeAlgorithm Matrix<T>::select_transpose_algorithm() const {
    if (is_small_matrix()) {
        return TransposeAlgorithm::SIMPLE;
    }

#ifdef __AVX__
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        if (rows_ >= 64 && cols_ >= 64) {
            return TransposeAlgorithm::SIMD_BLOCKED;
        }
    }
#endif

    if (should_use_blocking()) {
        return TransposeAlgorithm::BLOCKED;
    }

    return TransposeAlgorithm::SIMPLE;
}

template<typename T> bool Matrix<T>::is_small_matrix() const {
    return (rows_ * cols_) < 64;
}

template<typename T> bool Matrix<T>::should_use_blocking() const {
    if (rows_ < 32 && cols_ < 32) {
        return false;
    }

    if constexpr (!std::is_trivially_copyable_v<T>) {
        if constexpr (!std::is_arithmetic_v<T>) {
            return false;
        }
    }

    if constexpr (detail::is_matrix_v<T>) {
        return false;
    }

    const int optimal_block = compute_optimal_block_size();
    return optimal_block > 2;
}

template<typename T> int Matrix<T>::compute_optimal_block_size() const {
    constexpr int CACHE_LINE_SIZE = 64;

    if constexpr (sizeof(T) == 0) {
        return 8;
    }

    constexpr size_t type_size = sizeof(T);

    if (type_size >= CACHE_LINE_SIZE) {
        return 1;
    }

    int block = static_cast<int>(CACHE_LINE_SIZE / type_size);

    block = std::max(block, 4);
    block = std::min(block, 64);
    block = std::min(block, rows_);
    block = std::min(block, cols_);

    return block;
}

template<typename T> void Matrix<T>::transpose_simple(Matrix<T> &result) const {
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result.matrix_[j][i] = matrix_[i][j];
        }
    }
}

template<typename T> void Matrix<T>::transpose_blocked(Matrix<T> &result) const {
    const int block_size = compute_optimal_block_size();

    if (block_size <= 2) {
        transpose_simple(result);
        return;
    }

    transpose_blocked_impl(result, block_size);
}

template<typename T>
void Matrix<T>::transpose_blocked_impl(Matrix<T> &result, int block_size) const {
    for (int outer_i = 0; outer_i < rows_; outer_i += block_size) {
        for (int outer_j = 0; outer_j < cols_; outer_j += block_size) {
            const int i_end = std::min(outer_i + block_size, rows_);
            const int j_end = std::min(outer_j + block_size, cols_);

            for (int i = outer_i; i < i_end; ++i) {
                for (int j = outer_j; j < j_end; ++j) {
                    result.matrix_[j][i] = matrix_[i][j];
                }
            }
        }
    }
}

template<> void Matrix<int>::transpose_simple(Matrix<int> &result) const {
    if (rows_ == 1) {
        for (int j = 0; j < cols_; ++j) {
            result.matrix_[j][0] = matrix_[0][j];
        }
    } else if (cols_ == 1) {
        for (int i = 0; i < rows_; ++i) {
            result.matrix_[0][i] = matrix_[i][0];
        }
    } else if (rows_ == 2 && cols_ == 2) {
        result.matrix_[0][0] = matrix_[0][0];
        result.matrix_[0][1] = matrix_[1][0];
        result.matrix_[1][0] = matrix_[0][1];
        result.matrix_[1][1] = matrix_[1][1];
    } else if (rows_ == 3 && cols_ == 3) {
        result.matrix_[0][0] = matrix_[0][0];
        result.matrix_[0][1] = matrix_[1][0];
        result.matrix_[0][2] = matrix_[2][0];
        result.matrix_[1][0] = matrix_[0][1];
        result.matrix_[1][1] = matrix_[1][1];
        result.matrix_[1][2] = matrix_[2][1];
        result.matrix_[2][0] = matrix_[0][2];
        result.matrix_[2][1] = matrix_[1][2];
        result.matrix_[2][2] = matrix_[2][2];
    } else if (rows_ == 4 && cols_ == 4) {
        result.matrix_[0][0] = matrix_[0][0];
        result.matrix_[0][1] = matrix_[1][0];
        result.matrix_[0][2] = matrix_[2][0];
        result.matrix_[0][3] = matrix_[3][0];
        result.matrix_[1][0] = matrix_[0][1];
        result.matrix_[1][1] = matrix_[1][1];
        result.matrix_[1][2] = matrix_[2][1];
        result.matrix_[1][3] = matrix_[3][1];
        result.matrix_[2][0] = matrix_[0][2];
        result.matrix_[2][1] = matrix_[1][2];
        result.matrix_[2][2] = matrix_[2][2];
        result.matrix_[2][3] = matrix_[3][2];
        result.matrix_[3][0] = matrix_[0][3];
        result.matrix_[3][1] = matrix_[1][3];
        result.matrix_[3][2] = matrix_[2][3];
        result.matrix_[3][3] = matrix_[3][3];
    } else {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result.matrix_[j][i] = matrix_[i][j];
            }
        }
    }
}

template<typename T>
template<typename U, int SIMD_WIDTH>
void Matrix<T>::transpose_simd_blocked(Matrix<U> &result) const {
    constexpr int BLOCK_SIZE = 64;
    constexpr int SIMD_LANES = SIMD_WIDTH;

    for (int outer_i = 0; outer_i < rows_; outer_i += BLOCK_SIZE) {
        for (int outer_j = 0; outer_j < cols_; outer_j += BLOCK_SIZE) {
            const int i_end = std::min(outer_i + BLOCK_SIZE, rows_);
            const int j_end = std::min(outer_j + BLOCK_SIZE, cols_);

            for (int i = outer_i; i < i_end; i += SIMD_LANES) {
                for (int j = outer_j; j < j_end; j += SIMD_LANES) {
                    const int i_remaining = std::min(i_end - i, SIMD_LANES);
                    const int j_remaining = std::min(j_end - j, SIMD_LANES);

                    if constexpr (std::is_same_v<U, float>) {
                        transpose_block_simd_float(result,
                                                   i,
                                                   j,
                                                   i_remaining,
                                                   j_remaining);
                    } else if constexpr (std::is_same_v<U, double>) {
                        transpose_block_simd_double(result,
                                                    i,
                                                    j,
                                                    i_remaining,
                                                    j_remaining);
                    }
                }
            }
        }
    }
}

template<typename T>
void Matrix<T>::transpose_block_simd_float(Matrix<float> &result,
                                           int start_i,
                                           int start_j,
                                           int rows_in_block,
                                           int cols_in_block) const {
    __m256 rows[8];
    for (int i = 0; i < rows_in_block; ++i) {
        float temp[8] = {0};
        for (int j = 0; j < cols_in_block; ++j) {
            temp[j] = matrix_[start_i + i][start_j + j];
        }
        rows[i] = _mm256_loadu_ps(temp);
    }

    if (rows_in_block >= 8 && cols_in_block >= 8) {
        __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

        tmp0 = _mm256_unpacklo_ps(rows[0], rows[1]);
        tmp1 = _mm256_unpackhi_ps(rows[0], rows[1]);
        tmp2 = _mm256_unpacklo_ps(rows[2], rows[3]);
        tmp3 = _mm256_unpackhi_ps(rows[2], rows[3]);
        tmp4 = _mm256_unpacklo_ps(rows[4], rows[5]);
        tmp5 = _mm256_unpackhi_ps(rows[4], rows[5]);
        tmp6 = _mm256_unpacklo_ps(rows[6], rows[7]);
        tmp7 = _mm256_unpackhi_ps(rows[6], rows[7]);

        __m256 col0 = _mm256_shuffle_ps(tmp0, tmp2, 0x44);
        __m256 col1 = _mm256_shuffle_ps(tmp0, tmp2, 0xEE);
        __m256 col2 = _mm256_shuffle_ps(tmp1, tmp3, 0x44);
        __m256 col3 = _mm256_shuffle_ps(tmp1, tmp3, 0xEE);
        __m256 col4 = _mm256_shuffle_ps(tmp4, tmp6, 0x44);
        __m256 col5 = _mm256_shuffle_ps(tmp4, tmp6, 0xEE);
        __m256 col6 = _mm256_shuffle_ps(tmp5, tmp7, 0x44);
        __m256 col7 = _mm256_shuffle_ps(tmp5, tmp7, 0xEE);

        float out_cols[8][8];
        _mm256_storeu_ps(out_cols[0], col0);
        _mm256_storeu_ps(out_cols[1], col1);
        _mm256_storeu_ps(out_cols[2], col2);
        _mm256_storeu_ps(out_cols[3], col3);
        _mm256_storeu_ps(out_cols[4], col4);
        _mm256_storeu_ps(out_cols[5], col5);
        _mm256_storeu_ps(out_cols[6], col6);
        _mm256_storeu_ps(out_cols[7], col7);

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                result.matrix_[start_j + j][start_i + i] = out_cols[j][i];
            }
        }
    } else {
        for (int i = 0; i < rows_in_block; ++i) {
            for (int j = 0; j < cols_in_block; ++j) {
                result.matrix_[start_j + j][start_i + i] =
                    reinterpret_cast<float *>(&rows[i])[j];
            }
        }
    }
}

template<typename T>
void Matrix<T>::transpose_block_simd_double(Matrix<double> &result,
                                            int start_i,
                                            int start_j,
                                            int rows_in_block,
                                            int cols_in_block) const {
    __m256d rows[4];
    for (int i = 0; i < rows_in_block; ++i) {
        double temp[4] = {0};
        for (int j = 0; j < cols_in_block; ++j) {
            temp[j] = matrix_[start_i + i][start_j + j];
        }
        rows[i] = _mm256_loadu_pd(temp);
    }

    if (rows_in_block >= 4 && cols_in_block >= 4) {
        __m256d tmp0 = _mm256_unpacklo_pd(rows[0], rows[1]);
        __m256d tmp1 = _mm256_unpackhi_pd(rows[0], rows[1]);
        __m256d tmp2 = _mm256_unpacklo_pd(rows[2], rows[3]);
        __m256d tmp3 = _mm256_unpackhi_pd(rows[2], rows[3]);

        __m256d col0 = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
        __m256d col1 = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
        __m256d col2 = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);
        __m256d col3 = _mm256_permute2f128_pd(tmp1, tmp3, 0x31);

        double out_cols[4][4];
        _mm256_storeu_pd(out_cols[0], col0);
        _mm256_storeu_pd(out_cols[1], col1);
        _mm256_storeu_pd(out_cols[2], col2);
        _mm256_storeu_pd(out_cols[3], col3);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result.matrix_[start_j + j][start_i + i] = out_cols[j][i];
            }
        }
    } else {
        for (int i = 0; i < rows_in_block; ++i) {
            for (int j = 0; j < cols_in_block; ++j) {
                result.matrix_[start_j + j][start_i + i] =
                    reinterpret_cast<double *>(&rows[i])[j];
            }
        }
    }
}

template<typename T> Matrix<T> Matrix<T>::transpose_deep() const {
    return transpose_deep_impl();
}

template<typename T> Matrix<T> Matrix<T>::transpose_deep_impl() const {
    if constexpr (detail::is_matrix_v<T>) {
        Matrix<T> result(cols_, rows_);

        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result.matrix_[j][i] = matrix_[i][j].transpose_deep();
            }
        }

        if (determinant_) {
            result.determinant_ = determinant_->transpose_deep();
        }

        return result;
    } else {
        return transpose();
    }
}

template<typename T> Matrix<T> &Matrix<T>::transpose_deep_in_place() {
    if (rows_ != cols_) {
        throw std::invalid_argument("Cannot transpose non-square matrix in place");
    }

    if constexpr (detail::is_matrix_v<T>) {
        for (int i = 0; i < rows_; ++i) {
            for (int j = i + 1; j < cols_; ++j) {
                std::swap(matrix_[i][j], matrix_[j][i]);
            }
        }

        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                matrix_[i][j].transpose_deep_in_place();
            }
        }

        if (determinant_) {
            determinant_->transpose_deep_in_place();
        }
    } else {
        return transpose_in_place();
    }

    return *this;
}
