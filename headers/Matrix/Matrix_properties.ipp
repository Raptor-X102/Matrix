template<typename T> T Matrix<T>::frobenius_norm_squared() const {
    if (rows_ == 0 || cols_ == 0) {
        return T{};
    }

    T sum = matrix_[0][0] - matrix_[0][0];

#if defined(__AVX__) || defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
        __m256 sum_vec = _mm256_setzero_ps();
        int total_elements = rows_ * cols_;
        std::vector<float> flat_data(total_elements);

        int idx = 0;
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                flat_data[idx++] = matrix_[i][j];
            }
        }
        const float *data = flat_data.data();

        int i = 0;
        for (; i + 7 < total_elements; i += 8) {
            __m256 v = _mm256_loadu_ps(&data[i]);
            sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
        }

        alignas(32) float temp[8];
        _mm256_store_ps(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6]
              + temp[7];

        for (; i < total_elements; ++i) {
            sum += data[i] * data[i];
        }
        return sum;
    } else if constexpr (std::is_same_v<T, double>) {
        __m256d sum_vec = _mm256_setzero_pd();
        int total_elements = rows_ * cols_;
        std::vector<double> flat_data(total_elements);

        int idx = 0;
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                flat_data[idx++] = matrix_[i][j];
            }
        }
        const double *data = flat_data.data();

        int i = 0;
        for (; i + 3 < total_elements; i += 4) {
            __m256d v = _mm256_loadu_pd(&data[i]);
            sum_vec = _mm256_fmadd_pd(v, v, sum_vec);
        }

        alignas(32) double temp[4];
        _mm256_store_pd(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];

        for (; i < total_elements; ++i) {
            sum += data[i] * data[i];
        }
        return sum;
    }
#endif

    if constexpr (std::is_arithmetic_v<T>) {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                T val = matrix_[i][j];
                sum += val * val;
            }
        }
    } else if constexpr (detail::is_complex_v<T>) {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                T val = matrix_[i][j];
                sum += std::norm(val);
            }
        }
    } else if constexpr (detail::is_matrix_v<T>) {
        using ElemType = typename T::value_type;
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                T val = matrix_[i][j];
                sum = sum + val.frobenius_norm_squared();
            }
        }
    } else {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                T val = matrix_[i][j];
                sum += val * val;
            }
        }
    }

    return sum;
}

template<typename T>
typename Matrix<T>::norm_return_type Matrix<T>::frobenius_norm() const {
    using ReturnType = typename Matrix<T>::norm_return_type;

    if constexpr (detail::is_matrix_v<T>) {
        using ElemType = typename T::value_type;
        using ElemNormType = typename Matrix<ElemType>::norm_return_type;

        ElemNormType sum = ElemNormType(0);

        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                auto block_norm = matrix_[i][j].frobenius_norm();
                sum = sum + block_norm * block_norm;
            }
        }

        if constexpr (std::is_integral_v<ElemNormType>) {
            return static_cast<ReturnType>(std::sqrt(static_cast<double>(sum)));
        } else {
            using std::sqrt;
            return static_cast<ReturnType>(sqrt(sum));
        }
    } else {
        T sum_squared = frobenius_norm_squared();

        if constexpr (detail::is_builtin_integral_v<T>) {
            return static_cast<ReturnType>(std::sqrt(static_cast<double>(sum_squared)));
        } else if constexpr (detail::is_complex_v<T>) {
            using std::sqrt;
            return sqrt(sum_squared);
        } else {
            using std::sqrt;
            return sqrt(sum_squared);
        }
    }
}

template<typename T> T Matrix<T>::trace() const {
    if (min_dim_ == 0) {
        return matrix_[0][0] - matrix_[0][0];
    }

    T result = matrix_[0][0] - matrix_[0][0];

    for (int i = 0; i < min_dim_; ++i) {
        result = result + matrix_[i][i];
    }

    return result;
}

template<typename T>
bool Matrix<T>::is_symmetric() const {
    if (rows_ != cols_) return false;

    for (int i = 0; i < rows_; ++i) {
        for (int j = i + 1; j < cols_; ++j) {
            if (!is_zero(matrix_[i][j] - matrix_[j][i])) {
                return false;
            }
        }
    }
    return true;
}

template<typename T>
template<typename ComputeType>
typename Matrix<ComputeType>::norm_return_type Matrix<T>::off_diagonal_norm(const Matrix<ComputeType>& H) const {
    int n = H.get_rows();
    using NormType = typename Matrix<ComputeType>::norm_return_type;
    
    NormType sum{};
    if constexpr (detail::is_matrix_v<NormType>) {
        int block_rows = 1;
        int block_cols = 1;
        if (n > 0 && H(0,0).get_rows() > 0) {
            block_rows = H(0,0).get_rows();
            block_cols = H(0,0).get_cols();
        }
        sum = Matrix<typename NormType::value_type>::Zero(block_rows, block_cols);
    } else if constexpr (detail::is_complex_v<NormType>) {
        sum = NormType{0.0, 0.0};
    } else {
        sum = NormType{0.0};
    }
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                auto elem = H(i, j);
                if constexpr (detail::is_matrix_v<decltype(elem)>) {
                    auto norm_elem = elem.frobenius_norm_squared();
                    sum = sum + norm_elem;
                } else if constexpr (detail::is_complex_v<decltype(elem)>) {
                    using std::abs;
                    sum = sum + abs(elem) * abs(elem);
                } else {
                    sum = sum + elem * elem;
                }
            }
        }
    }
    
    using std::sqrt;
    return sqrt(sum);
}

template<typename T>
template<typename ComputeType>
bool Matrix<T>::is_norm_zero(const ComputeType& norm_value) const {
    if constexpr (detail::is_matrix_v<ComputeType>) {
        auto frob_norm = norm_value.frobenius_norm();
        return is_norm_zero(frob_norm);
    } else if constexpr (detail::is_complex_v<ComputeType>) {
        using std::abs;
        constexpr double epsilon = 1e-12;
        return abs(norm_value) < epsilon;
    } else if constexpr (std::is_floating_point_v<ComputeType>) {
        constexpr double epsilon = 1e-12;
        if constexpr (std::is_same_v<ComputeType, float>) {
            constexpr float epsilon_f = 1e-6f;
            return std::abs(norm_value) < epsilon_f;
        } else {
            return std::abs(norm_value) < epsilon;
        }
    } else {
        return norm_value == ComputeType{0};
    }
}
