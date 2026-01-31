template<typename T>
template<typename U>
Matrix<T> &Matrix<T>::operator+=(const Matrix<U> &other) {
    *this = *this + other;
    return *this;
}

template<typename T>
template<typename U>
Matrix<T> &Matrix<T>::operator-=(const Matrix<U> &other) {
    *this = *this - other;
    return *this;
}

template<typename T> Matrix<T> &Matrix<T>::operator*=(const Matrix &other) {
    *this = *this * other;
    return *this;
}

template<typename T>
template<typename U>
Matrix<T> &Matrix<T>::operator*=(const U &scalar) {
    *this = *this * scalar;
    return *this;
}

template<typename T>
template<typename U>
Matrix<T> &Matrix<T>::operator/=(const U &scalar) {
    *this = *this / scalar;
    return *this;
}

template<typename T> bool Matrix<T>::operator==(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        return false;
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            if (!is_equal((*this)(i, j), other(i, j)))
                return false;
        }
    }
    return true;
}

template<typename T> bool Matrix<T>::operator!=(const Matrix &other) const {
    return !(*this == other);
}

template<typename T> template<typename U> Matrix<T>::operator Matrix<U>() const {
    using NonRefU = std::remove_reference_t<U>;
    static_assert(!std::is_reference_v<NonRefU>, "Cannot cast to reference type");

    Matrix<NonRefU> result(rows_, cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result(i, j) = static_cast<NonRefU>((*this)(i, j));
        }
    }
    return result;
}

template<typename T> T &Matrix<T>::operator()(int i, int j) {
    // no index check, due to speed loss
    return matrix_[i][j];
}

template<typename T> const T &Matrix<T>::operator()(int i, int j) const {
    // no index check, due to speed loss
    return matrix_[i][j];
}

template<typename T> Matrix<T> Matrix<T>::operator-() const {
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result(i, j) = -(*this)(i, j);
        }
    }
    return result;
}

template<typename T, typename U>
auto operator+(const Matrix<T> &lhs, const Matrix<U> &rhs) {
    if (lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    using CommonType = detail::matrix_common_type_t<T, U>;
    Matrix<CommonType> result(lhs.get_rows(), lhs.get_cols());

    if constexpr (detail::is_matrix_v<CommonType>) {
        if (lhs.get_rows() > 0 && lhs.get_cols() > 0) {
            int inner_rows = 1, inner_cols = 1;
            if constexpr (detail::is_matrix_v<T>) {
                inner_rows = lhs(0, 0).get_rows();
                inner_cols = lhs(0, 0).get_cols();
            } else if constexpr (detail::is_matrix_v<U>) {
                inner_rows = rhs(0, 0).get_rows();
                inner_cols = rhs(0, 0).get_cols();
            }
            for (int i = 0; i < result.get_rows(); ++i) {
                for (int j = 0; j < result.get_cols(); ++j) {
                    result(i, j) = CommonType::Zero(inner_rows, inner_cols);
                }
            }
        }
    }

#ifdef __AVX__
    if constexpr (std::is_same_v<T, U> && detail::is_avx_double<T>) {
        for (int i = 0; i < lhs.get_rows(); ++i) {
            int j = 0;
            for (; j + 4 <= lhs.get_cols(); j += 4) {
                double lhs_temp[4] = {lhs(i, j),
                                      lhs(i, j + 1),
                                      lhs(i, j + 2),
                                      lhs(i, j + 3)};
                double rhs_temp[4] = {rhs(i, j),
                                      rhs(i, j + 1),
                                      rhs(i, j + 2),
                                      rhs(i, j + 3)};

                __m256d lhs_vec = _mm256_loadu_pd(lhs_temp);
                __m256d rhs_vec = _mm256_loadu_pd(rhs_temp);
                __m256d res_vec = _mm256_add_pd(lhs_vec, rhs_vec);
                _mm256_storeu_pd(lhs_temp, res_vec);

                result(i, j) = lhs_temp[0];
                result(i, j + 1) = lhs_temp[1];
                result(i, j + 2) = lhs_temp[2];
                result(i, j + 3) = lhs_temp[3];
            }
            for (; j < lhs.get_cols(); ++j) {
                result(i, j) = lhs(i, j) + rhs(i, j);
            }
        }
    } else if constexpr (std::is_same_v<T, U> && detail::is_avx_float<T>) {
        for (int i = 0; i < lhs.get_rows(); ++i) {
            int j = 0;
            for (; j + 8 <= lhs.get_cols(); j += 8) {
                float lhs_temp[8] = {lhs(i, j),
                                     lhs(i, j + 1),
                                     lhs(i, j + 2),
                                     lhs(i, j + 3),
                                     lhs(i, j + 4),
                                     lhs(i, j + 5),
                                     lhs(i, j + 6),
                                     lhs(i, j + 7)};
                float rhs_temp[8] = {rhs(i, j),
                                     rhs(i, j + 1),
                                     rhs(i, j + 2),
                                     rhs(i, j + 3),
                                     rhs(i, j + 4),
                                     rhs(i, j + 5),
                                     rhs(i, j + 6),
                                     rhs(i, j + 7)};

                __m256 lhs_vec = _mm256_loadu_ps(lhs_temp);
                __m256 rhs_vec = _mm256_loadu_ps(rhs_temp);
                __m256 res_vec = _mm256_add_ps(lhs_vec, rhs_vec);
                _mm256_storeu_ps(lhs_temp, res_vec);

                for (int n = 0; n < 8; ++n) {
                    result(i, j + n) = lhs_temp[n];
                }
            }
            for (; j < lhs.get_cols(); ++j) {
                result(i, j) = lhs(i, j) + rhs(i, j);
            }
        }
    } else {
        for (int i = 0; i < lhs.get_rows(); ++i) {
            for (int j = 0; j < lhs.get_cols(); ++j) {
                result(i, j) = static_cast<CommonType>(lhs(i, j))
                               + static_cast<CommonType>(rhs(i, j));
            }
        }
    }
#else
    for (int i = 0; i < lhs.get_rows(); ++i) {
        for (int j = 0; j < lhs.get_cols(); ++j) {
            result(i, j) =
                static_cast<CommonType>(lhs(i, j)) + static_cast<CommonType>(rhs(i, j));
        }
    }
#endif

    return result;
}

template<typename T, typename U>
auto operator-(const Matrix<T> &lhs, const Matrix<U> &rhs) {
    if (lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    using CommonType = detail::matrix_common_type_t<T, U>;
    Matrix<CommonType> result(lhs.get_rows(), lhs.get_cols());

    if constexpr (detail::is_matrix_v<CommonType>) {
        if (lhs.get_rows() > 0 && lhs.get_cols() > 0) {
            int inner_rows = 1, inner_cols = 1;
            if constexpr (detail::is_matrix_v<T>) {
                inner_rows = lhs(0, 0).get_rows();
                inner_cols = lhs(0, 0).get_cols();
            } else if constexpr (detail::is_matrix_v<U>) {
                inner_rows = rhs(0, 0).get_rows();
                inner_cols = rhs(0, 0).get_cols();
            }
            for (int i = 0; i < result.get_rows(); ++i) {
                for (int j = 0; j < result.get_cols(); ++j) {
                    result(i, j) = CommonType::Zero(inner_rows, inner_cols);
                }
            }
        }
    }

#ifdef __AVX__
    if constexpr (std::is_same_v<T, U> && detail::is_avx_double<T>) {
        for (int i = 0; i < lhs.get_rows(); ++i) {
            int j = 0;
            for (; j + 4 <= lhs.get_cols(); j += 4) {
                double lhs_temp[4] = {lhs(i, j),
                                      lhs(i, j + 1),
                                      lhs(i, j + 2),
                                      lhs(i, j + 3)};
                double rhs_temp[4] = {rhs(i, j),
                                      rhs(i, j + 1),
                                      rhs(i, j + 2),
                                      rhs(i, j + 3)};

                __m256d lhs_vec = _mm256_loadu_pd(lhs_temp);
                __m256d rhs_vec = _mm256_loadu_pd(rhs_temp);
                __m256d res_vec = _mm256_sub_pd(lhs_vec, rhs_vec);
                _mm256_storeu_pd(lhs_temp, res_vec);

                result(i, j) = lhs_temp[0];
                result(i, j + 1) = lhs_temp[1];
                result(i, j + 2) = lhs_temp[2];
                result(i, j + 3) = lhs_temp[3];
            }
            for (; j < lhs.get_cols(); ++j) {
                result(i, j) = lhs(i, j) - rhs(i, j);
            }
        }
    } else if constexpr (std::is_same_v<T, U> && detail::is_avx_float<T>) {
        for (int i = 0; i < lhs.get_rows(); ++i) {
            int j = 0;
            for (; j + 8 <= lhs.get_cols(); j += 8) {
                float lhs_temp[8] = {lhs(i, j),
                                     lhs(i, j + 1),
                                     lhs(i, j + 2),
                                     lhs(i, j + 3),
                                     lhs(i, j + 4),
                                     lhs(i, j + 5),
                                     lhs(i, j + 6),
                                     lhs(i, j + 7)};
                float rhs_temp[8] = {rhs(i, j),
                                     rhs(i, j + 1),
                                     rhs(i, j + 2),
                                     rhs(i, j + 3),
                                     rhs(i, j + 4),
                                     rhs(i, j + 5),
                                     rhs(i, j + 6),
                                     rhs(i, j + 7)};

                __m256 lhs_vec = _mm256_loadu_ps(lhs_temp);
                __m256 rhs_vec = _mm256_loadu_ps(rhs_temp);
                __m256 res_vec = _mm256_sub_ps(lhs_vec, rhs_vec);
                _mm256_storeu_ps(lhs_temp, res_vec);

                for (int n = 0; n < 8; ++n) {
                    result(i, j + n) = lhs_temp[n];
                }
            }
            for (; j < lhs.get_cols(); ++j) {
                result(i, j) = lhs(i, j) - rhs(i, j);
            }
        }
    } else {
        for (int i = 0; i < lhs.get_rows(); ++i) {
            for (int j = 0; j < lhs.get_cols(); ++j) {
                result(i, j) = static_cast<CommonType>(lhs(i, j))
                               - static_cast<CommonType>(rhs(i, j));
            }
        }
    }
#else
    for (int i = 0; i < lhs.get_rows(); ++i) {
        for (int j = 0; j < lhs.get_cols(); ++j) {
            result(i, j) =
                static_cast<CommonType>(lhs(i, j)) - static_cast<CommonType>(rhs(i, j));
        }
    }
#endif

    return result;
}

template<typename T, typename U> auto operator*(const Matrix<T> &A, const Matrix<U> &B) {
    if (A.get_cols() != B.get_rows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    using CommonType = detail::matrix_common_type_t<T, U>;
    Matrix<CommonType> result(A.get_rows(), B.get_cols());

    if constexpr (detail::is_matrix_v<CommonType>) {
        if (A.get_rows() > 0 && A.get_cols() > 0 && B.get_cols() > 0) {
            int inner_rows = 1, inner_cols = 1;
            if constexpr (detail::is_matrix_v<T>) {
                inner_rows = A(0, 0).get_rows();
                inner_cols = A(0, 0).get_cols();
            } else if constexpr (detail::is_matrix_v<U>) {
                inner_rows = B(0, 0).get_rows();
                inner_cols = B(0, 0).get_cols();
            }
            for (int i = 0; i < result.get_rows(); ++i) {
                for (int j = 0; j < result.get_cols(); ++j) {
                    result(i, j) = CommonType::Zero(inner_rows, inner_cols);
                }
            }
        }
    }

#ifdef __AVX__
    if constexpr (std::is_same_v<T, U> && detail::is_avx_double<T>) {
        for (int i = 0; i < A.get_rows(); ++i) {
            for (int k = 0; k < A.get_cols(); ++k) {
                double a_ik = A(i, k);
                __m256d a_vec = _mm256_set1_pd(a_ik);
                int j = 0;
                for (; j + 4 <= B.get_cols(); j += 4) {
                    double b_temp[4] = {B(k, j), B(k, j + 1), B(k, j + 2), B(k, j + 3)};
                    double c_temp[4] = {result(i, j),
                                        result(i, j + 1),
                                        result(i, j + 2),
                                        result(i, j + 3)};

                    __m256d b_vec = _mm256_loadu_pd(b_temp);
                    __m256d c_vec = _mm256_loadu_pd(c_temp);
                    c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec));
                    _mm256_storeu_pd(c_temp, c_vec);

                    result(i, j) = c_temp[0];
                    result(i, j + 1) = c_temp[1];
                    result(i, j + 2) = c_temp[2];
                    result(i, j + 3) = c_temp[3];
                }
                for (; j < B.get_cols(); ++j) {
                    result(i, j) += A(i, k) * B(k, j);
                }
            }
        }
    } else if constexpr (std::is_same_v<T, U> && detail::is_avx_float<T>) {
        for (int i = 0; i < A.get_rows(); ++i) {
            for (int k = 0; k < A.get_cols(); ++k) {
                float a_ik = A(i, k);
                __m256 a_vec = _mm256_set1_ps(a_ik);
                int j = 0;
                for (; j + 8 <= B.get_cols(); j += 8) {
                    float b_temp[8] = {B(k, j),
                                       B(k, j + 1),
                                       B(k, j + 2),
                                       B(k, j + 3),
                                       B(k, j + 4),
                                       B(k, j + 5),
                                       B(k, j + 6),
                                       B(k, j + 7)};
                    float c_temp[8] = {result(i, j),
                                       result(i, j + 1),
                                       result(i, j + 2),
                                       result(i, j + 3),
                                       result(i, j + 4),
                                       result(i, j + 5),
                                       result(i, j + 6),
                                       result(i, j + 7)};

                    __m256 b_vec = _mm256_loadu_ps(b_temp);
                    __m256 c_vec = _mm256_loadu_ps(c_temp);
                    c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));
                    _mm256_storeu_ps(c_temp, c_vec);

                    for (int n = 0; n < 8; ++n) {
                        result(i, j + n) = c_temp[n];
                    }
                }
                for (; j < B.get_cols(); ++j) {
                    result(i, j) += A(i, k) * B(k, j);
                }
            }
        }
    } else {
        for (int i = 0; i < A.get_rows(); ++i) {
            for (int k = 0; k < A.get_cols(); ++k) {
                CommonType a_ik = static_cast<CommonType>(A(i, k));
                for (int j = 0; j < B.get_cols(); ++j) {
                    result(i, j) += a_ik * static_cast<CommonType>(B(k, j));
                }
            }
        }
    }
#else
    for (int i = 0; i < A.get_rows(); ++i) {
        for (int k = 0; k < A.get_cols(); ++k) {
            CommonType a_ik = static_cast<CommonType>(A(i, k));
            for (int j = 0; j < B.get_cols(); ++j) {
                result(i, j) += a_ik * B(k, j);
            }
        }
    }
#endif

    return result;
}

template<typename T, typename U>
auto operator*(const Matrix<T> &matrix, const U &scalar) {
    using CommonType = detail::matrix_common_type_t<T, U>;
    Matrix<CommonType> result(matrix.get_rows(), matrix.get_cols());

#ifdef __AVX__
    if constexpr (std::is_same_v<T, U> && detail::is_avx_double<T>) {
        __m256d scalar_vec = _mm256_set1_pd(scalar);
        for (int i = 0; i < matrix.get_rows(); ++i) {
            int j = 0;
            for (; j + 4 <= matrix.get_cols(); j += 4) {
                double temp[4] = {matrix(i, j),
                                  matrix(i, j + 1),
                                  matrix(i, j + 2),
                                  matrix(i, j + 3)};
                __m256d vec = _mm256_loadu_pd(temp);
                vec = _mm256_mul_pd(vec, scalar_vec);
                _mm256_storeu_pd(temp, vec);

                result(i, j) = temp[0];
                result(i, j + 1) = temp[1];
                result(i, j + 2) = temp[2];
                result(i, j + 3) = temp[3];
            }
            for (; j < matrix.get_cols(); ++j) {
                auto scalar_cast =
                    Matrix<CommonType>::create_scalar(matrix(i, j), scalar);
                result(i, j) = matrix(i, j) * scalar_cast;
            }
        }
    } else if constexpr (std::is_same_v<T, U> && detail::is_avx_float<T>) {
        __m256 scalar_vec = _mm256_set1_ps(scalar);
        for (int i = 0; i < matrix.get_rows(); ++i) {
            int j = 0;
            for (; j + 8 <= matrix.get_cols(); j += 8) {
                float temp[8] = {matrix(i, j),
                                 matrix(i, j + 1),
                                 matrix(i, j + 2),
                                 matrix(i, j + 3),
                                 matrix(i, j + 4),
                                 matrix(i, j + 5),
                                 matrix(i, j + 6),
                                 matrix(i, j + 7)};
                __m256 vec = _mm256_loadu_ps(temp);
                vec = _mm256_mul_ps(vec, scalar_vec);
                _mm256_storeu_ps(temp, vec);

                for (int k = 0; k < 8; ++k) {
                    result(i, j + k) = temp[k];
                }
            }
            for (; j < matrix.get_cols(); ++j) {
                auto scalar_cast =
                    Matrix<CommonType>::create_scalar(matrix(i, j), scalar);
                result(i, j) = matrix(i, j) * scalar_cast;
            }
        }
    } else {
        for (int i = 0; i < matrix.get_rows(); ++i) {
            for (int j = 0; j < matrix.get_cols(); ++j) {
                auto scalar_cast =
                    Matrix<CommonType>::create_scalar(matrix(i, j), scalar);
                result(i, j) = matrix(i, j) * scalar_cast;
            }
        }
    }
#else
    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int j = 0; j < matrix.get_cols(); ++j) {
            auto scalar_cast = Matrix<CommonType>::create_scalar(matrix(i, j), scalar);
            result(i, j) = matrix(i, j) * scalar_cast;
        }
    }
#endif

    return result;
}

template<typename T, typename U>
auto operator/(const Matrix<T> &matrix, const U &scalar) {
    if (Matrix<T>::is_zero(scalar)) {
        throw std::invalid_argument("Division by zero");
    }

    using CommonType = detail::matrix_common_type_t<T, U>;
    Matrix<CommonType> result(matrix.get_rows(), matrix.get_cols());

#ifdef __AVX__
    if constexpr (std::is_same_v<T, U> && detail::is_avx_double<T>) {
        __m256d inv_vec = _mm256_set1_pd(1.0 / scalar);
        for (int i = 0; i < matrix.get_rows(); ++i) {
            int j = 0;
            for (; j + 4 <= matrix.get_cols(); j += 4) {
                double temp[4] = {matrix(i, j),
                                  matrix(i, j + 1),
                                  matrix(i, j + 2),
                                  matrix(i, j + 3)};
                __m256d vec = _mm256_loadu_pd(temp);
                vec = _mm256_mul_pd(vec, inv_vec);
                _mm256_storeu_pd(temp, vec);

                result(i, j) = temp[0];
                result(i, j + 1) = temp[1];
                result(i, j + 2) = temp[2];
                result(i, j + 3) = temp[3];
            }
            for (; j < matrix.get_cols(); ++j) {
                auto scalar_cast =
                    Matrix<CommonType>::create_scalar(matrix(i, j), scalar);
                result(i, j) = matrix(i, j) / scalar_cast;
            }
        }
    } else if constexpr (std::is_same_v<T, U> && detail::is_avx_float<T>) {
        __m256 inv_vec = _mm256_set1_ps(1.0f / scalar);
        for (int i = 0; i < matrix.get_rows(); ++i) {
            int j = 0;
            for (; j + 8 <= matrix.get_cols(); j += 8) {
                float temp[8] = {matrix(i, j),
                                 matrix(i, j + 1),
                                 matrix(i, j + 2),
                                 matrix(i, j + 3),
                                 matrix(i, j + 4),
                                 matrix(i, j + 5),
                                 matrix(i, j + 6),
                                 matrix(i, j + 7)};
                __m256 vec = _mm256_loadu_ps(temp);
                vec = _mm256_mul_ps(vec, inv_vec);
                _mm256_storeu_ps(temp, vec);

                for (int k = 0; k < 8; ++k) {
                    result(i, j + k) = temp[k];
                }
            }
            for (; j < matrix.get_cols(); ++j) {
                auto scalar_cast =
                    Matrix<CommonType>::create_scalar(matrix(i, j), scalar);
                result(i, j) = matrix(i, j) / scalar_cast;
            }
        }
    } else {
        for (int i = 0; i < matrix.get_rows(); ++i) {
            for (int j = 0; j < matrix.get_cols(); ++j) {
                auto scalar_cast =
                    Matrix<CommonType>::create_scalar(matrix(i, j), scalar);
                result(i, j) = matrix(i, j) / scalar_cast;
            }
        }
    }
#else
    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int j = 0; j < matrix.get_cols(); ++j) {
            auto scalar_cast = Matrix<CommonType>::create_scalar(matrix(i, j), scalar);
            result(i, j) = matrix(i, j) / scalar_cast;
        }
    }
#endif

    return result;
}

template<typename T, typename U>
auto operator*(const U &scalar, const Matrix<T> &matrix) {
    return matrix * scalar;
}

template<typename T, typename U>
auto operator+(const Matrix<T> &matrix, const U &scalar) {
    using CommonType = detail::matrix_common_type_t<T, U>;
    Matrix<CommonType> result(matrix.get_rows(), matrix.get_cols());

    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int j = 0; j < matrix.get_cols(); ++j) {
            result(i, j) = static_cast<CommonType>(matrix(i, j));
        }
    }

    int min_dim = std::min(matrix.get_rows(), matrix.get_cols());
    for (int i = 0; i < min_dim; ++i) {
        result(i, i) += static_cast<CommonType>(scalar);
    }

    return result;
}

template<typename T, typename U>
auto operator+(const U &scalar, const Matrix<T> &matrix) {
    return matrix + scalar;
}

template<typename T, typename U>
auto operator-(const Matrix<T> &matrix, const U &scalar) {
    using CommonType = detail::matrix_common_type_t<T, U>;
    Matrix<CommonType> result(matrix.get_rows(), matrix.get_cols());

    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int j = 0; j < matrix.get_cols(); ++j) {
            result(i, j) = static_cast<CommonType>(matrix(i, j));
        }
    }

    int min_dim = std::min(matrix.get_rows(), matrix.get_cols());
    for (int i = 0; i < min_dim; ++i) {
        result(i, i) -= static_cast<CommonType>(scalar);
    }

    return result;
}

template<typename T, typename U>
auto operator-(const U &scalar, const Matrix<T> &matrix) {
    using CommonType = detail::matrix_common_type_t<U, T>;
    Matrix<CommonType> result(matrix.get_rows(), matrix.get_cols());

    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int j = 0; j < matrix.get_cols(); ++j) {
            if (i == j && i < std::min(matrix.get_rows(), matrix.get_cols())) {
                result(i, j) = static_cast<CommonType>(scalar)
                               - static_cast<CommonType>(matrix(i, j));
            } else {
                result(i, j) = -static_cast<CommonType>(matrix(i, j));
            }
        }
    }

    return result;
}

template<typename T, typename U> auto operator/(const Matrix<T> &A, const Matrix<U> &B) {
    if (A.get_cols() != B.get_rows()) {
        throw std::invalid_argument(
            "Matrix dimensions don't match for division (A.cols != B.rows)");
    }

    try {
        using ResultType = typename Matrix<T>::template inverse_return_type<T>;

        Matrix<ResultType> A_cast = A.template cast_to<ResultType>();
        Matrix<ResultType> B_cast = B.template cast_to<ResultType>();

        Matrix<ResultType> B_inv = B_cast.inverse();

        return A_cast * B_inv;
    } catch (const std::exception &e) {
        throw std::runtime_error("Cannot divide by singular matrix B");
    }
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix) {
    if constexpr (detail::is_matrix_v<T>) {
        if (matrix.rows_ <= 3 && matrix.cols_ <= 3) {
            os << "BlockMatrix " << matrix.rows_ << "x" << matrix.cols_ << ":\n";
            for (int i = 0; i < matrix.rows_; ++i) {
                for (int inner_i = 0; inner_i < matrix(0, 0).get_rows(); ++inner_i) {
                    os << "  ";
                    for (int j = 0; j < matrix.cols_; ++j) {
                        const auto &block = matrix(i, j);
                        os << "[";
                        for (int inner_j = 0; inner_j < block.get_cols(); ++inner_j) {
                            os << block(inner_i, inner_j);
                            if (inner_j < block.get_cols() - 1)
                                os << " ";
                        }
                        os << "]";
                        if (j < matrix.cols_ - 1)
                            os << " ";
                    }
                    os << "\n";
                }
                if (i < matrix.rows_ - 1)
                    os << "\n";
            }
        } else {
            os << "[BlockMatrix " << matrix.rows_ << "x" << matrix.cols_ << " of "
               << matrix(0, 0).get_rows() << "x" << matrix(0, 0).get_cols() << "]";
        }
    } else {
        if (matrix.rows_ <= 5 && matrix.cols_ <= 5) {
            for (int i = 0; i < matrix.rows_; ++i) {
                for (int j = 0; j < matrix.cols_; ++j) {
                    os << matrix(i, j);
                    if (j < matrix.cols_ - 1)
                        os << " ";
                }
                if (i < matrix.rows_ - 1)
                    os << "\n";
            }
        } else {
            os << "[Matrix " << matrix.rows_ << "x" << matrix.cols_ << "]";
        }
    }
    return os;
}
