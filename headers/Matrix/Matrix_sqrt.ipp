#include <limits>
#include <cmath>
#include <string>

template<typename T> struct real_type_from_norm {
    using type = T;
};

template<typename T> struct real_type_from_norm<std::complex<T>> {
    using type = T;
};

template<typename T> using real_type_from_norm_t = typename real_type_from_norm<T>::type;

template<typename T>
Matrix<typename Matrix<T>::template sqrt_return_type<T>> sqrt(const Matrix<T> &m) {
    return m.sqrt();
}

template<typename T>
Matrix<typename Matrix<T>::template sqrt_return_type<T>> Matrix<T>::sqrt() const {
    using ResultType = sqrt_return_type<T>;

    if (min_dim_ == 0) {
        throw std::invalid_argument("Cannot compute square root of empty matrix");
    }

    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix square root requires square matrix");
    }

    if constexpr (detail::is_matrix_v<T>) {
        return compute_block_matrix_sqrt<ResultType>();
    } else {
        return compute_scalar_matrix_sqrt<ResultType>();
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::compute_scalar_matrix_sqrt() const {
    try {
        Matrix<ResultType> A = this->template cast_to<ResultType>();

        if (rows_ == 1) {
            return compute_1x1_sqrt<ResultType>(A);
        }

        if (rows_ == 2) {
            return compute_2x2_scalar_sqrt<ResultType>(A);
        }

        auto tolerance_value = Matrix<ResultType>::create_scalar(ResultType{}, 1e-10);
        return compute_scalar_newton_sqrt<ResultType>(A, 100, tolerance_value);
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to compute matrix square root: "
                                 + std::string(e.what()));
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::compute_1x1_sqrt(const Matrix<ResultType> &A) const {
    Matrix<ResultType> result(1, 1);
    try {
        if constexpr (detail::is_builtin_integral_v<ResultType>) {
            result(0, 0) =
                static_cast<ResultType>(std::sqrt(static_cast<double>(A(0, 0))));
        } else if constexpr (detail::is_matrix_v<ResultType>) {
            result(0, 0) = A(0, 0).sqrt();
        } else {
            using std::sqrt;
            result(0, 0) = sqrt(A(0, 0));
        }
    } catch (...) {
        throw std::runtime_error("Failed to compute 1x1 matrix square root");
    }
    return result;
}

template<typename T>
template<typename ResultType>
Matrix<ResultType>
Matrix<T>::compute_2x2_scalar_sqrt(const Matrix<ResultType> &A) const {
    try {
        ResultType a = static_cast<ResultType>(A(0, 0));
        ResultType b = static_cast<ResultType>(A(0, 1));
        ResultType c = static_cast<ResultType>(A(1, 0));
        ResultType d = static_cast<ResultType>(A(1, 1));

        if (is_zero(b) && is_zero(c)) {
            return compute_diagonal_2x2_scalar_sqrt<ResultType>(a, d);
        }

        return compute_full_2x2_scalar_sqrt<ResultType>(a, b, c, d);
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed in 2x2 matrix square root: "
                                 + std::string(e.what()));
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType>
Matrix<T>::compute_diagonal_2x2_scalar_sqrt(const ResultType &a,
                                            const ResultType &d) const {
    Matrix<ResultType> result(2, 2);
    try {
        if constexpr (detail::is_builtin_integral_v<ResultType>) {
            result(0, 0) = static_cast<ResultType>(std::sqrt(static_cast<double>(a)));
            result(1, 1) = static_cast<ResultType>(std::sqrt(static_cast<double>(d)));
        } else if constexpr (detail::is_matrix_v<ResultType>) {
            result(0, 0) = a.sqrt();
            result(1, 1) = d.sqrt();
        } else {
            using std::sqrt;
            result(0, 0) = sqrt(a);
            result(1, 1) = sqrt(d);
        }
    } catch (...) {
        throw std::runtime_error("Failed to compute diagonal 2x2 sqrt");
    }
    return result;
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::compute_full_2x2_scalar_sqrt(const ResultType &a,
                                                           const ResultType &b,
                                                           const ResultType &c,
                                                           const ResultType &d) const {
    try {
        ResultType det = a * d - b * c;
        ResultType tr = a + d;

        ResultType four, two;

        if constexpr (detail::is_matrix_v<ResultType>) {
            using ElemType = typename ResultType::value_type;
            ElemType elem_4 = static_cast<ElemType>(4);
            ElemType elem_2 = static_cast<ElemType>(2);

            int block_rows = 1, block_cols = 1;
            try {
                block_rows = a.get_rows();
                block_cols = a.get_cols();
            } catch (...) {
            }

            four = ResultType::Identity(block_rows, block_cols) * elem_4;
            two = ResultType::Identity(block_rows, block_cols) * elem_2;
        } else {
            four = static_cast<ResultType>(4);
            two = static_cast<ResultType>(2);
        }

        ResultType delta = tr * tr - four * det;
        ResultType sqrt_delta = compute_scalar_sqrt<ResultType>(delta);

        ResultType s_plus =
            compute_scalar_sqrt_of_half_sum<ResultType>(tr, sqrt_delta, two, true);
        ResultType s_minus =
            compute_scalar_sqrt_of_half_sum<ResultType>(tr, sqrt_delta, two, false);

        return compute_final_2x2_scalar_result<ResultType>(a,
                                                           b,
                                                           c,
                                                           d,
                                                           s_plus,
                                                           s_minus,
                                                           two);
    } catch (const std::exception &e) {
        throw std::runtime_error(
            "Failed to compute 2x2 sqrt with non-diagonal elements: "
            + std::string(e.what()));
    }
}

template<typename T>
template<typename ResultType>
ResultType Matrix<T>::compute_scalar_sqrt(const ResultType &value) const {
    try {
        if constexpr (detail::is_builtin_integral_v<ResultType>) {
            return static_cast<ResultType>(std::sqrt(static_cast<double>(value)));
        } else if constexpr (detail::is_matrix_v<ResultType>) {
            return value.sqrt();
        } else {
            using std::sqrt;
            return sqrt(value);
        }
    } catch (...) {
        throw std::runtime_error("Failed to compute sqrt");
    }
}

template<typename T>
template<typename ResultType>
ResultType Matrix<T>::compute_scalar_sqrt_of_half_sum(const ResultType &tr,
                                                      const ResultType &sqrt_delta,
                                                      const ResultType &two,
                                                      bool plus) const {
    try {
        if constexpr (detail::is_builtin_integral_v<ResultType>) {
            ResultType arg = plus ? (tr + sqrt_delta) / two : (tr - sqrt_delta) / two;
            return static_cast<ResultType>(std::sqrt(static_cast<double>(arg)));
        } else if constexpr (detail::is_matrix_v<ResultType>) {
            ResultType arg = plus ? (tr + sqrt_delta) / two : (tr - sqrt_delta) / two;
            return arg.sqrt();
        } else {
            using std::sqrt;
            ResultType arg = plus ? (tr + sqrt_delta) / two : (tr - sqrt_delta) / two;
            return sqrt(arg);
        }
    } catch (...) {
        throw std::runtime_error("Failed to compute sqrt of half sum");
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType>
Matrix<T>::compute_final_2x2_scalar_result(const ResultType &a,
                                           const ResultType &b,
                                           const ResultType &c,
                                           const ResultType &d,
                                           const ResultType &s_plus,
                                           const ResultType &s_minus,
                                           const ResultType &two) const {
    Matrix<ResultType> result(2, 2);
    try {
        ResultType denominator = s_plus + s_minus;

        if (!is_zero(s_plus - s_minus)) {
            result(0, 0) = (a + s_plus * s_minus) / denominator;
            result(0, 1) = b / denominator;
            result(1, 0) = c / denominator;
            result(1, 1) = (d + s_plus * s_minus) / denominator;
        } else {
            ResultType s = s_plus;
            result(0, 0) = (a + s * s) / (two * s);
            result(0, 1) = b / (two * s);
            result(1, 0) = c / (two * s);
            result(1, 1) = (d + s * s) / (two * s);
        }
    } catch (...) {
        throw std::runtime_error("Failed to compute final 2x2 result");
    }
    return result;
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::compute_scalar_newton_sqrt(const Matrix<ResultType> &A,
                                                         int max_iter,
                                                         ResultType tolerance) const {
    int n = A.get_rows();

    if (n == 0) {
        return Matrix<ResultType>();
    }

    try {
        Matrix<ResultType> X = A;
        Matrix<ResultType> X_initial;

        if constexpr (detail::is_matrix_v<ResultType>) {
            int block_rows = 1, block_cols = 1;
            try {
                if (n > 0 && X(0, 0).get_rows() > 0) {
                    block_rows = X(0, 0).get_rows();
                    block_cols = X(0, 0).get_cols();
                }
            } catch (...) {
                block_rows = 1;
                block_cols = 1;
            }

            X_initial = Matrix<ResultType>::BlockIdentity(n, n, block_rows, block_cols);

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        try {
                            X_initial(i, j) = ResultType::Zero(block_rows, block_cols);
                        } catch (...) {
                            throw std::runtime_error("Failed to initialize zero block");
                        }
                    }
                }
            }
        } else {
            try {
                X_initial = Matrix<ResultType>::Identity(n);
            } catch (...) {
                throw std::runtime_error("Failed to initialize identity matrix");
            }
        }

        X = X_initial;

        using NormType = typename Matrix<ResultType>::norm_return_type;
        using RealType = real_type_from_norm_t<NormType>;

        RealType tol_norm;
        try {
            if constexpr (detail::is_matrix_v<ResultType>) {
                tol_norm = static_cast<RealType>(tolerance.frobenius_norm());
            } else if constexpr (detail::is_complex_v<ResultType>) {
                using std::abs;
                tol_norm = static_cast<RealType>(abs(tolerance));
            } else {
                tol_norm = static_cast<RealType>(tolerance);
            }
        } catch (...) {
            tol_norm = static_cast<RealType>(1e-10);
        }

        return perform_scalar_newton_iterations<ResultType, NormType, RealType>(
            A,
            X,
            n,
            max_iter,
            tol_norm);
    } catch (const std::exception &e) {
        throw std::runtime_error("Newton iteration failed: " + std::string(e.what()));
    }
}

template<typename T>
template<typename ResultType, typename NormType, typename RealType>
Matrix<ResultType>
Matrix<T>::perform_scalar_newton_iterations(const Matrix<ResultType> &A,
                                            Matrix<ResultType> X,
                                            int n,
                                            int max_iter,
                                            RealType tol_norm) const {
    Matrix<ResultType> best_X = X;
    RealType best_residual = std::numeric_limits<RealType>::max();
    bool any_successful_update = false;
    std::string last_error_msg;

    try {
        NormType norm_residual = (X * X - A).frobenius_norm();
        if constexpr (detail::is_complex_v<NormType>) {
            using std::abs;
            best_residual = abs(norm_residual);
        } else {
            best_residual = norm_residual;
        }
    } catch (...) {
        best_residual = std::numeric_limits<RealType>::max();
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        try {
            bool iteration_success =
                perform_single_scalar_newton_iteration<ResultType>(A, X, n, iter);
            if (iteration_success) {
                any_successful_update = true;

                Matrix<ResultType> X_prev;
                try {
                    X_prev = X;
                } catch (...) {
                    X_prev = best_X;
                }

                RealType error, residual;
                try {
                    NormType norm_error = (X - X_prev).frobenius_norm();
                    NormType norm_residual = (X * X - A).frobenius_norm();

                    if constexpr (detail::is_complex_v<NormType>) {
                        using std::abs;
                        error = abs(norm_error);
                        residual = abs(norm_residual);
                    } else {
                        error = norm_error;
                        residual = norm_residual;
                    }

                    if (residual < best_residual) {
                        best_X = X;
                        best_residual = residual;
                    }

                    if (error < tol_norm && residual < tol_norm * 10) {
                        return X;
                    }
                } catch (...) {
                    // Ignoring exception in error/residual calculation to continue
                    // iterations The iteration might still converge despite measurement
                    // issues
                    DEBUG_PRINTF(
                        "Warning: Failed to compute error/residual norms at iteration %d, continuing with next iteration",
                        iter);
                }
            }
        } catch (const std::exception &e) {
            last_error_msg = e.what();
            if (iter > 0 && any_successful_update) {
                std::cerr << "Warning: Newton iteration failed at iteration " << iter
                          << ": " << last_error_msg << "\n";
                std::cerr << "Returning best approximation found so far (residual: "
                          << best_residual << ")\n";
                return best_X;
            }
        } catch (...) {
            last_error_msg = "Unknown error in Newton iteration";
            if (iter > 0 && any_successful_update) {
                std::cerr << "Warning: Newton iteration failed at iteration " << iter
                          << ": " << last_error_msg << "\n";
                std::cerr << "Returning best approximation found so far (residual: "
                          << best_residual << ")\n";
                return best_X;
            }
        }
    }

    if (!any_successful_update) {
        throw std::runtime_error("Failed to perform any Newton iteration: "
                                 + last_error_msg);
    }

    return best_X;
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::perform_single_scalar_newton_iteration(const Matrix<ResultType> &A,
                                                       Matrix<ResultType> &X,
                                                       int n,
                                                       int iter) const {
    Matrix<ResultType> X_prev;
    try {
        X_prev = X;
    } catch (...) {
        throw std::runtime_error("Failed to save previous iteration state");
    }

    try {
        Matrix<ResultType> X_inv = compute_scalar_matrix_inverse<ResultType>(X, n, iter);

        if constexpr (detail::is_matrix_v<ResultType>) {
            using ElemType = typename ResultType::value_type;
            ElemType example_elem;
            try {
                if (X.get_rows() > 0 && X.get_cols() > 0) {
                    example_elem = static_cast<ElemType>(X(0, 0)(0, 0));
                } else {
                    example_elem = ElemType(0);
                }
            } catch (...) {
                example_elem = ElemType(0);
            }

            auto half_scalar = Matrix<ElemType>::create_scalar(example_elem, 0.5);

            Matrix<ResultType> temp;
            try {
                temp = X_inv * A;
            } catch (...) {
                throw std::runtime_error("Failed to multiply X_inv * A");
            }

            Matrix<ResultType> X_new(n, n);

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    try {
                        X_new(i, j) = (X(i, j) + temp(i, j)) * half_scalar;
                    } catch (...) {
                        throw std::runtime_error("Failed to compute new element");
                    }
                }
            }
            X = X_new;
        } else {
            auto half = Matrix<ResultType>::create_scalar(ResultType{}, 0.5);
            try {
                X = (X + X_inv * A) * half;
            } catch (...) {
                throw std::runtime_error("Failed to compute Newton update");
            }
        }

        return true;
    } catch (const std::exception &e) {
        if (iter == 0) {
            throw;
        }
        return false;
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType>
Matrix<T>::compute_scalar_matrix_inverse(Matrix<ResultType> &X, int n, int iter) const {
    try {
        return X.inverse();
    } catch (const std::exception &) {
        try {
            apply_scalar_shift_to_matrix<ResultType>(X, n);
            return X.inverse();
        } catch (const std::exception &e2) {
            if (iter == 0) {
                throw std::runtime_error("Failed to invert matrix even with shift: "
                                         + std::string(e2.what()));
            }
            throw;
        }
    }
}

template<typename T>
template<typename ResultType>
void Matrix<T>::apply_scalar_shift_to_matrix(Matrix<ResultType> &X, int n) const {
    if constexpr (detail::is_matrix_v<ResultType>) {
        using ElemType = typename ResultType::value_type;
        ElemType example_elem;
        try {
            if (X.get_rows() > 0 && X.get_cols() > 0) {
                example_elem = static_cast<ElemType>(X(0, 0)(0, 0));
            } else {
                example_elem = ElemType(0);
            }
        } catch (...) {
            // Block size information is not critical - using default value
            // Shift will still be applied with default element
            DEBUG_PRINTF(
                "Warning: Failed to get example element for shift, using default");
            example_elem = ElemType(0);
        }

        auto shift_scalar = Matrix<ElemType>::create_scalar(example_elem, 1e-6);

        int block_rows = 1, block_cols = 1;
        try {
            if (n > 0 && X(0, 0).get_rows() > 0) {
                block_rows = X(0, 0).get_rows();
                block_cols = X(0, 0).get_cols();
            }
        } catch (...) {
            // Block dimensions not critical - using default 1x1
            // Identity matrix of any size multiplied by scalar will still work
            DEBUG_PRINTF("Warning: Failed to get block dimensions, using 1x1");
            block_rows = 1;
            block_cols = 1;
        }

        ResultType shift;
        try {
            shift = ResultType::Identity(block_rows, block_cols) * shift_scalar;
        } catch (...) {
            throw std::runtime_error("Failed to create shift matrix");
        }

        for (int i = 0; i < n; ++i) {
            try {
                X(i, i) = X(i, i) + shift;
            } catch (...) {
                throw std::runtime_error("Failed to apply shift to diagonal element");
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            try {
                X(i, i) = X(i, i) + static_cast<ResultType>(1e-6);
            } catch (...) {
                throw std::runtime_error("Failed to apply shift to diagonal element");
            }
        }
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::compute_block_matrix_sqrt() const {
    if (rows_ == 1) {
        return compute_block_1x1_sqrt<ResultType>();
    }

    if (rows_ == 2) {
        return compute_block_2x2_sqrt<ResultType>();
    }

    int block_rows = 1;
    int block_cols = 1;

    try {
        if (rows_ > 0 && cols_ > 0) {
            block_rows = static_cast<ResultType>((*this)(0, 0)).get_rows();
            block_cols = static_cast<ResultType>((*this)(0, 0)).get_cols();
            if (block_rows == 0)
                block_rows = 1;
            if (block_cols == 0)
                block_cols = 1;
        }
    } catch (...) {
        block_rows = 1;
        block_cols = 1;
    }

    Matrix<ResultType> result(rows_, cols_);

    bool is_block_diagonal = check_block_diagonal<ResultType>();

    if (is_block_diagonal) {
        return compute_block_diagonal_sqrt<ResultType>(result, block_rows, block_cols);
    } else {
        return compute_block_non_diagonal_sqrt<ResultType>(block_rows, block_cols);
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::compute_block_1x1_sqrt() const {
    Matrix<ResultType> result(1, 1);
    try {
        result(0, 0) = static_cast<ResultType>((*this)(0, 0)).sqrt();
    } catch (...) {
        throw std::runtime_error("Failed to compute block 1x1 sqrt");
    }
    return result;
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::compute_block_2x2_sqrt() const {
    try {
        ResultType a = static_cast<ResultType>((*this)(0, 0));
        ResultType b = static_cast<ResultType>((*this)(0, 1));
        ResultType c = static_cast<ResultType>((*this)(1, 0));
        ResultType d = static_cast<ResultType>((*this)(1, 1));

        if (is_zero(b) && is_zero(c)) {
            Matrix<ResultType> result(2, 2);
            result(0, 0) = a.sqrt();
            result(1, 1) = d.sqrt();

            int block_rows = 1, block_cols = 1;
            try {
                block_rows = a.get_rows();
                block_cols = a.get_cols();
            } catch (...) {
            }

            result(0, 1) = ResultType::Zero(block_rows, block_cols);
            result(1, 0) = ResultType::Zero(block_rows, block_cols);
            return result;
        }

        return compute_block_non_diagonal_sqrt<ResultType>(a.get_rows(), a.get_cols());
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed in block 2x2 matrix square root: "
                                 + std::string(e.what()));
    }
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::check_block_diagonal() const {
    try {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (i != j && !is_zero((*this)(i, j))) {
                    return false;
                }
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::compute_block_diagonal_sqrt(Matrix<ResultType> &result,
                                                          int block_rows,
                                                          int block_cols) const {
    for (int i = 0; i < rows_; ++i) {
        try {
            result(i, i) = static_cast<ResultType>((*this)(i, i)).sqrt();
        } catch (const std::exception &e) {
            throw std::runtime_error("Failed to compute sqrt of diagonal block: "
                                     + std::string(e.what()));
        }

        for (int j = 0; j < cols_; ++j) {
            if (i != j) {
                try {
                    result(i, j) = ResultType::Zero(block_rows, block_cols);
                } catch (...) {
                    throw std::runtime_error("Failed to create zero block");
                }
            }
        }
    }
    return result;
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::compute_block_non_diagonal_sqrt(int block_rows,
                                                              int block_cols) const {
    using ElemType = typename ResultType::value_type;
    ElemType example_elem;

    try {
        if (rows_ > 0 && cols_ > 0) {
            example_elem = static_cast<ElemType>((*this)(0, 0)(0, 0));
        } else {
            example_elem = ElemType(0);
        }
    } catch (...) {
        example_elem = ElemType(0);
    }

    auto tolerance_scalar = Matrix<ElemType>::create_scalar(example_elem, 1e-10);
    ResultType tolerance;

    try {
        tolerance = ResultType::Identity(block_rows, block_cols) * tolerance_scalar;
    } catch (...) {
        throw std::runtime_error("Failed to create tolerance matrix");
    }

    Matrix<ResultType> A = this->template cast_to<ResultType>();
    return compute_scalar_newton_sqrt<ResultType>(A, 50, tolerance);
}

template<typename T> bool Matrix<T>::has_square_root() const {
    if (min_dim_ == 0) {
        return false;
    }

    if (rows_ != cols_) {
        return false;
    }

    if constexpr (detail::is_builtin_integral_v<T>) {
        using ResultType = sqrt_return_type<T>;
        return has_square_root_for_type<ResultType>();
    } else {
        return has_square_root_for_type<T>();
    }
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::has_square_root_for_type() const {
    if (rows_ == 1) {
        return check_1x1_has_square_root<ResultType>();
    }

    if (rows_ == 2) {
        return check_2x2_has_square_root<ResultType>();
    }

    constexpr bool should_check_directly = std::is_arithmetic_v<ResultType>
                                           && !detail::is_complex_v<ResultType>
                                           && !detail::is_matrix_v<ResultType>;

    if constexpr (should_check_directly) {
        bool symmetric;
        try {
            symmetric = is_symmetric();
        } catch (...) {
            symmetric = false;
        }

        bool positive_diag = check_positive_diagonal<ResultType>();

        if (symmetric && positive_diag) {
            return true;
        }

        try {
            return has_square_root_direct_check<ResultType>();
        } catch (...) {
            return false;
        }
    } else {
        return true;
    }
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::check_1x1_has_square_root() const {
    if constexpr (std::is_arithmetic_v<ResultType>
                  && !detail::is_complex_v<ResultType>) {
        try {
            ResultType val = static_cast<ResultType>((*this)(0, 0));
            if constexpr (detail::is_matrix_v<ResultType>) {
                return true;
            } else {
                return val >= ResultType(0);
            }
        } catch (...) {
            return false;
        }
    }
    return true;
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::check_2x2_has_square_root() const {
    if constexpr (std::is_arithmetic_v<ResultType>
                  && !detail::is_complex_v<ResultType>) {
        try {
            ResultType a = static_cast<ResultType>((*this)(0, 0));
            ResultType b = static_cast<ResultType>((*this)(0, 1));
            ResultType c = static_cast<ResultType>((*this)(1, 0));
            ResultType d = static_cast<ResultType>((*this)(1, 1));

            ResultType tr = a + d;
            ResultType det = a * d - b * c;

            ResultType discr = tr * tr - 4 * det;
            if (discr < ResultType(0)) {
                return false;
            }

            ResultType sqrt_discr;
            try {
                sqrt_discr = std::sqrt(discr);
            } catch (...) {
                return false;
            }

            ResultType lambda1 = (tr + sqrt_discr) / ResultType(2);
            ResultType lambda2 = (tr - sqrt_discr) / ResultType(2);

            return lambda1 >= ResultType(0) && lambda2 >= ResultType(0);
        } catch (...) {
            return false;
        }
    }

    return true;
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::check_positive_diagonal() const {
    try {
        for (int i = 0; i < rows_; ++i) {
            ResultType diag = static_cast<ResultType>((*this)(i, i));
            if (diag <= ResultType(0)) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::has_square_root_direct_check() const {
    try {
        for (int i = 0; i < rows_; ++i) {
            ResultType diag = static_cast<ResultType>((*this)(i, i));
            if (diag < ResultType(0)) {
                return false;
            }
        }

        Matrix<ResultType> test_sqrt = compute_scalar_matrix_sqrt<ResultType>();
        Matrix<ResultType> check = test_sqrt * test_sqrt;
        Matrix<ResultType> A = this->template cast_to<ResultType>();

        using NormType = typename decltype(check - A)::norm_return_type;
        NormType norm_error = (check - A).frobenius_norm();

        return norm_error < static_cast<NormType>(1e-6);
    } catch (...) {
        return false;
    }
}

template<typename T>
std::pair<Matrix<typename Matrix<T>::template sqrt_return_type<T>>, bool>
Matrix<T>::safe_sqrt() const {
    using ResultType = typename Matrix<T>::template sqrt_return_type<T>;

    if (min_dim_ == 0) {
        try {
            return {Matrix<ResultType>(), true};
        } catch (...) {
            return {Matrix<ResultType>(), false};
        }
    }

    if (rows_ != cols_) {
        try {
            return {Matrix<ResultType>(), false};
        } catch (...) {
            return {Matrix<ResultType>(), false};
        }
    }

    try {
        Matrix<ResultType> result = this->sqrt();
        return {result, true};
    } catch (...) {
        try {
            Matrix<ResultType> approx = create_approximate_sqrt<ResultType>();
            return {approx, false};
        } catch (...) {
            try {
                return {Matrix<ResultType>(rows_, cols_), false};
            } catch (...) {
                return {Matrix<ResultType>(), false};
            }
        }
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::create_approximate_sqrt() const {
    Matrix<ResultType> approx(rows_, cols_);

    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            try {
                approx(i, j) = compute_approximate_element<ResultType>(i, j);
            } catch (...) {
                try {
                    approx(i, j) = create_default_identity_element<ResultType>();
                } catch (...) {
                    throw std::runtime_error(
                        "Failed to create approximate sqrt element");
                }
            }
        }
    }

    try {
        check_and_fix_nan_elements<ResultType>(approx);
    } catch (...) {
        // If failed to check/fix NaN elements, continue with current approximation
        // This is acceptable as this is already a fallback method
        DEBUG_PRINTF(
            "Warning: Failed to check/fix NaN elements in approximate sqrt, continuing with current approximation");
    }

    return approx;
}

template<typename T>
template<typename ResultType>
ResultType Matrix<T>::compute_approximate_element(int i, int j) const {
    if (i == j) {
        return compute_approximate_diagonal_element<ResultType>(i);
    } else {
        return compute_approximate_off_diagonal_element<ResultType>();
    }
}

template<typename T>
template<typename ResultType>
ResultType Matrix<T>::compute_approximate_diagonal_element(int i) const {
    try {
        ResultType val = static_cast<ResultType>((*this)(i, i));

        if constexpr (detail::is_matrix_v<ResultType>) {
            auto [block_sqrt, block_success] = val.safe_sqrt();
            return block_sqrt;
        } else {
            using std::abs;
            using std::sqrt;

            if constexpr (std::is_arithmetic_v<ResultType>
                          && !detail::is_complex_v<ResultType>) {
                if (val >= ResultType(0)) {
                    return sqrt(val);
                } else {
                    return sqrt(abs(val));
                }
            } else if constexpr (detail::is_complex_v<ResultType>) {
                return sqrt(val);
            } else {
                return sqrt(val);
            }
        }
    } catch (...) {
        return create_default_identity_element<ResultType>();
    }
}

template<typename T>
template<typename ResultType>
ResultType Matrix<T>::compute_approximate_off_diagonal_element() const {
    if constexpr (detail::is_matrix_v<ResultType>) {
        int block_rows = 1, block_cols = 1;
        if (rows_ > 0 && cols_ > 0) {
            try {
                auto first_val = static_cast<ResultType>((*this)(0, 0));
                block_rows = first_val.get_rows();
                block_cols = first_val.get_cols();
            } catch (...) {
            }
        }
        try {
            return ResultType::Zero(block_rows, block_cols);
        } catch (...) {
            throw std::runtime_error("Failed to create zero block");
        }
    } else {
        try {
            return ResultType(0);
        } catch (...) {
            throw std::runtime_error("Failed to create zero");
        }
    }
}

template<typename T>
template<typename ResultType>
ResultType Matrix<T>::create_default_identity_element() const {
    if constexpr (detail::is_matrix_v<ResultType>) {
        int block_rows = 1, block_cols = 1;
        if (rows_ > 0 && cols_ > 0) {
            try {
                auto first_val = static_cast<ResultType>((*this)(0, 0));
                block_rows = first_val.get_rows();
                block_cols = first_val.get_cols();
            } catch (...) {
            }
        }
        try {
            return ResultType::Identity(block_rows, block_cols);
        } catch (...) {
            throw std::runtime_error("Failed to create identity block");
        }
    } else {
        try {
            return ResultType(1);
        } catch (...) {
            throw std::runtime_error("Failed to create identity element");
        }
    }
}

template<typename T>
template<typename ResultType>
void Matrix<T>::check_and_fix_nan_elements(Matrix<ResultType> &approx) const {
    try {
        bool has_nan = check_for_nan_elements<ResultType>(approx);

        if (has_nan) {
            try {
                approx = create_fallback_identity_matrix<ResultType>();
            } catch (...) {
                // Ignoring exception - cannot create fallback, but better to return
                // current approximation than nothing at all
                DEBUG_PRINTF(
                    "Warning: Failed to create fallback identity matrix, keeping current approximation");
            }
        }
    } catch (...) {
        // Ignoring exception - NaN check failed, but we should continue with
        // current approximation rather than failing completely
        DEBUG_PRINTF(
            "Warning: Failed to check for NaN elements, continuing with current approximation");
    }
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::check_for_nan_elements(const Matrix<ResultType> &approx) const {
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            if constexpr (detail::is_matrix_v<ResultType>) {
                auto &block = approx(i, j);
                for (int bi = 0; bi < block.get_rows(); ++bi) {
                    for (int bj = 0; bj < block.get_cols(); ++bj) {
                        if (std::isnan(static_cast<double>(block(bi, bj)))) {
                            return true;
                        }
                    }
                }
            } else if constexpr (std::is_arithmetic_v<ResultType>) {
                if (std::isnan(static_cast<double>(approx(i, j)))) {
                    return true;
                }
            }
        }
    }
    return false;
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::create_fallback_identity_matrix() const {
    if constexpr (detail::is_matrix_v<ResultType>) {
        int block_rows = 1, block_cols = 1;
        if (rows_ > 0 && cols_ > 0) {
            try {
                auto first_val = static_cast<ResultType>((*this)(0, 0));
                block_rows = first_val.get_rows();
                block_cols = first_val.get_cols();
            } catch (...) {
            }
        }
        try {
            return Matrix<ResultType>::BlockIdentity(rows_,
                                                     cols_,
                                                     block_rows,
                                                     block_cols);
        } catch (...) {
            throw std::runtime_error("Failed to create block identity matrix");
        }
    } else {
        try {
            return Matrix<ResultType>::Identity(rows_);
        } catch (...) {
            throw std::runtime_error("Failed to create identity matrix");
        }
    }
}
