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

    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix square root requires square matrix");
    }

    if constexpr (detail::is_matrix_v<T>) {
        return this->template sqrt_impl<ResultType>();
    } else {
        Matrix<ResultType> A = this->template cast_to<ResultType>();

        if (rows_ == 1) {
            Matrix<ResultType> result(1, 1);
            result(0, 0) = detail::sqrt_impl(A(0, 0));
            return result;
        }

        if (rows_ == 2) {
            return A.template sqrt_2x2_impl<ResultType>();
        }

        return A.template sqrt_newton_impl<ResultType>();
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::sqrt_2x2_impl() const {
    ResultType a = static_cast<ResultType>((*this)(0, 0));
    ResultType b = static_cast<ResultType>((*this)(0, 1));
    ResultType c = static_cast<ResultType>((*this)(1, 0));
    ResultType d = static_cast<ResultType>((*this)(1, 1));

    if (is_zero(b) && is_zero(c)) {
        Matrix<ResultType> result(2, 2);
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
        return result;
    }

    ResultType det = a * d - b * c;
    ResultType tr = a + d;

    auto four = Matrix<ResultType>::create_scalar(a, 4);
    auto two = Matrix<ResultType>::create_scalar(a, 2);

    ResultType delta = tr * tr - four * det;
    ResultType sqrt_delta;

    if constexpr (detail::is_builtin_integral_v<ResultType>) {
        sqrt_delta = static_cast<ResultType>(std::sqrt(static_cast<double>(delta)));
    } else if constexpr (detail::is_matrix_v<ResultType>) {
        sqrt_delta = delta.sqrt();
    } else {
        using std::sqrt;
        sqrt_delta = sqrt(delta);
    }

    ResultType s_plus;
    ResultType s_minus;

    if constexpr (detail::is_builtin_integral_v<ResultType>) {
        s_plus = static_cast<ResultType>(
            std::sqrt(static_cast<double>((tr + sqrt_delta) / two)));
        s_minus = static_cast<ResultType>(
            std::sqrt(static_cast<double>((tr - sqrt_delta) / two)));
    } else if constexpr (detail::is_matrix_v<ResultType>) {
        s_plus = ((tr + sqrt_delta) / two).sqrt();
        s_minus = ((tr - sqrt_delta) / two).sqrt();
    } else {
        using std::sqrt;
        s_plus = sqrt((tr + sqrt_delta) / two);
        s_minus = sqrt((tr - sqrt_delta) / two);
    }

    ResultType denominator = s_plus + s_minus;

    Matrix<ResultType> result(2, 2);

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

    return result;
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::sqrt_newton_impl(int max_iter,
                                               ResultType tolerance) const {
    int n = rows_;
    Matrix<ResultType> X = this->template cast_to<ResultType>();
    Matrix<ResultType> A = X;

    int block_rows = 1;
    int block_cols = 1;

    if constexpr (detail::is_matrix_v<ResultType>) {
        if (n > 0 && X(0, 0).get_rows() > 0) {
            block_rows = X(0, 0).get_rows();
            block_cols = X(0, 0).get_cols();
        }

        X = Matrix<ResultType>::BlockIdentity(n, n, block_rows, block_cols);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    X(i, j) = ResultType::Zero(block_rows, block_cols);
                }
            }
        }
    } else {
        X = Matrix<ResultType>::Identity(n);
    }

    using NormType = typename Matrix<ResultType>::norm_return_type;
    using RealType = real_type_from_norm_t<NormType>;

    RealType tol_norm;
    if constexpr (detail::is_matrix_v<ResultType>) {
        tol_norm = static_cast<RealType>(tolerance.frobenius_norm());
    } else if constexpr (detail::is_complex_v<ResultType>) {
        using std::abs;
        tol_norm = static_cast<RealType>(abs(tolerance));
    } else {
        tol_norm = static_cast<RealType>(tolerance);
    }

    Matrix<ResultType> best_X = X;
    RealType best_residual = std::numeric_limits<RealType>::max();

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

    bool any_successful_update = false;
    std::string last_error_msg;

    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix<ResultType> X_prev = X;
        bool iteration_failed = false;
        std::string iteration_error;

        try {
            Matrix<ResultType> X_inv;

            try {
                X_inv = X.inverse();
            } catch (const std::exception &e) {
                iteration_error = e.what();

                if constexpr (detail::is_matrix_v<ResultType>) {
                    using ElemType = typename ResultType::value_type;
                    ElemType example_elem;
                    if (X.get_rows() > 0 && X.get_cols() > 0) {
                        example_elem = static_cast<ElemType>(X(0, 0)(0, 0));
                    } else {
                        example_elem = ElemType(0);
                    }
                    auto shift_scalar =
                        Matrix<ElemType>::create_scalar(example_elem, 1e-6);
                    ResultType shift =
                        ResultType::Identity(block_rows, block_cols) * shift_scalar;

                    for (int i = 0; i < n; ++i) {
                        X(i, i) = X(i, i) + shift;
                    }

                    try {
                        X_inv = X.inverse();
                    } catch (const std::exception &e2) {
                        iteration_failed = true;
                        last_error_msg =
                            std::string("Failed to invert even with shift: ")
                            + e2.what();
                    } catch (...) {
                        iteration_failed = true;
                        last_error_msg =
                            "Failed to invert even with shift: unknown error";
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        X(i, i) = X(i, i) + static_cast<ResultType>(1e-6);
                    }

                    try {
                        X_inv = X.inverse();
                    } catch (const std::exception &e2) {
                        iteration_failed = true;
                        last_error_msg =
                            std::string("Failed to invert even with shift: ")
                            + e2.what();
                    } catch (...) {
                        iteration_failed = true;
                        last_error_msg =
                            "Failed to invert even with shift: unknown error";
                    }
                }
            }

            if (!iteration_failed) {
                if constexpr (detail::is_matrix_v<ResultType>) {
                    using ElemType = typename ResultType::value_type;
                    ElemType example_elem;
                    if (X.get_rows() > 0 && X.get_cols() > 0) {
                        example_elem = static_cast<ElemType>(X(0, 0)(0, 0));
                    } else {
                        example_elem = ElemType(0);
                    }
                    auto half_scalar =
                        Matrix<ElemType>::create_scalar(example_elem, 0.5);

                    Matrix<ResultType> temp = X_inv * A;
                    Matrix<ResultType> X_new(n, n);

                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < n; ++j) {
                            X_new(i, j) = (X(i, j) + temp(i, j)) * half_scalar;
                        }
                    }
                    X = X_new;
                    any_successful_update = true;
                } else {
                    auto half = Matrix<ResultType>::create_scalar(ResultType{}, 0.5);
                    X = (X + X_inv * A) * half;
                    any_successful_update = true;
                }
            }
        } catch (const std::exception &e) {
            iteration_failed = true;
            last_error_msg = e.what();
        } catch (...) {
            iteration_failed = true;
            last_error_msg = "Unknown error in Newton iteration";
        }

        if (!iteration_failed) {
            try {
                NormType norm_error = (X - X_prev).frobenius_norm();
                NormType norm_residual = (X * X - A).frobenius_norm();

                RealType error;
                RealType residual;

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
            }
        }

        if (iteration_failed && iter > 0 && any_successful_update) {
            std::cerr << "Warning: Newton iteration failed at iteration " << iter << ": "
                      << last_error_msg << "\n";
            std::cerr << "Returning best approximation found so far (residual: "
                      << best_residual << ")\n";
            return best_X;
        }
    }

    if (!any_successful_update) {
        throw std::runtime_error("Failed to perform any Newton iteration: "
                                 + last_error_msg);
    }

    return best_X;
}

template<typename T> bool Matrix<T>::has_square_root() const {
    if (rows_ != cols_)
        return false;

    if constexpr (detail::is_builtin_integral_v<T>) {
        using ResultType = sqrt_return_type<T>;
        Matrix<ResultType> A = this->template cast_to<ResultType>();
        return A.template has_square_root_impl<ResultType>();
    }

    return this->template has_square_root_impl<T>();
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::has_square_root_impl() const {
    if (rows_ == 1) {
        if constexpr (std::is_arithmetic_v<ResultType>
                      && !detail::is_complex_v<ResultType>) {
            ResultType val = static_cast<ResultType>((*this)(0, 0));
            if constexpr (detail::is_matrix_v<ResultType>) {
                return true;
            } else {
                return val >= ResultType(0);
            }
        }
        return true;
    }

    if (rows_ == 2) {
        if constexpr (std::is_arithmetic_v<ResultType>
                      && !detail::is_complex_v<ResultType>) {
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

            ResultType sqrt_discr = std::sqrt(discr);
            ResultType lambda1 = (tr + sqrt_discr) / ResultType(2);
            ResultType lambda2 = (tr - sqrt_discr) / ResultType(2);

            return lambda1 >= ResultType(0) && lambda2 >= ResultType(0);
        }

        return true;
    }

    constexpr bool should_check_directly = std::is_arithmetic_v<ResultType>
                                           && !detail::is_complex_v<ResultType>
                                           && !detail::is_matrix_v<ResultType>;

    if constexpr (should_check_directly) {
        bool symmetric = true;
        for (int i = 0; i < rows_ && symmetric; ++i) {
            for (int j = i + 1; j < cols_ && symmetric; ++j) {
                if (std::abs(static_cast<ResultType>((*this)(i, j))
                             - static_cast<ResultType>((*this)(j, i)))
                    > ResultType(1e-6)) {
                    symmetric = false;
                }
            }
        }

        bool positive_diag = true;
        for (int i = 0; i < rows_; ++i) {
            if (static_cast<ResultType>((*this)(i, i)) <= ResultType(0)) {
                positive_diag = false;
                break;
            }
        }

        if (symmetric && positive_diag) {
            return true;
        }

        try {
            return this->template has_square_root_direct_impl<ResultType>();
        } catch (...) {
            return false;
        }
    } else {
        return true;
    }
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::has_square_root_direct_impl() const {
    for (int i = 0; i < rows_; ++i) {
        ResultType diag = static_cast<ResultType>((*this)(i, i));
        if (diag < ResultType(0)) {
            return false;
        }
    }

    try {
        Matrix<ResultType> test_sqrt = this->template sqrt_impl<ResultType>();
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
template<typename ResultType>
Matrix<ResultType> Matrix<T>::sqrt_impl() const {
    if (rows_ == 1) {
        Matrix<ResultType> result(1, 1);
        if constexpr (detail::is_builtin_integral_v<ResultType>) {
            result(0, 0) = static_cast<ResultType>(
                std::sqrt(static_cast<double>(static_cast<ResultType>((*this)(0, 0)))));
        } else if constexpr (detail::is_matrix_v<ResultType>) {
            result(0, 0) = static_cast<ResultType>((*this)(0, 0)).sqrt();
        } else {
            using std::sqrt;
            result(0, 0) = sqrt(static_cast<ResultType>((*this)(0, 0)));
        }
        return result;
    }

    if (rows_ == 2) {
        return this->template sqrt_2x2_impl<ResultType>();
    }

    if constexpr (detail::is_matrix_v<ResultType>) {
        int block_rows = 1;
        int block_cols = 1;

        if (rows_ > 0 && cols_ > 0) {
            block_rows = static_cast<ResultType>((*this)(0, 0)).get_rows();
            block_cols = static_cast<ResultType>((*this)(0, 0)).get_cols();
            if (block_rows == 0)
                block_rows = 1;
            if (block_cols == 0)
                block_cols = 1;
        }

        Matrix<ResultType> result(rows_, cols_);

        bool is_block_diagonal = true;
        for (int i = 0; i < rows_ && is_block_diagonal; ++i) {
            for (int j = 0; j < cols_ && is_block_diagonal; ++j) {
                if (i != j && !is_zero((*this)(i, j))) {
                    is_block_diagonal = false;
                }
            }
        }

        if (is_block_diagonal) {
            for (int i = 0; i < rows_; ++i) {
                result(i, i) = static_cast<ResultType>((*this)(i, i)).sqrt();
                for (int j = 0; j < cols_; ++j) {
                    if (i != j) {
                        result(i, j) = ResultType::Zero(block_rows, block_cols);
                    }
                }
            }
        } else {
            using ElemType = typename ResultType::value_type;
            ElemType example_elem;
            if (rows_ > 0 && cols_ > 0) {
                example_elem = static_cast<ElemType>((*this)(0, 0)(0, 0));
            } else {
                example_elem = ElemType(0);
            }
            auto tolerance_scalar = Matrix<ElemType>::create_scalar(example_elem, 1e-10);
            ResultType tolerance =
                ResultType::Identity(block_rows, block_cols) * tolerance_scalar;
            return this->template sqrt_newton_impl<ResultType>(50, tolerance);
        }

        return result;
    } else {
        auto tolerance_value = Matrix<ResultType>::create_scalar(ResultType{}, 1e-10);
        return this->template sqrt_newton_impl<ResultType>(100, tolerance_value);
    }
}

template<typename T>
std::pair<Matrix<typename Matrix<T>::template sqrt_return_type<T>>, bool>
Matrix<T>::safe_sqrt() const {
    using ResultType = typename Matrix<T>::template sqrt_return_type<T>;

    if (rows_ != cols_) {
        return {Matrix<ResultType>(), false};
    }

    try {
        Matrix<ResultType> result = this->sqrt();
        return {result, true};
    } catch (...) {
        Matrix<ResultType> approx(rows_, cols_);

        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (i == j) {
                    try {
                        ResultType val = static_cast<ResultType>((*this)(i, i));

                        if constexpr (detail::is_matrix_v<ResultType>) {
                            auto [block_sqrt, block_success] = val.safe_sqrt();
                            approx(i, i) = block_sqrt;
                        } else {
                            using std::abs;
                            using std::sqrt;

                            if constexpr (std::is_arithmetic_v<ResultType>
                                          && !detail::is_complex_v<ResultType>) {
                                if (val >= ResultType(0)) {
                                    approx(i, i) = sqrt(val);
                                } else {
                                    approx(i, i) = sqrt(abs(val));
                                }
                            } else if constexpr (detail::is_complex_v<ResultType>) {
                                approx(i, i) = sqrt(val);
                            } else {
                                approx(i, i) = sqrt(val);
                            }
                        }
                    } catch (...) {
                        if constexpr (detail::is_matrix_v<ResultType>) {
                            int block_rows = 1, block_cols = 1;
                            if (rows_ > 0 && cols_ > 0) {
                                try {
                                    auto first_val =
                                        static_cast<ResultType>((*this)(0, 0));
                                    block_rows = first_val.get_rows();
                                    block_cols = first_val.get_cols();
                                } catch (...) {
                                }
                            }
                            approx(i, i) = ResultType::Identity(block_rows, block_cols);
                        } else {
                            approx(i, i) = ResultType(1);
                        }
                    }
                } else {
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
                        approx(i, j) = ResultType::Zero(block_rows, block_cols);
                    } else {
                        approx(i, j) = ResultType(0);
                    }
                }
            }
        }

        try {
            bool has_nan = false;
            for (int i = 0; i < rows_ && !has_nan; ++i) {
                for (int j = 0; j < cols_ && !has_nan; ++j) {
                    if constexpr (detail::is_matrix_v<ResultType>) {
                        auto &block = approx(i, j);
                        for (int bi = 0; bi < block.get_rows() && !has_nan; ++bi) {
                            for (int bj = 0; bj < block.get_cols() && !has_nan; ++bj) {
                                if (std::isnan(static_cast<double>(block(bi, bj)))) {
                                    has_nan = true;
                                }
                            }
                        }
                    } else if constexpr (std::is_arithmetic_v<ResultType>) {
                        if (std::isnan(static_cast<double>(approx(i, j)))) {
                            has_nan = true;
                        }
                    }
                }
            }

            if (has_nan) {
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
                    approx = Matrix<ResultType>::BlockIdentity(rows_,
                                                               cols_,
                                                               block_rows,
                                                               block_cols);
                } else {
                    approx = Matrix<ResultType>::Identity(rows_);
                }
            }
        } catch (...) {
        }

        return {approx, false};
    }
}
