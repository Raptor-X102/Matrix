#include "Vector.hpp"

template<typename T>
template<typename ComputeType>
Matrix<ComputeType> Matrix<T>::balance_matrix() const {
    if (min_dim_ == 0) {
        return Matrix<ComputeType>();
    }

    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square for balancing");
    }

    auto A = this->template cast_to<ComputeType>();
    int n = rows_;

    if constexpr (!detail::is_matrix_v<ComputeType>) {
        if constexpr (detail::is_complex_v<ComputeType>
                      || std::is_floating_point_v<ComputeType>) {
            using RealType = std::conditional_t<detail::is_complex_v<ComputeType>,
                                                typename ComputeType::value_type,
                                                ComputeType>;

            const RealType radix = RealType(2);
            const RealType sqrdx = radix * radix;
            const RealType balance_threshold = RealType(0.95);

            bool converged = false;
            int max_iterations = 100;
            int iteration = 0;

            while (!converged && iteration < max_iterations) {
                converged = true;
                iteration++;

                for (int i = 0; i < n; ++i) {
                    RealType row_norm = RealType{0};
                    RealType col_norm = RealType{0};

                    for (int j = 0; j < n; ++j) {
                        if (j != i) {
                            try {
                                using std::abs;
                                row_norm += abs(A(i, j));
                                col_norm += abs(A(j, i));
                            } catch (...) {
                                throw std::runtime_error(
                                    "Failed to compute norms in balancing");
                            }
                        }
                    }

                    if (row_norm > RealType{0} && col_norm > RealType{0}) {
                        RealType g = row_norm / radix;
                        RealType f = RealType{1};
                        RealType s = col_norm + row_norm;

                        while (col_norm < g) {
                            f *= radix;
                            col_norm *= sqrdx;
                        }

                        g = row_norm * radix;

                        while (col_norm > g) {
                            f /= radix;
                            col_norm /= sqrdx;
                        }

                        if ((row_norm + col_norm) < balance_threshold * s * f) {
                            converged = false;

                            for (int j = 0; j < n; ++j) {
                                if (j != i) {
                                    try {
                                        A(i, j) *= static_cast<ComputeType>(f);
                                        A(j, i) /= static_cast<ComputeType>(f);
                                    } catch (...) {
                                        throw std::runtime_error(
                                            "Failed to apply balancing transformation");
                                    }
                                }
                            }
                            try {
                                A(i, i) *= static_cast<ComputeType>(f);
                            } catch (...) {
                                throw std::runtime_error(
                                    "Failed to scale diagonal element");
                            }
                        }
                    }
                }
            }

            if (!converged) {
                std::cerr << "Warning: Balancing did not converge after "
                          << max_iterations << " iterations\n";
            }
        } else if constexpr (std::is_integral_v<ComputeType>) {
            // Nothing to do for integral types
        }
    }

    return A;
}

template<typename T>
template<typename ComputeType>
std::pair<Matrix<ComputeType>, Matrix<ComputeType>> Matrix<T>::qr_decomposition() const {
    if (min_dim_ == 0) {
        return {Matrix<ComputeType>(), Matrix<ComputeType>()};
    }

    int m = rows_;
    int n = cols_;

    Matrix<ComputeType> Q;
    Matrix<ComputeType> R;

    try {
        R = this->template cast_to<ComputeType>();
    } catch (...) {
        throw std::runtime_error(
            "Failed to cast matrix to ComputeType for QR decomposition");
    }

    if constexpr (detail::is_matrix_v<ComputeType>) {
        int block_rows = 1;
        int block_cols = 1;
        if (m > 0 && n > 0) {
            try {
                block_rows = R(0, 0).get_rows();
                block_cols = R(0, 0).get_cols();
            } catch (...) {
                block_rows = 1;
                block_cols = 1;
            }
        }
        try {
            Q = Matrix<ComputeType>::BlockIdentity(m, m, block_rows, block_cols);
        } catch (...) {
            throw std::runtime_error("Failed to create identity matrix for Q");
        }
    } else {
        try {
            Q = Matrix<ComputeType>::Identity(m);
        } catch (...) {
            throw std::runtime_error("Failed to create identity matrix for Q");
        }
    }

    for (int k = 0; k < std::min(m - 1, n); ++k) {
        Vector<ComputeType> x;
        try {
            x = Vector<ComputeType>(m - k);
            for (int i = k; i < m; ++i) {
                x[i - k] = R(i, k);
            }
        } catch (...) {
            throw std::runtime_error(
                "Failed to create column vector for Householder transformation");
        }

        auto v = householder_vector(x);
        auto v_norm = v.norm();

        if (is_norm_zero(v_norm))
            continue;

        ComputeType two_scalar;
        try {
            if constexpr (detail::is_matrix_v<ComputeType>) {
                using InnerType = typename ComputeType::value_type;
                auto block = R(0, 0);
                InnerType inner_two = create_scalar(block(0, 0), InnerType(2));
                two_scalar =
                    ComputeType::Diagonal(block.get_rows(), block.get_cols(), inner_two);
            } else if constexpr (detail::is_complex_v<ComputeType>) {
                using RealType = typename ComputeType::value_type;
                two_scalar = ComputeType(RealType(2), RealType(0));
            } else {
                two_scalar = ComputeType(2);
            }
        } catch (...) {
            throw std::runtime_error(
                "Failed to create scalar 2 for Householder transformation");
        }

        // Apply Householder transformation to R
        for (int j = k; j < n; ++j) {
            Vector<ComputeType> col;
            try {
                col = Vector<ComputeType>(m - k);
                for (int i = k; i < m; ++i) {
                    col[i - k] = R(i, j);
                }
            } catch (...) {
                throw std::runtime_error(
                    "Failed to extract column for Householder transformation");
            }

            ComputeType dot;
            try {
                dot = col.dot(v);
            } catch (...) {
                throw std::runtime_error(
                    "Failed to compute dot product for Householder transformation");
            }

            for (int i = k; i < m; ++i) {
                try {
                    R(i, j) = R(i, j) - v[i - k] * dot * two_scalar;
                } catch (...) {
                    throw std::runtime_error(
                        "Failed to update R matrix in Householder transformation");
                }
            }
        }

        // Apply Householder transformation to Q
        for (int j = 0; j < m; ++j) {
            Vector<ComputeType> col;
            try {
                col = Vector<ComputeType>(m - k);
                for (int i = k; i < m; ++i) {
                    col[i - k] = Q(i, j);
                }
            } catch (...) {
                throw std::runtime_error(
                    "Failed to extract column of Q for Householder transformation");
            }

            ComputeType dot;
            try {
                dot = col.dot(v);
            } catch (...) {
                throw std::runtime_error("Failed to compute dot product for Q update");
            }

            for (int i = k; i < m; ++i) {
                try {
                    Q(i, j) = Q(i, j) - v[i - k] * dot * two_scalar;
                } catch (...) {
                    throw std::runtime_error("Failed to update Q matrix");
                }
            }
        }
    }

    try {
        return {Q.transpose(), R};
    } catch (...) {
        throw std::runtime_error("Failed to transpose Q matrix");
    }
}

template<typename T>
template<typename ComputeType>
Vector<ComputeType> Matrix<T>::householder_vector(const Vector<ComputeType> &x) const {
    int m = x.size();

    if (m == 0) {
        return Vector<ComputeType>(0);
    }

    auto norm_x = x.norm();

    if (is_norm_zero(norm_x)) {
        Vector<ComputeType> zero_vec;
        if constexpr (detail::is_matrix_v<ComputeType>) {
            if (m > 0) {
                auto block = x[0];
                int block_rows = block.get_rows();
                int block_cols = block.get_cols();
                ComputeType zero_block;
                try {
                    zero_block = ComputeType::Zero(block_rows, block_cols);
                } catch (...) {
                    throw std::runtime_error(
                        "Failed to create zero block for householder vector");
                }
                zero_vec = Vector<ComputeType>(m, zero_block);
            } else {
                zero_vec = Vector<ComputeType>(0);
            }
        } else {
            auto zero_scalar = create_scalar(x[0], 0);
            zero_vec = Vector<ComputeType>(m, zero_scalar);
        }
        return zero_vec;
    }

    Vector<ComputeType> e1;

    if constexpr (detail::is_matrix_v<ComputeType>) {
        if (m > 0) {
            auto block = x[0];
            int block_rows = block.get_rows();
            int block_cols = block.get_cols();

            ComputeType zero_block;
            try {
                zero_block = ComputeType::Zero(block_rows, block_cols);
            } catch (...) {
                throw std::runtime_error("Failed to create zero block for e1");
            }
            e1 = Vector<ComputeType>(m, zero_block);

            ComputeType scalar_norm_block;
            try {
                scalar_norm_block =
                    ComputeType::Diagonal(block_rows,
                                          block_cols,
                                          create_scalar(block(0, 0), norm_x));
            } catch (...) {
                throw std::runtime_error("Failed to create norm block for e1");
            }
            e1[0] = scalar_norm_block;
        } else {
            e1 = Vector<ComputeType>(0);
        }
    } else {
        auto zero_scalar = create_scalar(x[0], 0);
        e1 = Vector<ComputeType>(m, zero_scalar);
        e1[0] = norm_x;
    }

    auto v = x + e1;
    auto norm_v = v.norm();

    if (is_norm_zero(norm_v)) {
        Vector<ComputeType> zero_vec;
        if constexpr (detail::is_matrix_v<ComputeType>) {
            if (m > 0) {
                auto block = x[0];
                int block_rows = block.get_rows();
                int block_cols = block.get_cols();
                ComputeType zero_block = ComputeType::Zero(block_rows, block_cols);
                zero_vec = Vector<ComputeType>(m, zero_block);
            } else {
                zero_vec = Vector<ComputeType>(0);
            }
        } else {
            auto zero_scalar = create_scalar(x[0], 0);
            zero_vec = Vector<ComputeType>(m, zero_scalar);
        }
        return zero_vec;
    }

    if constexpr (detail::is_matrix_v<ComputeType>) {
        using InnerType = typename ComputeType::value_type;
        InnerType norm_v_scalar;
        try {
            norm_v_scalar = create_scalar(v[0](0, 0), norm_v);
        } catch (...) {
            throw std::runtime_error("Failed to create norm scalar for block matrix");
        }

        for (int i = 0; i < m; ++i) {
            try {
                v[i] = v[i] / norm_v_scalar;
            } catch (...) {
                throw std::runtime_error("Failed to normalize block matrix element");
            }
        }
    } else {
        try {
            v = v / norm_v;
        } catch (...) {
            throw std::runtime_error("Failed to normalize vector");
        }
    }

    return v;
}

template<typename T>
template<typename ComputeType>
Matrix<ComputeType> Matrix<T>::hessenberg_form() const {
    if (min_dim_ == 0) {
        return Matrix<ComputeType>();
    }

    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square for Hessenberg form");
    }

    auto H = this->template cast_to<ComputeType>();
    int n = rows_;

    DEBUG_PRINTF("Hessenberg form: n=%d\n", n);

    if constexpr (detail::is_matrix_v<ComputeType>) {
        if (n > 0) {
            try {
                DEBUG_PRINTF("H block size: %dx%d\n",
                             H(0, 0).get_rows(),
                             H(0, 0).get_cols());
            } catch (...) {
                DEBUG_PRINTF("H block size: unknown\n");
            }
        }
    }

    for (int k = 0; k < n - 2; ++k) {
        Vector<ComputeType> x;
        try {
            x = Vector<ComputeType>(n - k - 1);
            for (int i = k + 1; i < n; ++i) {
                x[i - k - 1] = H(i, k);
            }
        } catch (...) {
            throw std::runtime_error("Failed to create vector for Hessenberg reduction");
        }

        auto v = householder_vector(x);
        auto v_norm = v.norm();

        if (is_norm_zero(v_norm))
            continue;

        try {
            apply_householder_left(H, v, k + 1);
            apply_householder_right(H, v, k + 1);
        } catch (const std::exception &e) {
            throw std::runtime_error(
                "Failed to apply Householder transformation in Hessenberg reduction: "
                + std::string(e.what()));
        } catch (...) {
            throw std::runtime_error(
                "Failed to apply Householder transformation in Hessenberg reduction");
        }
    }

    return H;
}

template<typename T>
template<typename ComputeType>
void Matrix<T>::apply_householder_left(Matrix<ComputeType> &A,
                                       const Vector<ComputeType> &v,
                                       int k) const {
    int n = A.get_rows();
    int m = A.get_cols();

    if (k < 0 || k >= n) {
        throw std::out_of_range("Index k out of range in apply_householder_left");
    }

    auto two_scalar = Matrix<ComputeType>::create_scalar(A(0, 0), 2.0);

    for (int j = k; j < m; ++j) {
        Vector<ComputeType> col;
        try {
            col = Vector<ComputeType>(n - k);
            for (int i = k; i < n; ++i) {
                col[i - k] = A(i, j);
            }
        } catch (...) {
            throw std::runtime_error(
                "Failed to extract column for left Householder transformation");
        }

        ComputeType dot;
        if constexpr (detail::is_complex_v<ComputeType>) {
            dot = ComputeType(0);
            for (int i = 0; i < n - k; ++i) {
                try {
                    dot += std::conj(v[i]) * col[i];
                } catch (...) {
                    throw std::runtime_error(
                        "Failed to compute complex conjugate dot product");
                }
            }
        } else {
            try {
                dot = col.dot(v);
            } catch (...) {
                throw std::runtime_error(
                    "Failed to compute dot product for left Householder transformation");
            }
        }

        ComputeType scale_factor;
        if constexpr (detail::is_matrix_v<ComputeType>) {
            try {
                scale_factor = two_scalar * dot;
            } catch (...) {
                throw std::runtime_error(
                    "Failed to compute scale factor for block matrix");
            }
        } else {
            scale_factor = dot * two_scalar;
        }

        for (int i = k; i < n; ++i) {
            try {
                A(i, j) = A(i, j) - v[i - k] * scale_factor;
            } catch (...) {
                throw std::runtime_error(
                    "Failed to update matrix element in left Householder transformation");
            }
        }
    }
}

template<typename T>
template<typename ComputeType>
void Matrix<T>::apply_householder_right(Matrix<ComputeType> &A,
                                        const Vector<ComputeType> &v,
                                        int k) const {
    int n = A.get_rows();
    int m = A.get_cols();

    if (k < 0 || k >= m) {
        throw std::out_of_range("Index k out of range in apply_householder_right");
    }

    auto two_scalar = Matrix<ComputeType>::create_scalar(A(0, 0), 2.0);

    for (int i = 0; i < n; ++i) {
        Vector<ComputeType> row;
        try {
            row = Vector<ComputeType>(m - k);
            for (int j = k; j < m; ++j) {
                row[j - k] = A(i, j);
            }
        } catch (...) {
            throw std::runtime_error(
                "Failed to extract row for right Householder transformation");
        }

        ComputeType dot;
        if constexpr (detail::is_complex_v<ComputeType>) {
            dot = ComputeType(0);
            for (int idx = 0; idx < m - k; ++idx) {
                try {
                    dot += std::conj(v[idx]) * row[idx];
                } catch (...) {
                    throw std::runtime_error(
                        "Failed to compute complex conjugate dot product");
                }
            }
        } else {
            try {
                dot = row.dot(v);
            } catch (...) {
                throw std::runtime_error(
                    "Failed to compute dot product for right Householder transformation");
            }
        }

        ComputeType scale_factor = dot * two_scalar;

        for (int j = k; j < m; ++j) {
            try {
                A(i, j) = A(i, j) - v[j - k] * scale_factor;
            } catch (...) {
                throw std::runtime_error(
                    "Failed to update matrix element in right Householder transformation");
            }
        }
    }
}

template<typename T>
template<typename ComputeType>
Vector<ComputeType> Matrix<T>::back_substitution(const Matrix<ComputeType> &R,
                                                 const Vector<ComputeType> &y) const {
    int n = R.get_rows();

    if (n == 0) {
        return Vector<ComputeType>(0);
    }

    if (R.get_cols() != n) {
        throw std::invalid_argument("R must be square for back substitution");
    }

    if (y.size() != n) {
        throw std::invalid_argument("Vector y must have same size as R");
    }

    Vector<ComputeType> x;

    if constexpr (detail::is_matrix_v<ComputeType>) {
        if (n > 0) {
            int block_rows = 1;
            int block_cols = 1;
            try {
                if (R(0, 0).get_rows() > 0 && R(0, 0).get_cols() > 0) {
                    block_rows = R(0, 0).get_rows();
                    block_cols = R(0, 0).get_cols();
                }
            } catch (...) {
                block_rows = 1;
                block_cols = 1;
            }
            ComputeType zero_block;
            try {
                zero_block = ComputeType::Zero(block_rows, block_cols);
            } catch (...) {
                throw std::runtime_error(
                    "Failed to create zero block for back substitution");
            }
            x = Vector<ComputeType>(n, zero_block);
        } else {
            x = Vector<ComputeType>(0);
        }
    } else {
        x = Vector<ComputeType>(n);
    }

    for (int i = n - 1; i >= 0; --i) {
        ComputeType sum;
        if constexpr (detail::is_matrix_v<ComputeType>) {
            int block_rows = 1;
            int block_cols = 1;
            if (n > 0) {
                try {
                    if (R(0, 0).get_rows() > 0 && R(0, 0).get_cols() > 0) {
                        block_rows = R(0, 0).get_rows();
                        block_cols = R(0, 0).get_cols();
                    }
                } catch (...) {
                    block_rows = 1;
                    block_cols = 1;
                }
            }
            try {
                sum = ComputeType::Zero(block_rows, block_cols);
            } catch (...) {
                throw std::runtime_error("Failed to create zero block for sum");
            }
        } else {
            sum = ComputeType{0};
        }

        for (int j = i + 1; j < n; ++j) {
            try {
                sum = sum + R(i, j) * x[j];
            } catch (...) {
                throw std::runtime_error("Failed to compute sum in back substitution");
            }
        }
        try {
            x[i] = (y[i] - sum) / R(i, i);
        } catch (...) {
            throw std::runtime_error("Failed to compute x[i] in back substitution");
        }
    }

    return x;
}

template<typename T>
template<typename ComputeType>
Vector<ComputeType> Matrix<T>::inverse_iteration(const Matrix<ComputeType> &A,
                                                 const ComputeType &lambda,
                                                 int max_iterations) const {
    if (max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }

    int n = A.get_rows();

    if (n == 0) {
        return Vector<ComputeType>(0);
    }

    if (A.get_cols() != n) {
        throw std::invalid_argument("A must be square for inverse iteration");
    }

    Matrix<ComputeType> I;
    if constexpr (detail::is_matrix_v<ComputeType>) {
        int block_rows = 1;
        int block_cols = 1;
        if (n > 0) {
            try {
                if (A(0, 0).get_rows() > 0 && A(0, 0).get_cols() > 0) {
                    block_rows = A(0, 0).get_rows();
                    block_cols = A(0, 0).get_cols();
                }
            } catch (...) {
                block_rows = 1;
                block_cols = 1;
            }
        }
        try {
            I = Matrix<ComputeType>::BlockIdentity(n, n, block_rows, block_cols);
        } catch (...) {
            throw std::runtime_error(
                "Failed to create identity matrix for inverse iteration");
        }
    } else {
        try {
            I = Matrix<ComputeType>::Identity(n);
        } catch (...) {
            throw std::runtime_error(
                "Failed to create identity matrix for inverse iteration");
        }
    }

    auto lambda_I = I * lambda;
    auto B = A - lambda_I;

    Vector<ComputeType> x;
    try {
        x = Vector<ComputeType>::random(n);
    } catch (...) {
        throw std::runtime_error("Failed to create random vector for inverse iteration");
    }

    auto x_norm = x.norm();
    try {
        x = x / x_norm;
    } catch (...) {
        throw std::runtime_error("Failed to normalize vector for inverse iteration");
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        try {
            auto y_vec = B.inverse() * x;
            auto y_norm = y_vec.norm();
            x = y_vec / y_norm;
        } catch (...) {
            break;
        }
    }

    return x;
}

template<typename T>
template<typename ComputeType>
std::vector<ComputeType> Matrix<T>::extract_eigenvalues_2x2(const Matrix<ComputeType> &H,
                                                            int i) const {
    int n = H.get_rows();

    if (n == 0) {
        return {};
    }

    if (i < 0 || i + 1 >= n) {
        throw std::out_of_range("Index i out of range in extract_eigenvalues_2x2");
    }

    std::vector<ComputeType> eigenvalues;

    if constexpr (detail::is_matrix_v<ComputeType>) {
        using InnerType = typename ComputeType::value_type;

        auto a = H(i, i);
        auto b = H(i, i + 1);
        auto c = H(i + 1, i);
        auto d = H(i + 1, i + 1);

        if constexpr (detail::is_matrix_v<InnerType>) {
            eigenvalues.push_back(H(i, i));
            eigenvalues.push_back(H(i + 1, i + 1));
        } else {
            auto trace = a + d;
            auto det_val = a * d - b * c;

            try {
                auto I = ComputeType::Identity(a.get_rows(), a.get_cols());
                auto scalar_two = create_scalar(a(0, 0), 2);
                auto discriminant = trace * trace - scalar_two * scalar_two * det_val;

                if constexpr (detail::has_sqrt_v<decltype(discriminant)>) {
                    auto sqrt_disc = discriminant.sqrt();
                    eigenvalues.push_back((trace + sqrt_disc)
                                          / (scalar_two * scalar_two));
                    eigenvalues.push_back((trace - sqrt_disc)
                                          / (scalar_two * scalar_two));
                } else {
                    eigenvalues.push_back(H(i, i));
                    eigenvalues.push_back(H(i + 1, i + 1));
                }
            } catch (...) {
                eigenvalues.push_back(H(i, i));
                eigenvalues.push_back(H(i + 1, i + 1));
            }
        }
    } else {
        auto a = H(i, i);
        auto b = H(i, i + 1);
        auto c = H(i + 1, i);
        auto d = H(i + 1, i + 1);

        auto trace = a + d;
        auto det_val = a * d - b * c;

        using std::sqrt;

        if constexpr (detail::is_complex_v<ComputeType>) {
            using RealType = typename ComputeType::value_type;
            auto discriminant = trace * trace - RealType(4) * det_val;
            auto sqrt_disc = sqrt(discriminant);
            eigenvalues.push_back((trace + sqrt_disc) / RealType(2));
            eigenvalues.push_back((trace - sqrt_disc) / RealType(2));
        } else if constexpr (std::is_floating_point_v<ComputeType>) {
            auto discriminant = trace * trace - ComputeType(4) * det_val;

            using ComplexType = std::complex<ComputeType>;
            auto sqrt_disc = sqrt(ComplexType(discriminant));
            eigenvalues.push_back(
                static_cast<ComputeType>((trace + sqrt_disc) / ComputeType(2)));
            eigenvalues.push_back(
                static_cast<ComputeType>((trace - sqrt_disc) / ComputeType(2)));
        } else if constexpr (std::is_integral_v<ComputeType>) {
            eigenvalues.push_back(H(i, i));
            eigenvalues.push_back(H(i + 1, i + 1));
        } else {
            eigenvalues.push_back(H(i, i));
            eigenvalues.push_back(H(i + 1, i + 1));
        }
    }

    return eigenvalues;
}

template<typename T>
template<typename ComputeType>
Matrix<ComputeType>
Matrix<T>::eigenvectors_2x2(const Matrix<ComputeType> &A,
                            const std::vector<ComputeType> &eigenvalues) const {
    if (A.get_rows() != 2 || A.get_cols() != 2) {
        throw std::invalid_argument("eigenvectors_2x2 requires a 2x2 matrix");
    }

    if (eigenvalues.size() != 2) {
        throw std::invalid_argument("eigenvalues must contain exactly 2 values");
    }

    Matrix<ComputeType> V(2, 2);

    if constexpr (detail::is_matrix_v<ComputeType>) {
        if (A.get_rows() > 0 && A(0, 0).get_rows() > 0) {
            int block_rows = A(0, 0).get_rows();
            int block_cols = A(0, 0).get_cols();

            auto zero_block = ComputeType::Zero(block_rows, block_cols);
            auto identity_block = ComputeType::Identity(block_rows, block_cols);

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    V(i, j) = zero_block;
                }
            }

            for (size_t i = 0; i < eigenvalues.size(); ++i) {
                auto lambda = eigenvalues[i];

                auto I_mat =
                    Matrix<ComputeType>::BlockIdentity(2, 2, block_rows, block_cols);
                auto B = A - I_mat * lambda;

                auto b00 = B(0, 0);
                auto b01 = B(0, 1);
                auto b10 = B(1, 0);
                auto b11 = B(1, 1);

                if (!is_norm_zero(b01)) {
                    try {
                        V(0, i) = identity_block;
                        V(1, i) = -b00 * b01.inverse();
                    } catch (...) {
                        V(i, i) = identity_block;
                    }
                } else if (!is_norm_zero(b10)) {
                    try {
                        V(0, i) = -b11 * b10.inverse();
                        V(1, i) = identity_block;
                    } catch (...) {
                        V(i, i) = identity_block;
                    }
                } else {
                    V(i, i) = identity_block;
                }
            }
        }
    } else {
        for (size_t i = 0; i < eigenvalues.size(); ++i) {
            auto lambda = eigenvalues[i];
            auto B = A - Matrix<ComputeType>::Identity(2) * lambda;

            auto b00 = B(0, 0);
            auto b01 = B(0, 1);
            auto b10 = B(1, 0);
            auto b11 = B(1, 1);

            ComputeType v0, v1;

            if (!Matrix<ComputeType>::is_zero(b01)) {
                v0 = create_scalar(b00, 1.0);
                v1 = -b00 / b01;
            } else if (!Matrix<ComputeType>::is_zero(b10)) {
                v0 = -b11 / b10;
                v1 = create_scalar(b00, 1.0);
            } else {
                v0 = (i == 0) ? create_scalar(b00, 1.0) : create_scalar(b00, 0.0);
                v1 = (i == 1) ? create_scalar(b00, 1.0) : create_scalar(b00, 0.0);
            }

            using std::sqrt;
            auto norm = sqrt(v0 * v0 + v1 * v1);
            if (!Matrix<ComputeType>::is_zero(norm)) {
                v0 = v0 / norm;
                v1 = v1 / norm;
            }

            V(0, i) = v0;
            V(1, i) = v1;
        }
    }

    return V;
}

template<typename T>
template<typename ComputeType>
std::vector<ComputeType> Matrix<T>::eigenvalues_qr(int max_iterations) const {
    if (min_dim_ == 0) {
        return {};
    }

    if (rows_ != cols_) {
        throw std::invalid_argument("Eigenvalues require square matrix");
    }

    if (max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }

    Matrix<ComputeType> H;
    int n = rows_;

    if constexpr (!detail::is_matrix_v<ComputeType>
                  && (std::is_floating_point_v<ComputeType>
                      || detail::is_complex_v<ComputeType>)) {
        if (n > 3) {
            try {
                H = this->template balance_matrix<ComputeType>();
            } catch (...) {
                H = this->template cast_to<ComputeType>();
            }
        } else {
            H = this->template cast_to<ComputeType>();
        }
    } else {
        H = this->template cast_to<ComputeType>();
    }

    if (n == 1) {
        return {H(0, 0)};
    }

    if (n == 2) {
        return extract_eigenvalues_2x2(H, 0);
    }

    try {
        H = H.template hessenberg_form<ComputeType>();
    } catch (...) {
        throw std::runtime_error("Failed to reduce matrix to Hessenberg form");
    }

    const int adjusted_iterations = max_iterations * 2;

    for (int iter = 0; iter < adjusted_iterations; ++iter) {
        try {
            auto [Q, R] = H.template qr_decomposition<ComputeType>();
            H = R * Q;
        } catch (...) {
            break;
        }

        bool converged = true;
        for (int i = 0; i < n - 1; ++i) {
            auto off_diag = H(i, i + 1);
            if (!is_norm_zero(off_diag)) {
                converged = false;
                break;
            }
        }

        if (converged) {
            DEBUG_PRINTF("QR algorithm converged after %d iterations\n", iter + 1);
            break;
        }
    }

    std::vector<ComputeType> eigenvalues;
    int i = 0;

    while (i < n) {
        if (i == n - 1) {
            eigenvalues.push_back(H(i, i));
            i++;
        } else {
            auto off_diag = H(i, i + 1);

            bool is_zero_off_diag = is_norm_zero(off_diag);

            if (is_zero_off_diag) {
                eigenvalues.push_back(H(i, i));
                i++;
            } else {
                auto eig_2x2 = this->template extract_eigenvalues_2x2(H, i);
                eigenvalues.insert(eigenvalues.end(), eig_2x2.begin(), eig_2x2.end());
                i += 2;
            }
        }
    }

    return eigenvalues;
}

template<typename T>
template<typename ComputeType>
Matrix<ComputeType> Matrix<T>::eigenvectors_qr(int max_iterations) const {
    if (min_dim_ == 0) {
        return Matrix<ComputeType>();
    }

    if (rows_ != cols_) {
        throw std::invalid_argument("Eigenvectors require square matrix");
    }

    if (max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }

    auto A_orig = this->template cast_to<ComputeType>();
    int n = rows_;

    Matrix<ComputeType> V;

    if constexpr (detail::is_matrix_v<ComputeType>) {
        int block_rows = 1;
        int block_cols = 1;
        if (n > 0) {
            try {
                if (A_orig(0, 0).get_rows() > 0 && A_orig(0, 0).get_cols() > 0) {
                    block_rows = A_orig(0, 0).get_rows();
                    block_cols = A_orig(0, 0).get_cols();
                }
            } catch (...) {
                block_rows = 1;
                block_cols = 1;
            }
        }
        try {
            V = Matrix<ComputeType>::BlockIdentity(n, n, block_rows, block_cols);
        } catch (...) {
            throw std::runtime_error(
                "Failed to create identity matrix for eigenvectors");
        }
    } else {
        try {
            V = Matrix<ComputeType>::Identity(n);
        } catch (...) {
            throw std::runtime_error(
                "Failed to create identity matrix for eigenvectors");
        }
    }

    Matrix<ComputeType> H = A_orig;

    if (n > 2) {
        try {
            H = H.template hessenberg_form<ComputeType>();
        } catch (...) {
            throw std::runtime_error("Failed to reduce matrix to Hessenberg form");
        }
    }

    const int adjusted_iterations = max_iterations;

    for (int iter = 0; iter < adjusted_iterations; ++iter) {
        try {
            auto [Q, R] = H.template qr_decomposition<ComputeType>();
            V = V * Q;
            H = R * Q;

            bool converged = true;
            for (int i = 0; i < n - 1; ++i) {
                auto off_diag = H(i, i + 1);
                if (!is_norm_zero(off_diag)) {
                    converged = false;
                    break;
                }
            }

            if (converged) {
                DEBUG_PRINTF("Eigenvectors QR converged after %d iterations\n",
                             iter + 1);
                break;
            }
        } catch (...) {
            break;
        }
    }

    if constexpr (std::is_floating_point_v<ComputeType>) {
        auto eigvals = this->template eigenvalues_qr<ComputeType>(max_iterations);

        bool has_complex = false;
        for (const auto &val : eigvals) {
            if (std::abs(std::imag(val)) > 1e-10) {
                has_complex = true;
                break;
            }
        }

        if (has_complex) {
            using ComplexType = std::complex<ComputeType>;
            auto A_complex = this->template cast_to<ComplexType>();
            auto V_complex =
                A_complex.template eigenvectors_qr<ComplexType>(max_iterations);

            return V_complex.template cast_to<ComputeType>();
        }
    }

    return V;
}

template<typename T>
template<typename ComputeType>
std::pair<std::vector<ComputeType>, Matrix<ComputeType>>
Matrix<T>::eigen_qr(int max_iterations) const {
    if (min_dim_ == 0) {
        return {{}, Matrix<ComputeType>()};
    }

    if (rows_ != cols_) {
        throw std::invalid_argument("Eigen decomposition requires square matrix");
    }

    if (max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }

    auto eigvals = this->template eigenvalues_qr<ComputeType>(max_iterations);
    auto eigvecs = this->template eigenvectors_qr<ComputeType>(max_iterations);

    return {eigvals, eigvecs};
}

template<typename T>
std::vector<typename Matrix<T>::template eigen_return_type<T>>
Matrix<T>::eigenvalues(int max_iterations) const {
    if (min_dim_ == 0) {
        return {};
    }

    if (rows_ != cols_) {
        throw std::invalid_argument("Eigenvalues require square matrix");
    }

    if (max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }

    return this->template eigenvalues_qr<eigen_return_type<T>>(max_iterations);
}

template<typename T>
Matrix<typename Matrix<T>::template eigen_return_type<T>>
Matrix<T>::eigenvectors(int max_iterations) const {
    if (min_dim_ == 0) {
        return Matrix<eigen_return_type<T>>();
    }

    if (rows_ != cols_) {
        throw std::invalid_argument("Eigenvectors require square matrix");
    }

    if (max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }

    return this->template eigenvectors_qr<eigen_return_type<T>>(max_iterations);
}

template<typename T>
std::pair<std::vector<typename Matrix<T>::template eigen_return_type<T>>,
          Matrix<typename Matrix<T>::template eigen_return_type<T>>>
Matrix<T>::eigen(int max_iterations) const {
    using ComputeType = eigen_return_type<T>;

    if (min_dim_ == 0) {
        return {{}, Matrix<ComputeType>()};
    }

    if (rows_ != cols_) {
        throw std::invalid_argument("Eigen decomposition requires square matrix");
    }

    if (max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }

    try {
        return this->template eigen_qr<ComputeType>(max_iterations);
    } catch (const std::exception &e) {
        if (rows_ <= 3) {
            auto A = this->template cast_to<ComputeType>();

            if (rows_ == 1) {
                std::vector<ComputeType> eigvals = {A(0, 0)};
                Matrix<ComputeType> eigvecs(1, 1);

                if constexpr (detail::is_matrix_v<ComputeType>) {
                    eigvecs(0, 0) =
                        ComputeType::Identity(A(0, 0).get_rows(), A(0, 0).get_cols());
                } else {
                    eigvecs(0, 0) = create_scalar(A(0, 0), 1.0);
                }

                return {eigvals, eigvecs};
            } else if (rows_ == 2) {
                auto eigvals = extract_eigenvalues_2x2(A, 0);
                auto eigvecs = eigenvectors_2x2(A, eigvals);
                return {eigvals, eigvecs};
            }
        }

        throw;
    }
}
