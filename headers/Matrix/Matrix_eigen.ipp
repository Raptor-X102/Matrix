#include "Vector.hpp"

template<typename T>
template<typename ComputeType>
Matrix<ComputeType> Matrix<T>::balance_matrix() const {
    auto A = this->template cast_to<ComputeType>();
    int n = rows_;

    // Балансировка только для скалярных типов
    if constexpr (!detail::is_matrix_v<ComputeType>) {
        // Для комплексных и вещественных чисел
        if constexpr (detail::is_complex_v<ComputeType>
                      || std::is_floating_point_v<ComputeType>) {
            using RealType = std::conditional_t<detail::is_complex_v<ComputeType>,
                                                typename ComputeType::value_type,
                                                ComputeType>;

            const RealType radix = RealType(2);
            const RealType sqrdx = radix * radix;
            const RealType balance_threshold = RealType(0.95);

            bool converged = false;
            while (!converged) {
                converged = true;

                for (int i = 0; i < n; ++i) {
                    RealType row_norm = RealType{0};
                    RealType col_norm = RealType{0};

                    for (int j = 0; j < n; ++j) {
                        if (j != i) {
                            using std::abs;
                            row_norm += abs(A(i, j));
                            col_norm += abs(A(j, i));
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

                        // Все сравнения с вещественными числами
                        if ((row_norm + col_norm) < balance_threshold * s * f) {
                            converged = false;

                            for (int j = 0; j < n; ++j) {
                                if (j != i) {
                                    A(i, j) *= static_cast<ComputeType>(f);
                                    A(j, i) /= static_cast<ComputeType>(f);
                                }
                            }
                            A(i, i) *= static_cast<ComputeType>(f);
                        }
                    }
                }
            }
        }
        // Для целочисленных типов - упрощенная балансировка
        else if constexpr (std::is_integral_v<ComputeType>) {
            // Просто возвращаем исходную матрицу для целых чисел
        }
    }

    return A;
}

template<typename T>
template<typename ComputeType>
std::pair<Matrix<ComputeType>, Matrix<ComputeType>> Matrix<T>::qr_decomposition() const {
    int m = rows_;
    int n = cols_;

    Matrix<ComputeType> Q;
    Matrix<ComputeType> R = this->template cast_to<ComputeType>();

    if constexpr (detail::is_matrix_v<ComputeType>) {
        int block_rows = 1;
        int block_cols = 1;
        if (m > 0 && n > 0) {
            // Получаем размер блока из R
            block_rows = R(0, 0).get_rows();
            block_cols = R(0, 0).get_cols();
        }
        Q = Matrix<ComputeType>::BlockIdentity(m, m, block_rows, block_cols);
    } else {
        Q = Matrix<ComputeType>::Identity(m);
    }

    for (int k = 0; k < std::min(m - 1, n); ++k) {
        Vector<ComputeType> x(m - k);
        for (int i = k; i < m; ++i) {
            x[i - k] = R(i, k);
        }

        auto v = householder_vector(x);
        auto v_norm = v.norm();

        if (is_norm_zero(v_norm))
            continue;

        ComputeType two_scalar;
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

        for (int j = k; j < n; ++j) {
            Vector<ComputeType> col(m - k);
            for (int i = k; i < m; ++i) {
                col[i - k] = R(i, j);
            }

            ComputeType dot = col.dot(v);

            for (int i = k; i < m; ++i) {
                R(i, j) = R(i, j) - v[i - k] * dot * two_scalar;
            }
        }

        for (int j = 0; j < m; ++j) {
            Vector<ComputeType> col(m - k);
            for (int i = k; i < m; ++i) {
                col[i - k] = Q(i, j);
            }

            ComputeType dot = col.dot(v);

            for (int i = k; i < m; ++i) {
                Q(i, j) = Q(i, j) - v[i - k] * dot * two_scalar;
            }
        }
    }

    return {Q.transpose(), R};
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

    Vector<ComputeType> e1;

    if constexpr (detail::is_matrix_v<ComputeType>) {
        if (m > 0) {
            auto block = x[0];
            int block_rows = block.get_rows();
            int block_cols = block.get_cols();

            ComputeType zero_block = ComputeType::Zero(block_rows, block_cols);
            e1 = Vector<ComputeType>(m, zero_block);

            ComputeType scalar_norm_block =
                ComputeType::Diagonal(block_rows,
                                      block_cols,
                                      create_scalar(block(0, 0), norm_x));
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
        InnerType norm_v_scalar = create_scalar(v[0](0, 0), norm_v);

        for (int i = 0; i < m; ++i) {
            v[i] = v[i] / norm_v_scalar;
        }
    } else {
        v = v / norm_v;
    }

    return v;
}

template<typename T>
template<typename ComputeType>
Matrix<ComputeType> Matrix<T>::hessenberg_form() const {
    auto H = this->template cast_to<ComputeType>();
    int n = rows_;

    DEBUG_PRINTF("Hessenberg form: n=%d\n", n);

    if constexpr (detail::is_matrix_v<ComputeType>) {
        if (n > 0) {
            DEBUG_PRINTF("H block size: %dx%d\n",
                         H(0, 0).get_rows(),
                         H(0, 0).get_cols());
        }
    }

    for (int k = 0; k < n - 2; ++k) {
        Vector<ComputeType> x(n - k - 1);
        for (int i = k + 1; i < n; ++i) {
            x[i - k - 1] = H(i, k);
        }

        auto v = householder_vector(x);
        auto v_norm = v.norm();

        if (is_norm_zero(v_norm))
            continue;

        apply_householder_left(H, v, k + 1);
        apply_householder_right(H, v, k + 1);
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

    auto two_scalar = Matrix<ComputeType>::create_scalar(A(0, 0), 2.0);

    for (int j = k; j < m; ++j) {
        Vector<ComputeType> col(n - k);
        for (int i = k; i < n; ++i) {
            col[i - k] = A(i, j);
        }

        ComputeType dot;
        if constexpr (detail::is_complex_v<ComputeType>) {
            dot = ComputeType(0);
            for (int i = 0; i < n - k; ++i) {
                dot += std::conj(v[i]) * col[i];
            }
        } else {
            dot = col.dot(v);
        }

        ComputeType scale_factor;
        if constexpr (detail::is_matrix_v<ComputeType>) {
            scale_factor = two_scalar * dot;
        } else {
            scale_factor = dot * two_scalar;
        }

        for (int i = k; i < n; ++i) {
            A(i, j) = A(i, j) - v[i - k] * scale_factor;
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

    auto two_scalar = Matrix<ComputeType>::create_scalar(A(0, 0), 2.0);

    for (int i = 0; i < n; ++i) {
        Vector<ComputeType> row(m - k);
        for (int j = k; j < m; ++j) {
            row[j - k] = A(i, j);
        }

        ComputeType dot;
        if constexpr (detail::is_complex_v<ComputeType>) {
            dot = ComputeType(0);
            for (int idx = 0; idx < m - k; ++idx) {
                dot += std::conj(v[idx]) * row[idx];
            }
        } else {
            dot = row.dot(v);
        }

        ComputeType scale_factor = dot * two_scalar;

        for (int j = k; j < m; ++j) {
            A(i, j) = A(i, j) - v[j - k] * scale_factor;
        }
    }
}

template<typename T>
template<typename ComputeType>
Vector<ComputeType> Matrix<T>::back_substitution(const Matrix<ComputeType> &R,
                                                 const Vector<ComputeType> &y) const {
    int n = R.get_rows();
    Vector<ComputeType> x;

    if constexpr (detail::is_matrix_v<ComputeType>) {
        if (n > 0) {
            int block_rows = 1;
            int block_cols = 1;
            if (R(0, 0).get_rows() > 0 && R(0, 0).get_cols() > 0) {
                block_rows = R(0, 0).get_rows();
                block_cols = R(0, 0).get_cols();
            }
            ComputeType zero_block = ComputeType::Zero(block_rows, block_cols);
            x = Vector<ComputeType>(n, zero_block);
        } else {
            x = Vector<ComputeType>(0);
        }
    } else {
        x = Vector<ComputeType>(n);
    }

    for (int i = n - 1; i >= 0; --i) { // ИСПРАВЛЕНО: --i вместо ++i
        ComputeType sum;
        if constexpr (detail::is_matrix_v<ComputeType>) {
            int block_rows = 1;
            int block_cols = 1;
            if (n > 0 && R(0, 0).get_rows() > 0 && R(0, 0).get_cols() > 0) {
                block_rows = R(0, 0).get_rows();
                block_cols = R(0, 0).get_cols();
            }
            sum = ComputeType::Zero(block_rows, block_cols);
        } else {
            sum = ComputeType{0};
        }

        for (int j = i + 1; j < n; ++j) {
            sum = sum + R(i, j) * x[j];
        }
        x[i] = (y[i] - sum) / R(i, i);
    }

    return x;
}

template<typename T>
template<typename ComputeType>
Vector<ComputeType> Matrix<T>::inverse_iteration(const Matrix<ComputeType> &A,
                                                 const ComputeType &lambda,
                                                 int max_iterations) const {
    int n = A.get_rows();

    Matrix<ComputeType> I;
    if constexpr (detail::is_matrix_v<ComputeType>) {
        int block_rows = 1;
        int block_cols = 1;
        if (n > 0 && A(0, 0).get_rows() > 0 && A(0, 0).get_cols() > 0) {
            block_rows = A(0, 0).get_rows();
            block_cols = A(0, 0).get_cols();
        }
        I = Matrix<ComputeType>::BlockIdentity(n, n, block_rows, block_cols);
    } else {
        I = Matrix<ComputeType>::Identity(n);
    }

    auto lambda_I = I * lambda;
    auto B = A - lambda_I;

    auto x = Vector<ComputeType>::random(n);
    auto x_norm = x.norm();
    x = x / x_norm;

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
std::vector<ComputeType> Matrix<T>::extract_eigenvalues_2x2(const Matrix<ComputeType>& H, int i) const {
    std::vector<ComputeType> eigenvalues;
    
    if constexpr (detail::is_matrix_v<ComputeType>) {
        using InnerType = typename ComputeType::value_type;
        
        auto a = H(i, i);
        auto b = H(i, i + 1);
        auto c = H(i + 1, i);
        auto d = H(i + 1, i + 1);
        
        if constexpr (detail::is_matrix_v<InnerType>) {
            // Для блочных матриц высокого уровня - простой подход
            eigenvalues.push_back(H(i, i));
            eigenvalues.push_back(H(i + 1, i + 1));
        } else {
            auto trace = a + d;
            auto det_val = a * d - b * c;
            
            try {
                auto I = ComputeType::Identity(a.get_rows(), a.get_cols());
                auto scalar_two = create_scalar(a(0,0), 2);
                auto discriminant = trace * trace - scalar_two * scalar_two * det_val;
                
                if constexpr (detail::has_sqrt_v<decltype(discriminant)>) {
                    auto sqrt_disc = discriminant.sqrt();
                    eigenvalues.push_back((trace + sqrt_disc) / (scalar_two * scalar_two));
                    eigenvalues.push_back((trace - sqrt_disc) / (scalar_two * scalar_two));
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
        } 
        else if constexpr (std::is_floating_point_v<ComputeType>) {
            auto discriminant = trace * trace - ComputeType(4) * det_val;
            
            // Всегда используем комплексные числа для вычисления корня
            using ComplexType = std::complex<ComputeType>;
            auto sqrt_disc = sqrt(ComplexType(discriminant));
            eigenvalues.push_back(static_cast<ComputeType>((trace + sqrt_disc) / ComputeType(2)));
            eigenvalues.push_back(static_cast<ComputeType>((trace - sqrt_disc) / ComputeType(2)));
        }
        else if constexpr (std::is_integral_v<ComputeType>) {
            // Для целых чисел - простой подход
            eigenvalues.push_back(H(i, i));
            eigenvalues.push_back(H(i + 1, i + 1));
        }
        else {
            // Для других типов
            eigenvalues.push_back(H(i, i));
            eigenvalues.push_back(H(i + 1, i + 1));
        }
    }
    
    return eigenvalues;
}

template<typename T>
template<typename ComputeType>
Matrix<ComputeType> Matrix<T>::eigenvectors_2x2(const Matrix<ComputeType>& A, 
                                               const std::vector<ComputeType>& eigenvalues) const {
    Matrix<ComputeType> V(2, 2);
    
    if constexpr (detail::is_matrix_v<ComputeType>) {
        // Для блочных матриц
        if (A.get_rows() > 0 && A(0,0).get_rows() > 0) {
            int block_rows = A(0,0).get_rows();
            int block_cols = A(0,0).get_cols();
            
            auto zero_block = ComputeType::Zero(block_rows, block_cols);
            auto identity_block = ComputeType::Identity(block_rows, block_cols);
            
            // Инициализируем нулями
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    V(i, j) = zero_block;
                }
            }
            
            for (size_t i = 0; i < eigenvalues.size(); ++i) {
                auto lambda = eigenvalues[i];
                
                // Создаем единичную матрицу
                auto I_mat = Matrix<ComputeType>::BlockIdentity(2, 2, block_rows, block_cols);
                // A - λI - должно работать, так как λ - скаляр (внутренний тип)
                auto B = A - I_mat * lambda;
                
                // Ищем собственный вектор
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
        // Для скалярных типов
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
            
            // Нормализуем
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
    if (rows_ != cols_) {
        throw std::invalid_argument("Eigenvalues require square matrix");
    }
    
    Matrix<ComputeType> H;
    int n = rows_;
    
    // Для скалярных типов применяем балансировку
    if constexpr (!detail::is_matrix_v<ComputeType> && 
                  (std::is_floating_point_v<ComputeType> || detail::is_complex_v<ComputeType>)) {
        if (n > 3) {  // Балансируем только матрицы размера > 3
            H = this->template balance_matrix<ComputeType>();
        } else {
            H = this->template cast_to<ComputeType>();
        }
    } else {
        H = this->template cast_to<ComputeType>();
    }
    
    // Для матриц размера 1 или 2 используем прямое вычисление
    if (n == 1) {
        return {H(0, 0)};
    }
    
    if (n == 2) {
        return extract_eigenvalues_2x2(H, 0);
    }
    
    // Приводим к форме Хессенберга
    H = H.template hessenberg_form<ComputeType>();
    
    // QR-алгоритм со сдвигом
    const int adjusted_iterations = max_iterations * 2;
    
    for (int iter = 0; iter < adjusted_iterations; ++iter) {
        // Вычисляем сдвиг для лучшей сходимости
        // Сдвиг - всегда скаляр
        auto shift = create_scalar(H(0,0), 0);
        
        if constexpr (!detail::is_matrix_v<ComputeType>) {
            // Для скалярных типов вычисляем Wilkinson shift
            if (n >= 2) {
                auto a = H(n-2, n-2);
                auto b = H(n-2, n-1);
                auto c = H(n-1, n-2);
                auto d = H(n-1, n-1);
                
                auto trace = a + d;
                auto det_val = a * d - b * c;
                
                if constexpr (detail::is_complex_v<ComputeType>) {
                    using RealType = typename ComputeType::value_type;
                    auto discriminant = trace * trace - RealType(4) * det_val;
                    
                    using std::sqrt;
                    auto sqrt_disc = sqrt(discriminant);
                    auto lambda1 = (trace + sqrt_disc) / RealType(2);
                    auto lambda2 = (trace - sqrt_disc) / RealType(2);
                    
                    using std::abs;
                    auto dist1 = abs(lambda1 - d);
                    auto dist2 = abs(lambda2 - d);
                    shift = (dist1 < dist2) ? lambda1 : lambda2;
                } 
                else if constexpr (std::is_floating_point_v<ComputeType>) {
                    auto discriminant = trace * trace - ComputeType(4) * det_val;
                    
                    using ComplexType = std::complex<ComputeType>;
                    using std::abs;
                    using std::sqrt;
                    
                    ComplexType sqrt_disc = sqrt(ComplexType(discriminant));
                    ComplexType lambda1 = ComplexType((trace + sqrt_disc) / ComputeType(2));
                    ComplexType lambda2 = ComplexType((trace - sqrt_disc) / ComputeType(2));
                    
                    auto dist1 = abs(lambda1 - ComplexType(d));
                    auto dist2 = abs(lambda2 - ComplexType(d));
                    shift = static_cast<ComputeType>((dist1 < dist2) ? lambda1.real() : lambda2.real());
                }
            }
        }
        
        // QR-разложение со сдвигом
        Matrix<ComputeType> I;
        if constexpr (detail::is_matrix_v<ComputeType>) {
            // Для блочных матриц
            int block_rows = 1, block_cols = 1;
            if (n > 0 && H(0,0).get_rows() > 0 && H(0,0).get_cols() > 0) {
                block_rows = H(0,0).get_rows();
                block_cols = H(0,0).get_cols();
            }
            I = Matrix<ComputeType>::BlockIdentity(n, n, block_rows, block_cols);
        } else {
            I = Matrix<ComputeType>::Identity(n);
        }
        
        // H - shift * I
        auto H_shifted = H - I * shift;
        
        try {
            auto [Q, R] = H_shifted.template qr_decomposition<ComputeType>();
            // H = R * Q + shift * I
            H = R * Q + I * shift;
        } catch (...) {
            // Если QR разложение не удалось, пробуем без сдвига
            try {
                auto [Q, R] = H.template qr_decomposition<ComputeType>();
                H = R * Q;
            } catch (...) {
                break;  // Если и это не работает, выходим
            }
        }
        
        // Проверяем сходимость
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
    
    // Извлекаем собственные значения
    std::vector<ComputeType> eigenvalues;
    int i = 0;
    
    while (i < n) {
        if (i == n - 1) {
            // 1x1 блок
            eigenvalues.push_back(H(i, i));
            i++;
        } else {
            auto off_diag = H(i, i + 1);
            
            bool is_zero_off_diag = is_norm_zero(off_diag);
            
            if (is_zero_off_diag) {
                // Диагональный элемент
                eigenvalues.push_back(H(i, i));
                i++;
            } else {
                // 2x2 блок
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
    auto A_orig = this->template cast_to<ComputeType>();
    int n = rows_;

    Matrix<ComputeType> V;

    if constexpr (detail::is_matrix_v<ComputeType>) {
        int block_rows = 1;
        int block_cols = 1;
        if (n > 0 && A_orig(0, 0).get_rows() > 0 && A_orig(0, 0).get_cols() > 0) {
            block_rows = A_orig(0, 0).get_rows();
            block_cols = A_orig(0, 0).get_cols();
        }
        V = Matrix<ComputeType>::BlockIdentity(n, n, block_rows, block_cols);
    } else {
        V = Matrix<ComputeType>::Identity(n);
    }

    Matrix<ComputeType> H = A_orig;

    // Приводим к форме Хессенберга
    if (n > 2) {
        H = H.template hessenberg_form<ComputeType>();
    }

    // Итерационный QR-алгоритм для накопления преобразований
    const int adjusted_iterations = max_iterations;

    for (int iter = 0; iter < adjusted_iterations; ++iter) {
        try {
            auto [Q, R] = H.template qr_decomposition<ComputeType>();
            V = V * Q;
            H = R * Q;

            // Проверяем сходимость
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

    // Для комплексных собственных значений нужна специальная обработка
    if constexpr (std::is_floating_point_v<ComputeType>) {
        // Для вещественных матриц проверяем комплексные собственные пары
        auto eigvals = this->template eigenvalues_qr<ComputeType>(max_iterations);

        // Проверяем наличие комплексных собственных значений
        bool has_complex = false;
        for (const auto &val : eigvals) {
            if (std::abs(std::imag(val)) > 1e-10) {
                has_complex = true;
                break;
            }
        }

        if (has_complex) {
            // Для комплексных собственных значений нужны комплексные собственные векторы
            using ComplexType = std::complex<ComputeType>;
            auto A_complex = this->template cast_to<ComplexType>();
            auto V_complex =
                A_complex.template eigenvectors_qr<ComplexType>(max_iterations);

            // Приводим обратно к вещественному типу (если возможно)
            return V_complex.template cast_to<ComputeType>();
        }
    }

    return V;
}

template<typename T>
template<typename ComputeType>
std::pair<std::vector<ComputeType>, Matrix<ComputeType>>
Matrix<T>::eigen_qr(int max_iterations) const {
    auto eigvals = this->template eigenvalues_qr<ComputeType>(max_iterations);
    auto eigvecs = this->template eigenvectors_qr<ComputeType>(max_iterations);

    return {eigvals, eigvecs};
}

template<typename T>
std::vector<typename Matrix<T>::template eigen_return_type<T>>
Matrix<T>::eigenvalues(int max_iterations) const {
    return this->template eigenvalues_qr<eigen_return_type<T>>(max_iterations);
}

template<typename T>
Matrix<typename Matrix<T>::template eigen_return_type<T>>
Matrix<T>::eigenvectors(int max_iterations) const {
    return this->template eigenvectors_qr<eigen_return_type<T>>(max_iterations);
}

template<typename T>
std::pair<std::vector<typename Matrix<T>::template eigen_return_type<T>>, 
          Matrix<typename Matrix<T>::template eigen_return_type<T>>> 
Matrix<T>::eigen(int max_iterations) const {
    using ComputeType = eigen_return_type<T>;
    
    // Увеличиваем число итераций для лучшей сходимости
    int adjusted_max_iterations = max_iterations;
    
    if constexpr (std::is_floating_point_v<T> || std::is_integral_v<T>) {
        adjusted_max_iterations = max_iterations * 2;
    }
    
    try {
        return this->template eigen_qr<ComputeType>(adjusted_max_iterations);
    } catch (const std::exception& e) {
        // В случае ошибки пробуем альтернативный подход для маленьких матриц
        if (rows_ <= 3) {
            auto A = this->template cast_to<ComputeType>();
            
            if (rows_ == 1) {
                // 1x1 матрица
                std::vector<ComputeType> eigvals = {A(0, 0)};
                Matrix<ComputeType> eigvecs(1, 1);
                
                if constexpr (detail::is_matrix_v<ComputeType>) {
                    eigvecs(0, 0) = ComputeType::Identity(A(0,0).get_rows(), A(0,0).get_cols());
                } else {
                    eigvecs(0, 0) = create_scalar(A(0,0), 1.0);
                }
                
                return {eigvals, eigvecs};
            }
            else if (rows_ == 2) {
                // 2x2 матрица
                auto eigvals = extract_eigenvalues_2x2(A, 0);
                auto eigvecs = eigenvectors_2x2(A, eigvals);
                return {eigvals, eigvecs};
            }
        }
        
        // Если ничего не помогло, пробрасываем исключение
        throw;
    }
}
