#include "Vector.hpp"

template<typename T>
template<typename ComputeType>
Vector<ComputeType> Matrix<T>::householder_vector(const Vector<ComputeType>& x) const {
    int m = x.size();
    
    auto norm_x = x.norm();
    
    if (this->template is_norm_zero<ComputeType>(norm_x)) {
        return Vector<ComputeType>(m);
    }
    
    auto e1 = Vector<ComputeType>(m);
    e1[0] = norm_x;
    
    auto v = x + e1;
    auto norm_v = v.norm();
    
    if (this->template is_norm_zero<ComputeType>(norm_v)) {
        return Vector<ComputeType>(m);
    }
    
    v = v / norm_v;
    
    return v;
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
        if (m > 0 && n > 0 && R(0,0).get_rows() > 0) {
            block_rows = R(0,0).get_rows();
            block_cols = R(0,0).get_cols();
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
        
        if (this->template is_norm_zero<ComputeType>(v_norm)) 
            continue;
        
        auto two_scalar = Matrix<ComputeType>::create_scalar(R(0,0), 2.0);
        
        for (int j = k; j < n; ++j) {
            Vector<ComputeType> col(m - k);
            for (int i = k; i < m; ++i) {
                col[i - k] = R(i, j);
            }
            
            auto dot = col.dot(v);
            
            for (int i = k; i < m; ++i) {
                R(i, j) = R(i, j) - v[i - k] * dot * two_scalar;
            }
        }
        
        for (int j = 0; j < m; ++j) {
            Vector<ComputeType> col(m - k);
            for (int i = k; i < m; ++i) {
                col[i - k] = Q(i, j);
            }
            
            auto dot = col.dot(v);
            
            for (int i = k; i < m; ++i) {
                Q(i, j) = Q(i, j) - v[i - k] * dot * two_scalar;
            }
        }
    }
    
    return {Q.transpose(), R};
}

template<typename T>
template<typename ComputeType>
Matrix<ComputeType> Matrix<T>::hessenberg_form() const {
    auto H = this->template cast_to<ComputeType>();
    int n = rows_;
    
    for (int k = 0; k < n - 2; ++k) {
        Vector<ComputeType> x(n - k - 1);
        for (int i = k + 1; i < n; ++i) {
            x[i - k - 1] = H(i, k);
        }
        
        auto v = householder_vector(x);
        auto v_norm = v.norm();
        
        if (this->template is_norm_zero<ComputeType>(v_norm)) continue;
        
        apply_householder_left(H, v, k + 1);
        apply_householder_right(H, v, k + 1);
    }
    
    return H;
}

template<typename T>
template<typename ComputeType>
void Matrix<T>::apply_householder_left(Matrix<ComputeType>& A, 
                                      const Vector<ComputeType>& v, 
                                      int k) const {
    int n = A.get_rows();
    int m = A.get_cols();
    
    auto two_scalar = Matrix<ComputeType>::create_scalar(A(0,0), 2.0);
    
    for (int j = k; j < m; ++j) {
        Vector<ComputeType> col(n - k);
        for (int i = k; i < n; ++i) {
            col[i - k] = A(i, j);
        }
        
        auto dot = col.dot(v);
        
        for (int i = k; i < n; ++i) {
            A(i, j) = A(i, j) - v[i - k] * dot * two_scalar;
        }
    }
}

template<typename T>
template<typename ComputeType>
void Matrix<T>::apply_householder_right(Matrix<ComputeType>& A, 
                                       const Vector<ComputeType>& v, 
                                       int k) const {
    int n = A.get_rows();
    int m = A.get_cols();
    
    auto two_scalar = Matrix<ComputeType>::create_scalar(A(0,0), 2.0);
    
    for (int i = 0; i < n; ++i) {
        Vector<ComputeType> row(m - k);
        for (int j = k; j < m; ++j) {
            row[j - k] = A(i, j);
        }
        
        auto dot = row.dot(v);
        
        for (int j = k; j < m; ++j) {
            A(i, j) = A(i, j) - v[j - k] * dot * two_scalar;
        }
    }
}

template<typename T>
template<typename ComputeType>
Vector<ComputeType> Matrix<T>::back_substitution(const Matrix<ComputeType>& R, 
                                                const Vector<ComputeType>& y) const {
    int n = R.get_rows();
    Vector<ComputeType> x(n);
    
    for (int i = n - 1; i >= 0; --i) {
        ComputeType sum;
        if constexpr (detail::is_matrix_v<ComputeType>) {
            int block_rows = 1;
            int block_cols = 1;
            if (n > 0 && R(0,0).get_rows() > 0) {
                block_rows = R(0,0).get_rows();
                block_cols = R(0,0).get_cols();
            }
            sum = Matrix<typename ComputeType::value_type>::Zero(block_rows, block_cols);
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
Vector<ComputeType> Matrix<T>::inverse_iteration(const Matrix<ComputeType>& A, const ComputeType& lambda, int max_iterations) const {
    int n = A.get_rows();
    
    Matrix<ComputeType> I;
    if constexpr (detail::is_matrix_v<ComputeType>) {
        int block_rows = 1;
        int block_cols = 1;
        if (n > 0 && A(0,0).get_rows() > 0) {
            block_rows = A(0,0).get_rows();
            block_cols = A(0,0).get_cols();
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
        eigenvalues.push_back(H(i, i));
        eigenvalues.push_back(H(i + 1, i + 1));
    } else {
        auto a = H(i, i);
        auto b = H(i, i + 1);
        auto c = H(i + 1, i);
        auto d = H(i + 1, i + 1);
        
        auto trace = a + d;
        auto det = a * d - b * c;
        
        auto two = Matrix<ComputeType>::create_scalar(a, 2.0);
        auto half = trace / two;
        auto quarter_trace_sq = half * half;
        auto discriminant = quarter_trace_sq - det;
        
        using std::sqrt;
        auto sqrt_disc = sqrt(discriminant);
        eigenvalues.push_back(half + sqrt_disc);
        eigenvalues.push_back(half - sqrt_disc);
    }
    
    return eigenvalues;
}

template<typename T>
template<typename ComputeType>
std::vector<ComputeType> Matrix<T>::eigenvalues_qr(int max_iterations) const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Eigenvalues require square matrix");
    }
    
    using ActualComputeType = ComputeType;
    
    auto H = this->template cast_to<ActualComputeType>();
    int n = H.get_rows();
    
    H = H.template hessenberg_form<ActualComputeType>();
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        auto [Q, R] = H.template qr_decomposition<ActualComputeType>();
        H = R * Q;
        
        auto off_norm = this->template off_diagonal_norm(H);
        
        if (this->template is_norm_zero<decltype(off_norm)>(off_norm)) break;
    }
    
    std::vector<ActualComputeType> eigenvalues;
    int i = 0;
    
    while (i < n) {
        if (i == n - 1) {
            eigenvalues.push_back(H(i, i));
            i++;
        } else {
            auto off_diag = H(i, i + 1);
            
            bool is_zero_off_diag = false;
            if constexpr (detail::is_matrix_v<ActualComputeType>) {
                is_zero_off_diag = this->template is_norm_zero<ActualComputeType>(off_diag);
            } else {
                is_zero_off_diag = this->template is_norm_zero<ActualComputeType>(off_diag);
            }
            
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
    auto eigvals = this->template eigenvalues_qr<ComputeType>(max_iterations);
    int n = rows_;
    
    Matrix<ComputeType> V;
    auto A_orig = this->template cast_to<ComputeType>();
    
    if constexpr (detail::is_matrix_v<ComputeType>) {
        int block_rows = 1;
        int block_cols = 1;
        if (n > 0 && A_orig(0,0).get_rows() > 0) {
            block_rows = A_orig(0,0).get_rows();
            block_cols = A_orig(0,0).get_cols();
        }
        V = Matrix<ComputeType>::BlockIdentity(n, n, block_rows, block_cols);
    } else {
        V = Matrix<ComputeType>::Identity(n);
    }
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        auto [Q, R] = A_orig.template qr_decomposition<ComputeType>();
        V = V * Q;
        A_orig = R * Q;
        
        auto off_norm = this->template off_diagonal_norm(A_orig);
        
        if (this->template is_norm_zero<decltype(off_norm)>(off_norm)) break;
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
    using ComputeType = eigen_return_type<T>;
    return this->template eigenvalues_qr<ComputeType>(max_iterations);
}

template<typename T>
Matrix<typename Matrix<T>::template eigen_return_type<T>> 
Matrix<T>::eigenvectors(int max_iterations) const {
    using ComputeType = eigen_return_type<T>;
    return this->template eigenvectors_qr<ComputeType>(max_iterations);
}

template<typename T>
std::pair<std::vector<typename Matrix<T>::template eigen_return_type<T>>, 
          Matrix<typename Matrix<T>::template eigen_return_type<T>>> 
Matrix<T>::eigen(int max_iterations) const {
    using ComputeType = eigen_return_type<T>;
    return this->template eigen_qr<ComputeType>(max_iterations);
}
