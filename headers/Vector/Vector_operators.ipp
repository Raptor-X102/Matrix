template<typename T> Vector<T> &Vector<T>::operator+=(const Vector<T> &other) {
    Matrix<T>::operator+=(other);
    return *this;
}

template<typename T> Vector<T> &Vector<T>::operator-=(const Vector<T> &other) {
    Matrix<T>::operator-=(other);
    return *this;
}

template<typename T> Vector<T> &Vector<T>::operator*=(T scalar) {
    Matrix<T>::operator*=(scalar);
    return *this;
}

template<typename T> Vector<T> &Vector<T>::operator/=(T scalar) {
    Matrix<T>::operator/=(scalar);
    return *this;
}

template<typename T> template<typename U> Vector<U> Vector<T>::cast_to() const {
    return Vector<U>(Matrix<T>::template cast_to<U>());
}

template<typename T> template<typename U> Vector<T>::operator Vector<U>() const {
    return cast_to<U>();
}

template<typename T, typename U>
auto operator+(const Vector<T> &lhs, const Vector<U> &rhs) {
    using CommonType = typename detail::matrix_common_type<T, U>::type;
    Vector<CommonType> result(lhs.size());
    for (int i = 0; i < lhs.size(); ++i) {
        result[i] = static_cast<CommonType>(lhs[i]) + static_cast<CommonType>(rhs[i]);
    }
    return result;
}

template<typename T, typename U>
auto operator-(const Vector<T> &lhs, const Vector<U> &rhs) {
    using CommonType = typename detail::matrix_common_type<T, U>::type;
    Vector<CommonType> result(lhs.size());
    for (int i = 0; i < lhs.size(); ++i) {
        result[i] = static_cast<CommonType>(lhs[i]) - static_cast<CommonType>(rhs[i]);
    }
    return result;
}

template<typename T, typename U> 
auto operator*(const Vector<T> &vec, const U &scalar) {
    using ResultType = typename detail::matrix_common_type<T, U>::type;
    Vector<ResultType> result(vec.size());
    
    // Для блочных матриц - инициализируем правильно
    if constexpr (detail::is_matrix_v<ResultType>) {
        if (vec.size() > 0) {
            // Получаем размер блока
            int block_rows = 1, block_cols = 1;
            if constexpr (detail::is_matrix_v<T>) {
                block_rows = vec[0].get_rows();
                block_cols = vec[0].get_cols();
            }
            
            // Создаем нулевые блоки правильного размера
            auto zero_block = ResultType::Zero(block_rows, block_cols);
            for (int i = 0; i < vec.size(); ++i) {
                result[i] = zero_block;
            }
        }
    }
    
    for (int i = 0; i < vec.size(); ++i) {
        if constexpr (detail::is_matrix_v<T>) {
            // Для блочных матриц используем create_scalar
            auto scalar_cast = Matrix<T>::create_scalar(vec[i], scalar);
            result[i] = vec[i] * scalar_cast;
        } else {
            result[i] = vec[i] * scalar;
        }
    }
    
    return result;
}

template<typename T, typename U> 
auto operator/(const Vector<T> &vec, const U &scalar) {
    using ResultType = typename detail::matrix_common_type<T, U>::type;
    Vector<ResultType> result(vec.size());
    
    // Для блочных матриц - инициализируем правильно
    if constexpr (detail::is_matrix_v<ResultType>) {
        if (vec.size() > 0) {
            // Получаем размер блока
            int block_rows = 1, block_cols = 1;
            if constexpr (detail::is_matrix_v<T>) {
                block_rows = vec[0].get_rows();
                block_cols = vec[0].get_cols();
            }
            
            // Создаем нулевые блоки правильного размера
            auto zero_block = ResultType::Zero(block_rows, block_cols);
            for (int i = 0; i < vec.size(); ++i) {
                result[i] = zero_block;
            }
        }
    }
    
    for (int i = 0; i < vec.size(); ++i) {
        if constexpr (detail::is_matrix_v<T>) {
            // Для блочных матриц используем create_scalar
            auto scalar_cast = Matrix<T>::create_scalar(vec[i], scalar);
            result[i] = vec[i] / scalar_cast;
        } else {
            result[i] = vec[i] / scalar;
        }
    }
    
    return result;
}

template<typename T, typename U> auto operator*(const U &scalar, const Vector<T> &vec) {
    return vec * scalar;
}

template<typename T> T &Vector<T>::operator()(int i) {
    return Matrix<T>::operator()(i, 0);
}

template<typename T> const T &Vector<T>::operator()(int i) const {
    return Matrix<T>::operator()(i, 0);
}

template<typename T> T &Vector<T>::operator[](int i) {
    return (*this)(i);
}

template<typename T> const T &Vector<T>::operator[](int i) const {
    return (*this)(i);
}

template<typename T, typename U>
auto operator*(const Matrix<T> &matrix, const Vector<U> &vec) {
    if (matrix.get_cols() != vec.size()) {
        throw std::invalid_argument("Matrix and vector dimensions don't match for multiplication");
    }
    
    using CommonType = typename detail::matrix_common_type<T, U>::type;
    Vector<CommonType> result(matrix.get_rows());
    
    // Для блочных матриц - инициализируем правильно
    if constexpr (detail::is_matrix_v<CommonType>) {
        if (matrix.get_rows() > 0) {
            // Получаем размер блока
            int block_rows = 1, block_cols = 1;
            if constexpr (detail::is_matrix_v<T>) {
                if (matrix.get_rows() > 0 && matrix.get_cols() > 0) {
                    block_rows = matrix(0,0).get_rows();
                    block_cols = matrix(0,0).get_cols();
                }
            } else if constexpr (detail::is_matrix_v<U>) {
                if (vec.size() > 0) {
                    block_rows = vec[0].get_rows();
                    block_cols = vec[0].get_cols();
                }
            }
            
            // Создаем нулевые блоки правильного размера
            auto zero_block = CommonType::Zero(block_rows, block_cols);
            for (int i = 0; i < matrix.get_rows(); ++i) {
                result[i] = zero_block;
            }
        }
    }
    
    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int k = 0; k < matrix.get_cols(); ++k) {
            CommonType a_ik = static_cast<CommonType>(matrix(i, k));
            CommonType vec_k = static_cast<CommonType>(vec[k]);
            CommonType product = a_ik * vec_k;
            result[i] = result[i] + product;
        }
    }
    
    return result;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& vec) {
    if constexpr (detail::is_matrix_v<T>) {
        os << "Vector of " << vec.size() << " blocks:\n";
        for (int i = 0; i < vec.size(); ++i) {
            os << "  [" << i << "]: " << vec[i].get_rows() << "x" << vec[i].get_cols();
            if (vec[i].get_rows() <= 3 && vec[i].get_cols() <= 3) {
                os << " = [";
                for (int ri = 0; ri < vec[i].get_rows(); ++ri) {
                    if (ri > 0) os << "     ";
                    for (int cj = 0; cj < vec[i].get_cols(); ++cj) {
                        if (cj > 0) os << " ";
                        os << vec[i](ri, cj);
                    }
                    if (ri < vec[i].get_rows() - 1) os << "\n";
                }
                os << "]\n";
            } else {
                os << " matrix\n";
            }
        }
    } else {
        os << "[";
        for (int i = 0; i < vec.size(); ++i) {
            if (i > 0) os << ", ";
            if constexpr (detail::is_complex_v<T>) {
                os << "(" << vec[i].real() << "+" << vec[i].imag() << "i)";
            } else {
                os << vec[i];
            }
        }
        os << "]";
    }
    return os;
}
