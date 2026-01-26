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

template<typename T, typename U> auto operator*(const Vector<T> &vec, const U &scalar) {
    using ResultType = decltype(std::declval<T>() * std::declval<U>());
    Vector<ResultType> result(vec.size());

    for (int i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] * scalar;
    }
    return result;
}

template<typename T, typename U> auto operator/(const Vector<T> &vec, const U &scalar) {
    using ResultType = decltype(std::declval<T>() / std::declval<U>());
    Vector<ResultType> result(vec.size());

    for (int i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] / scalar;
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
auto operator*(const Matrix<T> &A, const Vector<U> &vec) {
    if (A.get_cols() != vec.size()) {
        throw std::invalid_argument("Matrix columns must match vector size for multiplication");
    }
    
    using ResultType = typename detail::matrix_common_type<T, U>::type;
    
    Vector<ResultType> result(A.get_rows());
    
    for (int i = 0; i < A.get_rows(); ++i) {
        ResultType sum = ResultType{};
        for (int j = 0; j < A.get_cols(); ++j) {
            sum = sum + A(i, j) * vec[j];
        }
        result[i] = sum;
    }
    
    return result;
}

template<typename T, typename U>
auto operator*(const Vector<T> &vec, const Matrix<U> &A) {
    if (vec.size() != A.get_rows()) {
        throw std::invalid_argument("Vector size must match matrix rows for multiplication");
    }
    
    using ResultType = typename detail::matrix_common_type<T, U>::type;
    
    Vector<ResultType> result(A.get_cols());
    
    for (int j = 0; j < A.get_cols(); ++j) {
        ResultType sum = ResultType{};
        for (int i = 0; i < A.get_rows(); ++i) {
            sum = sum + vec[i] * A(i, j);
        }
        result[j] = sum;
    }
    
    return result;
}
