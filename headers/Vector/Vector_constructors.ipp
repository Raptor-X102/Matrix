template<typename T>
Vector<T>::Vector()
    : Matrix<T>(0, 1) {}

template<typename T>
Vector<T>::Vector(int size)
    : Matrix<T>(size, 1) {}

template<typename T>
Vector<T>::Vector(int size, T initial_value)
    : Matrix<T>(size, 1) {
    for (int i = 0; i < size; ++i) {
        Matrix<T>::operator()(i, 0) = initial_value;
    }
}

template<typename T>
Vector<T>::Vector(const std::vector<T> &data)
    : Matrix<T>(static_cast<int>(data.size()), 1) {
    for (size_t i = 0; i < data.size(); ++i) {
        Matrix<T>::operator()(static_cast<int>(i), 0) = data[i];
    }
}

template<typename T>
Vector<T>::Vector(const Matrix<T> &matrix)
    : Matrix<T>(matrix) {
    if (matrix.get_cols() != 1) {
        throw std::invalid_argument("Matrix must be a column vector (n x 1)");
    }
}

template<typename T>
Vector<T> Vector<T>::from_row(const Matrix<T> &matrix) {
    if (matrix.get_rows() != 1) {
        throw std::invalid_argument("Matrix must be a row vector (1 x n)");
    }
    Vector<T> result(matrix.get_cols());
    for (int i = 0; i < matrix.get_cols(); ++i) {
        result(i) = matrix(0, i);
    }
    return result;
}
