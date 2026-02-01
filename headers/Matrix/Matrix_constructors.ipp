template<typename T>
Matrix<T>::Matrix() noexcept
    : rows_(0)
    , cols_(0)
    , min_dim_(0)
    , matrix_(nullptr) {}

template<typename T>
Matrix<T>::Matrix(int rows, int cols)
    : rows_(rows)
    , cols_(cols)
    , min_dim_(std::min(rows, cols)) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }
    alloc_matrix_();
}

template<typename T>
Matrix<T>::Matrix(const Matrix &rhs)
    : rows_(rhs.rows_)
    , cols_(rhs.cols_)
    , min_dim_(rhs.min_dim_) {
    alloc_matrix_();

    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            (*this)(i, j) = rhs(i, j);
        }
    }
}

template<typename T> Matrix<T> &Matrix<T>::operator=(const Matrix &rhs) {
    if (this != &rhs) {
        Matrix temp(rhs);
        swap_data(temp);
    }
    return *this;
}

template<typename T> Matrix<T> Matrix<T>::Square(int size) {
    if (size <= 0) {
        throw std::invalid_argument("Square matrix size must be positive");
    }
    return Matrix(size, size);
}

template<typename T> Matrix<T> Matrix<T>::Rectangular(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    return Matrix(rows, cols);
}

template<typename T> Matrix<T> Matrix<T>::Identity(int rows, int cols) {
    if constexpr (detail::is_matrix_v<T>) {
        throw std::runtime_error(
            "For block matrices use BlockIdentity() with block dimensions");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Identity matrix dimensions must be positive");
    }

    Matrix result(rows, cols);
    int min_dim = std::min(rows, cols);
    for (int i = 0; i < min_dim; i++) {
        result(i, i) = T{1};
    }
    return result;
}

template<typename T> Matrix<T> Matrix<T>::Identity(int rows) {
    return Identity(rows, rows);
}

template<typename T>
Matrix<T> Matrix<T>::BlockMatrix(int rows, int cols, int block_rows, int block_cols) {
    if constexpr (detail::is_matrix_v<T>) {
        if (rows <= 0 || cols <= 0 || block_rows <= 0 || block_cols <= 0) {
            throw std::invalid_argument("Block matrix dimensions must be positive");
        }

        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = T::Zero(block_rows, block_cols);
            }
        }
        return result;
    }
    throw std::runtime_error(
        "BlockMatrix can only be used with Matrix<Matrix<U>> types");
}

template<typename T>
Matrix<T> Matrix<T>::BlockIdentity(int rows, int cols, int block_rows, int block_cols) {
    if constexpr (detail::is_matrix_v<T>) {
        using InnerType = typename T::value_type;
        if (rows <= 0 || cols <= 0 || block_rows <= 0 || block_cols <= 0) {
            throw std::invalid_argument("Block identity dimensions must be positive");
        }

        Matrix result(rows, cols);
        int min_dim = std::min(rows, cols);
        for (int i = 0; i < min_dim; ++i) {
            result(i, i) = T::Identity(block_rows, block_cols);
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (result(i, j).get_rows() == 0) {
                    result(i, j) = T::Zero(block_rows, block_cols);
                }
            }
        }
        return result;
    }
    throw std::runtime_error(
        "BlockIdentity can only be used with Matrix<Matrix<U>> types");
}

template<typename T>
Matrix<T> Matrix<T>::BlockZero(int rows, int cols, int block_rows, int block_cols) {
    if constexpr (detail::is_matrix_v<T>) {
        if (rows <= 0 || cols <= 0 || block_rows <= 0 || block_cols <= 0) {
            throw std::invalid_argument("Block zero dimensions must be positive");
        }
        return BlockMatrix(rows, cols, block_rows, block_cols);
    }
    throw std::runtime_error("BlockZero can only be used with Matrix<Matrix<U>> types");
}

template<typename T>
Matrix<T> Matrix<T>::Diagonal(int rows, int cols, const std::vector<T> &diagonal) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Diagonal matrix dimensions must be positive");
    }

    Matrix result = Matrix::Zero(static_cast<int>(rows), static_cast<int>(cols));
    int diag_size = std::min(result.get_min_dim(), static_cast<int>(diagonal.size()));
    for (int i = 0; i < diag_size; i++)
        result.matrix_[i][i] = diagonal[i];

    return result;
}

template<typename T> Matrix<T> Matrix<T>::Diagonal(const std::vector<T> &diagonal) {
    if (diagonal.empty()) {
        throw std::invalid_argument("Diagonal vector cannot be empty");
    }

    int min_dim = diagonal.size();
    Matrix result(min_dim, min_dim);
    for (int i = 0; i < min_dim; i++)
        result.matrix_[i][i] = diagonal[i];

    return result;
}

template<typename T>
Matrix<T> Matrix<T>::Diagonal(int rows, int cols, T diagonal_value) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Diagonal matrix dimensions must be positive");
    }

    Matrix result(rows, cols);
    int min_dim = std::min(rows, cols);
    for (int i = 0; i < min_dim; i++)
        result(i, i) = diagonal_value;

    return result;
}

template<typename T> Matrix<T> Matrix<T>::Diagonal(int size, T diagonal_value) {
    return Diagonal(size, size, diagonal_value);
}

template<typename T>
Matrix<T> Matrix<T>::From_vector(const std::vector<std::vector<T>> &input) {
    if (input.empty()) {
        return Matrix(0, 0);
    }

    size_t max_cols = 0;
    for (const auto &row : input) {
        max_cols = std::max(max_cols, row.size());
    }

    size_t rows = input.size();
    Matrix result = Matrix::Zero(static_cast<int>(rows), static_cast<int>(max_cols));

    for (size_t i = 0; i < rows; i++) {
        const auto &current_row = input[i];
        size_t current_cols = current_row.size();

        for (size_t j = 0; j < current_cols; j++)
            result(i, j) = current_row[j];
    }

    return result;
}

template<typename T> Matrix<T> Matrix<T>::Zero(int rows, int cols) {
    if constexpr (detail::is_matrix_v<T>) {
        throw std::runtime_error(
            "For block matrices use BlockZero() with block dimensions");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Zero matrix dimensions must be positive");
    }

    Matrix result(rows, cols);
    result.init_zero_();
    return result;
}

template<typename T> Matrix<T> Matrix<T>::Zero(int rows) {
    return Zero(rows, rows);
}

template<typename T> Matrix<T> Matrix<T>::Read_vector() {
    int n;
    std::cin >> n;

    if (n <= 0) {
        throw std::runtime_error("Matrix size must be positive when reading from input");
    }

    if (!std::cin) {
        throw std::runtime_error("Failed to read matrix size from input");
    }

    Matrix matrix(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(std::cin >> matrix(i, j))) {
                throw std::runtime_error("Failed to read matrix element from input");
            }
        }
    }

    return matrix;
}

template<typename T>
Matrix<T> Matrix<T>::Submatrix(const Matrix &source,
                               int start_row,
                               int start_col,
                               int num_rows,
                               int num_cols) {
    if (start_row < 0 || start_row >= source.rows_ || start_col < 0
        || start_col >= source.cols_ || num_rows <= 0 || num_cols <= 0
        || start_row + num_rows > source.rows_ || start_col + num_cols > source.cols_) {
        throw std::invalid_argument("Invalid submatrix bounds");
    }

    Matrix result(num_rows, num_cols);
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            result(i, j) = source(start_row + i, start_col + j);
        }
    }
    return result;
}

template<typename T>
Matrix<T>
Matrix<T>::Submatrix(const Matrix &source, int start_row, int start_col, int size) {
    return Submatrix(source, start_row, start_col, size, size);
}

template<typename T>
Matrix<T> Matrix<T>::get_submatrix(int start_row,
                                   int start_col,
                                   int num_rows,
                                   int num_cols) const {
    return Submatrix(*this, start_row, start_col, num_rows, num_cols);
}
