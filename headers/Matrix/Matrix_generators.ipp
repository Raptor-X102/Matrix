template<typename T>
Matrix<T> Matrix<T>::Generate_matrix(int rows,
                                     int cols,
                                     T min_val,
                                     T max_val,
                                     int iterations,
                                     T target_determinant_magnitude,
                                     T max_condition_number) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }

    if (iterations < 0) {
        throw std::invalid_argument("Iterations must be non-negative");
    }

    if constexpr (detail::is_matrix_v<T>) {
        return generate_block_matrix(rows,
                                     cols,
                                     min_val,
                                     max_val,
                                     iterations,
                                     target_determinant_magnitude,
                                     max_condition_number);
    } else {
        return generate_scalar_matrix(rows,
                                      cols,
                                      min_val,
                                      max_val,
                                      iterations,
                                      target_determinant_magnitude,
                                      max_condition_number);
    }
}

template<typename T>
Matrix<T> Matrix<T>::Generate_binary_matrix(int rows,
                                            int cols,
                                            T target_determinant_magnitude,
                                            int iterations) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }

    if (iterations < 0) {
        throw std::invalid_argument("Iterations must be non-negative");
    }

    try {
        return generate_binary_matrix_impl(rows,
                                           cols,
                                           target_determinant_magnitude,
                                           iterations);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Failed to generate binary matrix: ")
                                 + e.what());
    }
}

template<typename T>
Matrix<T> Matrix<T>::generate_block_matrix(int rows,
                                           int cols,
                                           T min_val,
                                           T max_val,
                                           int iterations,
                                           T target_determinant_magnitude,
                                           T max_condition_number) {
    Matrix result(rows, cols);

    int inner_rows = min_val.get_rows();
    int inner_cols = min_val.get_cols();

    using InnerType = typename T::value_type;

    InnerType inner_min = static_cast<InnerType>(-10);
    InnerType inner_max = static_cast<InnerType>(10);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = T::Generate_matrix(inner_rows,
                                              inner_cols,
                                              inner_min,
                                              inner_max,
                                              iterations,
                                              static_cast<InnerType>(1),
                                              static_cast<InnerType>(1e10));
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::generate_scalar_matrix(int rows,
                                            int cols,
                                            T min_val,
                                            T max_val,
                                            int iterations,
                                            T target_determinant_magnitude,
                                            T max_condition_number) {
    validate_generation_parameters(min_val, max_val, iterations, max_condition_number);

    std::vector<T> diagonal = create_controlled_diagonal(std::min(rows, cols),
                                                         min_val,
                                                         max_val,
                                                         target_determinant_magnitude);
    Matrix result = Matrix::Diagonal(rows, cols, diagonal);
    result.fill_upper_triangle(min_val, max_val);

    apply_controlled_transformations(result, rows, cols, min_val, max_val, iterations);

    return result;
}

template<typename T> void Matrix<T>::fill_upper_triangle(T min_val, T max_val) {
    if constexpr (detail::is_ordered_v<T>) {
        if (min_val > max_val) {
            throw std::invalid_argument("min_val must be less than or equal to max_val");
        }
    }

    for (int i = 0; i < min_dim_; ++i) {
        for (int j = i + 1; j < cols_; ++j) {
            T val = generate_random(min_val, max_val);
            if constexpr (detail::is_ordered_v<T>) {
                if constexpr (std::is_floating_point_v<T>) {
                    val = std::clamp(val, T{-10.0}, T{10.0});
                } else {
                    val = std::clamp(val,
                                     static_cast<T>(min_val),
                                     static_cast<T>(max_val));
                }
            }
            (*this)(i, j) = val;
        }
    }
}

template<typename T> int Matrix<T>::generate_random_int_(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);

    return dis(gen);
}

template<typename T> double Matrix<T>::generate_random_double_(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);

    return dis(gen);
}

template<typename T> T Matrix<T>::generate_random(T min_val, T max_val) {
    if constexpr (std::is_same_v<T, int>) {
        int actual_min = static_cast<int>(min_val);
        int actual_max = static_cast<int>(max_val);
        if (actual_min == 0 && actual_max == 0) {
            actual_min = 1;
            actual_max = 100;
        }
        return generate_random_int_(actual_min, actual_max);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        double real_min = 0.0, real_max = 1.0;
        double imag_min = 0.0, imag_max = 1.0;

        if (min_val != std::complex<double>{} || max_val != std::complex<double>{}) {
            real_min = min_val.real();
            real_max = max_val.real();
            imag_min = min_val.imag();
            imag_max = max_val.imag();
        }

        return std::complex<double>(generate_random_double_(real_min, real_max),
                                    generate_random_double_(imag_min, imag_max));
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        float real_min = 0.0f, real_max = 1.0f;
        float imag_min = 0.0f, imag_max = 1.0f;

        if (min_val != std::complex<float>{} || max_val != std::complex<float>{}) {
            real_min = min_val.real();
            real_max = max_val.real();
            imag_min = min_val.imag();
            imag_max = max_val.imag();
        }

        return std::complex<float>(
            static_cast<float>(generate_random_double_(real_min, real_max)),
            static_cast<float>(generate_random_double_(imag_min, imag_max)));
    } else if constexpr (std::is_floating_point_v<T>) {
        double actual_min = static_cast<double>(min_val);
        double actual_max = static_cast<double>(max_val);
        if (actual_min == 0.0 && actual_max == 0.0) {
            actual_min = 0.0;
            actual_max = 1.0;
        }
        return generate_random_double_(actual_min, actual_max);
    } else {
        double actual_min = 0.0;
        double actual_max = 1.0;

        if constexpr (detail::is_ordered_v<T>) {
            actual_min = static_cast<double>(min_val);
            actual_max = static_cast<double>(max_val);
        }

        return T{generate_random_double_(actual_min, actual_max)};
    }
}

template<typename T>
void Matrix<T>::validate_generation_parameters(T &min_val,
                                               T &max_val,
                                               int &iterations,
                                               T &max_condition_number) {
    if constexpr (std::is_floating_point_v<T>) {
        if (min_val == T{} && max_val == T{}) {
            min_val = T{-10};
            max_val = T{10};
        }
        if (max_condition_number == T{}) {
            max_condition_number = T{1e10};
        }
    } else {
        if (iterations > 10) {
            iterations = 10;
        }
        if (max_condition_number == T{}) {
            max_condition_number = T{1000000000};
        }
    }

    if constexpr (detail::is_ordered_v<T>) {
        if (min_val > max_val) {
            throw std::invalid_argument("min_val must be less than or equal to max_val");
        }
    }
}

template<typename T>
Matrix<T> Matrix<T>::generate_binary_matrix_impl(int rows,
                                                 int cols,
                                                 T target_determinant_magnitude,
                                                 int iterations) {
    int min_dim = std::min(rows, cols);
    std::vector<T> diagonal = generate_binary_diagonal(min_dim);

    Matrix result = create_initial_binary_matrix(rows, cols, diagonal);

    return apply_binary_transformations(result, min_dim, rows, cols, iterations);
}

template<typename T> std::vector<T> Matrix<T>::generate_binary_diagonal(int size) {
    std::vector<T> diagonal(size);
    for (int i = 0; i < size; ++i) {
        diagonal[i] = generate_random_int_(0, 1) == 0 ? T{-1} : T{1};
    }
    return diagonal;
}

template<typename T>
Matrix<T> Matrix<T>::create_initial_binary_matrix(int rows,
                                                  int cols,
                                                  const std::vector<T> &diagonal) {
    Matrix result = Matrix::Diagonal(rows, cols, diagonal);

    int min_dim = std::min(rows, cols);
    for (int i = 0; i < min_dim; ++i) {
        for (int j = i + 1; j < cols; ++j) {
            int val = generate_random_int_(-1, 1);
            result(i, j) = static_cast<T>(val);
        }
    }

    return result;
}

template<typename T>
Matrix<T> Matrix<T>::apply_binary_transformations(Matrix &matrix,
                                                  int min_dim,
                                                  int rows,
                                                  int cols,
                                                  int iterations) {
    T effective_det = compute_initial_determinant(matrix, min_dim);
    T sign = T{1};

    for (int i = 0; i < iterations; ++i) {
        apply_random_binary_transformation(matrix, rows, cols, effective_det, sign);
    }

    matrix.determinant_ = sign * effective_det;
    return matrix;
}

template<typename T>
T Matrix<T>::compute_initial_determinant(const Matrix &matrix, int min_dim) {
    T current_det = T{1};
    for (int i = 0; i < min_dim; ++i) {
        current_det *= matrix(i, i);
    }
    return current_det;
}

template<typename T>
void Matrix<T>::apply_random_binary_transformation(Matrix &matrix,
                                                   int rows,
                                                   int cols,
                                                   T &effective_det,
                                                   T &sign) {
    Tranformation_types rand_transformation =
        static_cast<Tranformation_types>(generate_random_int_(0, 2));

    switch (rand_transformation) {
    case Tranformation_types::I_TYPE:
        apply_binary_transformation_I(matrix, rows, sign);
        break;
    case Tranformation_types::II_TYPE:
        apply_binary_transformation_II(matrix, rows, effective_det);
        break;
    case Tranformation_types::III_TYPE:
        apply_binary_transformation_III(matrix, rows, cols);
        break;
    }
}

template<typename T>
void Matrix<T>::apply_binary_transformation_I(Matrix &matrix, int rows, T &sign) {
    int row_1 = generate_random_int_(0, rows - 1);
    int row_2 = generate_random_int_(0, rows - 1);
    if (row_1 != row_2) {
        matrix.swap_rows(row_1, row_2);
        sign = -sign;
    }
}

template<typename T>
void Matrix<T>::apply_binary_transformation_II(Matrix &matrix,
                                               int rows,
                                               T &effective_det) {
    int row = generate_random_int_(0, rows - 1);
    T scalar = generate_random_int_(0, 1) == 0 ? T{-1} : T{1};

    matrix.multiply_row(row, scalar);
    effective_det *= scalar;
}

template<typename T>
void Matrix<T>::apply_binary_transformation_III(Matrix &matrix, int rows, int cols) {
    int row_1 = generate_random_int_(0, rows - 1);
    int row_2 = generate_random_int_(0, rows - 1);
    if (row_1 == row_2)
        return;

    T scalar = generate_random_int_(0, 1) == 0 ? T{-1} : T{1};

    bool safe_operation = true;
    for (int c = 0; c < cols; ++c) {
        T new_val = matrix(row_1, c) + scalar * matrix(row_2, c);
        if (new_val < T{-1} || new_val > T{1}) {
            safe_operation = false;
            break;
        }
    }

    if (safe_operation) {
        matrix.add_row_scaled(row_1, row_2, scalar);
    }
}

template<typename T>
std::vector<T> Matrix<T>::create_controlled_diagonal(int size,
                                                     T min_val,
                                                     T max_val,
                                                     T target_determinant_magnitude) {
    if (size <= 0) {
        return {};
    }

    if constexpr (detail::is_ordered_v<T>) {
        if (min_val > max_val) {
            throw std::invalid_argument("min_val must be <= max_val");
        }
    }

    if constexpr (std::is_integral_v<T>) {
        return create_integer_diagonal(size,
                                       min_val,
                                       max_val,
                                       target_determinant_magnitude);
    } else {
        return create_numeric_diagonal(size,
                                       min_val,
                                       max_val,
                                       target_determinant_magnitude);
    }
}

template<typename T>
std::vector<T> Matrix<T>::create_integer_diagonal(int size,
                                                  T min_val,
                                                  T max_val,
                                                  T target_determinant_magnitude) {
    std::vector<T> diagonal(size);
    T current_product = T{1};

    for (int i = 0; i < size - 1; ++i) {
        T element = generate_nonzero_random(min_val, max_val);
        diagonal[i] = element;
        current_product *= element;
    }

    if (current_product != T{0}) {
        diagonal[size - 1] = find_best_last_element(current_product,
                                                    min_val,
                                                    max_val,
                                                    target_determinant_magnitude);
    } else {
        diagonal[size - 1] = generate_nonzero_random(min_val, max_val);
    }

    return diagonal;
}

template<typename T>
std::vector<T> Matrix<T>::create_numeric_diagonal(int size,
                                                  T min_val,
                                                  T max_val,
                                                  T target_determinant_magnitude) {
    std::vector<T> diagonal(size);
    T current_product = T{1};

    for (int i = 0; i < size; ++i) {
        T element = generate_random(min_val, max_val);

        if constexpr (detail::is_ordered_v<T>) {
            element = normalize_element(element);
        }

        if (i == size - 1 && current_product != T{0}) {
            element =
                compute_last_element(current_product, target_determinant_magnitude);

            if constexpr (detail::is_ordered_v<T>) {
                element = clamp_extreme_value(element);
            }
        }

        diagonal[i] = element;
        current_product *= element;
    }

    return diagonal;
}

template<typename T>
T Matrix<T>::find_best_last_element(T current_product,
                                    T min_val,
                                    T max_val,
                                    T target_determinant_magnitude) {
    T best_element = T{1};
    T best_diff = std::abs(target_determinant_magnitude - current_product);

    for (T test_val = min_val; test_val <= max_val; ++test_val) {
        if (test_val == T{0})
            continue;

        T test_product = current_product * test_val;
        T diff = std::abs(target_determinant_magnitude - test_product);

        if (diff < best_diff) {
            best_diff = diff;
            best_element = test_val;
        }
    }

    return best_element;
}

template<typename T> T Matrix<T>::generate_nonzero_random(T min_val, T max_val) {
    T element;
    do {
        element = generate_random(min_val, max_val);
    } while (element == T{0});
    return element;
}

template<typename T> T Matrix<T>::normalize_element(T element) {
    T abs_elem = std::abs(element);
    T scale_factor = std::clamp(abs_elem, T{0.1}, T{10.0});
    T sign = (element >= T{0}) ? T{1} : T{-1};
    return sign * scale_factor;
}

template<typename T>
T Matrix<T>::compute_last_element(T current_product, T target_determinant_magnitude) {
    if (current_product == T{0}) {
        throw std::runtime_error("Cannot compute last element: product is zero");
    }
    return target_determinant_magnitude / current_product;
}

template<typename T> T Matrix<T>::clamp_extreme_value(T value) {
    T abs_val = std::abs(value);

    if constexpr (std::is_same_v<T, float>) {
        constexpr T max_limit = T{1e10f};
        constexpr T min_limit = T{1e-10f};

        if (abs_val > max_limit) {
            return (value >= T{0}) ? max_limit : -max_limit;
        } else if (abs_val < min_limit && abs_val > T{0}) {
            return (value >= T{0}) ? min_limit : -min_limit;
        }
    } else {
        constexpr T max_limit = static_cast<T>(1e50);
        constexpr T min_limit = static_cast<T>(1e-50);

        if (abs_val > max_limit) {
            return (value >= T{0}) ? max_limit : -max_limit;
        } else if (abs_val < min_limit && abs_val > T{0}) {
            return (value >= T{0}) ? min_limit : -min_limit;
        }
    }

    return value;
}

template<typename T>
void Matrix<T>::apply_controlled_transformations(Matrix &matrix,
                                                 int rows,
                                                 int cols,
                                                 T min_val,
                                                 T max_val,
                                                 int iterations) {
    if (iterations <= 0) {
        return;
    }

    T effective_det = compute_diagonal_product(matrix, rows, cols);
    int sign = 1;

    for (int i = 0; i < iterations; ++i) {
        apply_single_transformation(matrix,
                                    rows,
                                    cols,
                                    min_val,
                                    max_val,
                                    effective_det,
                                    sign);
    }

    matrix.determinant_ = (sign == 1) ? effective_det : -effective_det;
}

template<typename T>
T Matrix<T>::compute_diagonal_product(const Matrix &matrix, int rows, int cols) {
    int min_dim = std::min(rows, cols);
    T product = T{1};

    for (int i = 0; i < min_dim; ++i) {
        product *= matrix(i, i);
    }

    return product;
}

template<typename T>
void Matrix<T>::apply_single_transformation(Matrix &matrix,
                                            int rows,
                                            int cols,
                                            T min_val,
                                            T max_val,
                                            T &effective_det,
                                            int &sign) {
    Tranformation_types transformation = select_random_transformation();

    try {
        switch (transformation) {
        case Tranformation_types::I_TYPE:
            apply_type_I_transformation(matrix, rows, sign);
            break;

        case Tranformation_types::II_TYPE:
            apply_type_II_transformation(matrix,
                                         rows,
                                         cols,
                                         min_val,
                                         max_val,
                                         effective_det);
            break;

        case Tranformation_types::III_TYPE:
            apply_type_III_transformation(matrix, rows, cols, min_val, max_val);
            break;
        }
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Transformation failed: ") + e.what());
    }
}

template<typename T>
typename Matrix<T>::Tranformation_types Matrix<T>::select_random_transformation() {
    return static_cast<Tranformation_types>(generate_random_int_(0, 2));
}

template<typename T>
void Matrix<T>::apply_type_I_transformation(Matrix &matrix, int rows, int &sign) {
    int row1 = generate_random_int_(0, rows - 1);
    int row2 = generate_random_int_(0, rows - 1);

    if (row1 != row2) {
        matrix.swap_rows(row1, row2);
        sign = -sign;
    }
}

template<typename T>
void Matrix<T>::apply_type_II_transformation(Matrix &matrix,
                                             int rows,
                                             int cols,
                                             T min_val,
                                             T max_val,
                                             T &effective_det) {
    int row = generate_random_int_(0, rows - 1);
    T scalar = generate_random(min_val, max_val);

    if (is_zero(scalar)) {
        return;
    }

    if constexpr (std::is_same_v<T, int>) {
        scalar = clamp_integer_scalar(scalar);
        if (is_safe_for_integer_multiplication(matrix,
                                               row,
                                               cols,
                                               scalar,
                                               min_val,
                                               max_val)) {
            apply_integer_multiplication(matrix, row, scalar, effective_det);
            stabilize_integer_determinant(matrix, rows, cols, effective_det);
        }
    } else {
        if constexpr (detail::is_ordered_v<T>) {
            scalar = clamp_numeric_scalar(scalar);
            if (is_safe_for_numeric_multiplication(matrix,
                                                   row,
                                                   cols,
                                                   scalar,
                                                   min_val,
                                                   max_val)) {
                apply_numeric_multiplication(matrix, row, scalar, effective_det);
                if (needs_stabilization(effective_det)) {
                    Matrix<T>::stabilize_matrix(matrix, rows, cols, effective_det);
                }
            }
        }
    }
}

template<typename T>
void Matrix<T>::apply_type_III_transformation(Matrix &matrix,
                                              int rows,
                                              int cols,
                                              T min_val,
                                              T max_val) {
    int row1 = generate_random_int_(0, rows - 1);
    int row2 = generate_random_int_(0, rows - 1);

    if (row1 == row2) {
        return;
    }

    T scalar = generate_random(min_val, max_val);
    scalar = Matrix<T>::clamp_transformation_scalar(scalar);

    if (is_safe_for_addition(matrix, row1, row2, cols, scalar, min_val, max_val)) {
        matrix.add_row_scaled(row1, row2, scalar);
    }
}

template<typename T> T Matrix<T>::clamp_integer_scalar(T scalar) {
    constexpr T limit = 100;
    if (std::abs(scalar) > limit) {
        return (scalar >= T{0}) ? limit : -limit;
    }
    return scalar;
}

template<typename T> T Matrix<T>::clamp_numeric_scalar(T scalar) {
    if constexpr (std::is_same_v<T, float>) {
        constexpr T lower_limit = T{-10.0f};
        constexpr T upper_limit = T{10.0f};
        constexpr T epsilon = T{1e-6f};

        scalar = std::clamp(scalar, lower_limit, upper_limit);
        if (std::abs(scalar) < epsilon) {
            return (scalar >= T{0}) ? epsilon : -epsilon;
        }
    } else {
        constexpr T lower_limit = T{-10.0};
        constexpr T upper_limit = T{10.0};
        constexpr T epsilon = T{1e-10};

        scalar = std::clamp(scalar, lower_limit, upper_limit);
        if (std::abs(scalar) < epsilon) {
            return (scalar >= T{0}) ? epsilon : -epsilon;
        }
    }

    return scalar;
}

template<typename T> T Matrix<T>::clamp_transformation_scalar(T scalar) {
    if constexpr (std::is_same_v<T, int>) {
        return clamp_integer_scalar(scalar);
    } else if constexpr (detail::is_ordered_v<T>) {
        constexpr T lower_limit = T{-10.0};
        constexpr T upper_limit = T{10.0};
        return std::clamp(scalar, lower_limit, upper_limit);
    }
    return scalar;
}

template<typename T>
bool Matrix<T>::is_safe_for_integer_multiplication(const Matrix &matrix,
                                                   int row,
                                                   int cols,
                                                   T scalar,
                                                   T min_val,
                                                   T max_val) {
    for (int c = 0; c < cols; ++c) {
        T new_val = matrix(row, c) * scalar;
        if (new_val < min_val || new_val > max_val) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool Matrix<T>::is_safe_for_numeric_multiplication(const Matrix &matrix,
                                                   int row,
                                                   int cols,
                                                   T scalar,
                                                   T min_val,
                                                   T max_val) {
    if constexpr (detail::is_ordered_v<T>) {
        for (int c = 0; c < cols; ++c) {
            T new_val = matrix(row, c) * scalar;
            if (new_val < min_val || new_val > max_val) {
                return false;
            }
        }
    }
    return true;
}

template<typename T>
void Matrix<T>::apply_integer_multiplication(Matrix &matrix,
                                             int row,
                                             T scalar,
                                             T &effective_det) {
    matrix.multiply_row(row, scalar);
    effective_det *= scalar;
}

template<typename T>
void Matrix<T>::apply_numeric_multiplication(Matrix &matrix,
                                             int row,
                                             T scalar,
                                             T &effective_det) {
    matrix.multiply_row(row, scalar);
    effective_det *= scalar;
}

template<typename T>
void Matrix<T>::stabilize_integer_determinant(Matrix &matrix,
                                              int rows,
                                              int cols,
                                              T &effective_det) {
    constexpr T det_limit = 1000000000;
    constexpr T row_sum_limit = 10000;
    constexpr T scale_factor = 100;

    T abs_det = std::abs(effective_det);
    if (abs_det > det_limit) {
        for (int r = 0; r < rows; ++r) {
            T row_sum = T{0};
            for (int c = 0; c < cols; ++c) {
                row_sum += std::abs(matrix(r, c));
            }
            if (row_sum > row_sum_limit) {
                matrix.multiply_row(r, scale_factor);
                effective_det /= scale_factor;
                break;
            }
        }
    }
}

template<typename T> bool Matrix<T>::needs_stabilization(T effective_det) {
    if constexpr (detail::is_ordered_v<T>) {
        if constexpr (std::is_same_v<T, float>) {
            constexpr T limit = T{1e10f};
            T abs_det = std::abs(effective_det);
            return abs_det > limit;
        } else {
            constexpr T limit = T{1e100};
            T abs_det = std::abs(effective_det);
            return abs_det > limit;
        }
    }
    return false;
}

template<typename T>
bool Matrix<T>::is_safe_for_addition(const Matrix &matrix,
                                     int row1,
                                     int row2,
                                     int cols,
                                     T scalar,
                                     T min_val,
                                     T max_val) {
    if constexpr (detail::is_ordered_v<T>) {
        for (int c = 0; c < cols; ++c) {
            T new_val = matrix(row1, c) + scalar * matrix(row2, c);
            if (new_val < min_val || new_val > max_val) {
                return false;
            }
        }
    }
    return true;
}

template<typename T>
void Matrix<T>::stabilize_matrix(Matrix &matrix, int rows, int cols, T &effective_det) {
    static_assert(detail::is_ordered_v<T>, "stabilize_matrix requires ordered type");

    if constexpr (std::is_same_v<T, float>) {
        constexpr T threshold = T{1e-6f};
        constexpr T target = T{1e10f};

        for (int r = 0; r < rows; ++r) {
            T row_norm = T{0};
            for (int c = 0; c < cols; ++c) {
                row_norm += matrix(r, c) * matrix(r, c);
            }
            if (row_norm > threshold) {
                T factor = std::sqrt(target / std::abs(effective_det));
                if (factor < T{1}) {
                    factor = T{1};
                }
                matrix.multiply_row(r, factor);
                effective_det *= factor;
                break;
            }
        }
    } else {
        constexpr T threshold = T{1e-10};
        constexpr T target = T{1e20};

        for (int r = 0; r < rows; ++r) {
            T row_norm = T{0};
            for (int c = 0; c < cols; ++c) {
                row_norm += matrix(r, c) * matrix(r, c);
            }
            if (row_norm > threshold) {
                T factor = std::sqrt(target / std::abs(effective_det));
                if (factor < T{1}) {
                    factor = T{1};
                }
                matrix.multiply_row(r, factor);
                effective_det *= factor;
                break;
            }
        }
    }
}
