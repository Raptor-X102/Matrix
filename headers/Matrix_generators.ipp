template<typename T>
Matrix<T> Matrix<T>::Generate_matrix(int rows,
                                     int cols,
                                     T min_val,
                                     T max_val,
                                     int iterations,
                                     T target_determinant_magnitude,
                                     T max_condition_number) {
    // Generate matrix of matrices
    if constexpr (detail::is_matrix_v<T>) {
        Matrix result(rows, cols);

        int inner_rows = min_val.get_rows();
        int inner_cols = min_val.get_cols();

        using InnerType = typename T::value_type;

        InnerType inner_min, inner_max;

        inner_min = static_cast<InnerType>(-10);
        inner_max = static_cast<InnerType>(10);

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
    } else {
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

        std::vector<T> diagonal =
            create_controlled_diagonal(std::min(rows, cols),
                                       min_val,
                                       max_val,
                                       target_determinant_magnitude);
        Matrix result = Matrix::Diagonal(rows, cols, diagonal);
        result.fill_upper_triangle(min_val, max_val);

        apply_controlled_transformations(result,
                                         rows,
                                         cols,
                                         min_val,
                                         max_val,
                                         iterations);

        return result;
    }
}

template<typename T>
Matrix<T> Matrix<T>::Generate_binary_matrix(int rows,
                                            int cols,
                                            T target_determinant_magnitude,
                                            int iterations) {
    int min_dim = std::min(rows, cols);
    std::vector<T> diagonal(min_dim);

    T current_det = T{1};
    for (int i = 0; i < min_dim; ++i) {
        T sign = generate_random_int_(0, 1) == 0 ? T{-1} : T{1};
        diagonal[i] = sign;
        current_det *= sign;
    }

    Matrix result = Matrix::Diagonal(rows, cols, diagonal);

    for (int i = 0; i < min_dim; ++i) {
        for (int j = i + 1; j < cols; ++j) {
            int val = generate_random_int_(-1, 1);
            result(i, j) = static_cast<T>(val);
        }
    }

    T effective_det = current_det;
    T sign = T{1};

    for (int i = 0; i < iterations; ++i) {
        Tranformation_types rand_transformation =
            static_cast<Tranformation_types>(generate_random_int_(0, 2));

        switch (rand_transformation) {
        case Tranformation_types::I_TYPE: {
            int row_1 = generate_random_int_(0, rows - 1);
            int row_2 = generate_random_int_(0, rows - 1);
            if (row_1 != row_2) {
                result.swap_rows(row_1, row_2);
                sign = -sign;
            }
            break;
        }

        case Tranformation_types::II_TYPE: {
            int row = generate_random_int_(0, rows - 1);
            T scalar = generate_random_int_(0, 1) == 0 ? T{-1} : T{1};

            result.multiply_row(row, scalar);
            effective_det *= scalar;
            break;
        }

        case Tranformation_types::III_TYPE: {
            int row_1 = generate_random_int_(0, rows - 1);
            int row_2 = generate_random_int_(0, rows - 1);
            if (row_1 != row_2) {
                T scalar = generate_random_int_(0, 1) == 0 ? T{-1} : T{1};

                bool safe_operation = true;
                for (int c = 0; c < cols; ++c) {
                    T new_val = result(row_1, c) + scalar * result(row_2, c);
                    if (new_val < T{-1} || new_val > T{1}) {
                        safe_operation = false;
                        break;
                    }
                }

                if (safe_operation) {
                    result.add_row_scaled(row_1, row_2, scalar);
                }
            }
            break;
        }
        }
    }

    result.determinant_ = sign * effective_det;
    return result;
}

template<typename T>
std::vector<T> Matrix<T>::create_controlled_diagonal(int size,
                                                     T min_val,
                                                     T max_val,
                                                     T target_determinant_magnitude) {
    std::vector<T> diagonal(size);

    if (size == 0)
        return diagonal;

    T current_det = T{1};

    if constexpr (std::is_integral_v<T>) {
        for (int i = 0; i < size - 1; ++i) {
            T rand_element;
            do {
                rand_element = generate_random(min_val, max_val);
            } while (rand_element == T{0});

            diagonal[i] = rand_element;
            current_det *= rand_element;
        }

        if (current_det != T{0}) {
            T best_last = T{1};
            T best_diff = std::abs(target_determinant_magnitude - current_det);

            for (T test_val = min_val; test_val <= max_val; ++test_val) {
                if (test_val == T{0})
                    continue;

                T test_product = current_det * test_val;
                T diff = std::abs(target_determinant_magnitude - test_product);

                if (diff < best_diff) {
                    best_diff = diff;
                    best_last = test_val;
                }
            }

            diagonal[size - 1] = best_last;
        } else {
            diagonal[size - 1] = generate_random(min_val, max_val);
        }
    } else {
        for (int i = 0; i < size; ++i) {
            T rand_element = generate_random(min_val, max_val);

            if constexpr (detail::is_ordered_v<T>) {
                T abs_elem = std::abs(rand_element);
                T scale_factor = std::clamp(abs_elem, T{0.1}, T{10.0});
                T sign = (rand_element >= T{0}) ? T{1} : T{-1};
                rand_element = sign * scale_factor;
            }

            if (i == size - 1 && current_det != T{0}) {
                rand_element = (target_determinant_magnitude / current_det);

                if constexpr (detail::is_ordered_v<T>) {
                    T abs_elem = std::abs(rand_element);
                    if (abs_elem > static_cast<T>(1e50)) {
                        rand_element = (rand_element >= T{0}) ? static_cast<T>(1e50)
                                                              : static_cast<T>(-1e50);
                    } else if (abs_elem < static_cast<T>(1e-50)) {
                        rand_element = (rand_element >= T{0}) ? static_cast<T>(1e-50)
                                                              : static_cast<T>(-1e-50);
                    }
                }
            }

            diagonal[i] = rand_element;
            current_det *= rand_element;
        }
    }

    return diagonal;
}

template<typename T>
void Matrix<T>::apply_transformation_type_I(Matrix &matrix, int rows) {
    int row_1 = generate_random_int_(0, rows - 1);
    int row_2 = generate_random_int_(0, rows - 1);
    if (row_1 != row_2) {
        matrix.swap_rows(row_1, row_2);
    }
}

template<typename T>
void Matrix<T>::apply_transformation_type_II(Matrix &matrix,
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
        if (std::abs(scalar) > T{100}) {
            scalar = (scalar >= T{0}) ? T{100} : T{-100};
        }

        bool safe_operation = true;
        for (int c = 0; c < cols; ++c) {
            T new_val = matrix(row, c) * scalar;
            if (new_val < min_val || new_val > max_val) {
                safe_operation = false;
                break;
            }
        }

        if (safe_operation) {
            matrix.multiply_row(row, scalar);
            effective_det *= scalar;

            T abs_det = std::abs(effective_det);
            if (abs_det > T{1000000000}) {
                for (int r = 0; r < rows; ++r) {
                    T row_sum = T{0};
                    for (int c = 0; c < cols; ++c) {
                        row_sum += std::abs(matrix(r, c));
                    }
                    if (row_sum > T{10000}) {
                        matrix.multiply_row(r, T{100});
                        effective_det /= T{100};
                        break;
                    }
                }
            }
        }
    } else {
        if constexpr (detail::is_ordered_v<T>) {
            scalar = std::clamp(scalar, T{-10.0}, T{10.0});
            if (std::abs(scalar) < T{1e-10}) {
                scalar = (scalar >= T{0}) ? T{1e-10} : T{-1e-10};
            }
        }

        bool safe_operation = true;
        if constexpr (detail::is_ordered_v<T>) {
            for (int c = 0; c < cols; ++c) {
                T new_val = matrix(row, c) * scalar;
                if (new_val < min_val || new_val > max_val) {
                    safe_operation = false;
                    break;
                }
            }
        }

        if (safe_operation) {
            matrix.multiply_row(row, scalar);
            effective_det *= scalar;

            if constexpr (detail::is_ordered_v<T>) {
                T abs_det = std::abs(effective_det);
                if (abs_det > static_cast<T>(1e100)) {
                    stabilize_matrix(matrix, rows, cols, effective_det);
                }
            }
        }
    }
}

template<typename T>
void Matrix<T>::apply_transformation_type_III(Matrix &matrix,
                                              int rows,
                                              int cols,
                                              T min_val,
                                              T max_val) {
    int row_1 = generate_random_int_(0, rows - 1);
    int row_2 = generate_random_int_(0, rows - 1);
    if (row_1 == row_2)
        return;

    T scalar = generate_random(min_val, max_val);

    if constexpr (std::is_same_v<T, int>) {
        if (std::abs(scalar) > T{100}) {
            scalar = (scalar >= T{0}) ? T{100} : T{-100};
        }
    } else {
        if constexpr (detail::is_ordered_v<T>) {
            scalar = std::clamp(scalar, T{-10.0}, T{10.0});
        }
    }

    bool safe_operation = true;
    if constexpr (detail::is_ordered_v<T>) {
        for (int c = 0; c < cols; ++c) {
            T new_val = matrix(row_1, c) + scalar * matrix(row_2, c);
            if (new_val < min_val || new_val > max_val) {
                safe_operation = false;
                break;
            }
        }
    }

    if (safe_operation) {
        matrix.add_row_scaled(row_1, row_2, scalar);
    }
}

template<typename T>
void Matrix<T>::apply_controlled_transformations(Matrix &matrix,
                                                 int rows,
                                                 int cols,
                                                 T min_val,
                                                 T max_val,
                                                 int iterations) {
    T effective_det = T{1};
    int sign = 1;

    T initial_det = T{1};
    int min_dim = std::min(rows, cols);
    for (int i = 0; i < min_dim; ++i) {
        initial_det *= matrix(i, i);
    }
    effective_det = initial_det;

    for (int i = 0; i < iterations; ++i) {
        Tranformation_types rand_transformation =
            static_cast<Tranformation_types>(generate_random_int_(0, 2));

        switch (rand_transformation) {
        case Tranformation_types::I_TYPE: {
            apply_transformation_type_I(matrix, rows);
            sign = -sign;
            break;
        }

        case Tranformation_types::II_TYPE: {
            apply_transformation_type_II(matrix,
                                         rows,
                                         cols,
                                         min_val,
                                         max_val,
                                         effective_det);
            break;
        }

        case Tranformation_types::III_TYPE: {
            apply_transformation_type_III(matrix, rows, cols, min_val, max_val);
            break;
        }
        }
    }

    matrix.determinant_ = (sign == 1) ? effective_det : -effective_det;
}

template<typename T>
void Matrix<T>::stabilize_matrix(Matrix &matrix, int rows, int cols, T &effective_det) {
    static_assert(detail::is_ordered_v<T>, "stabilize_matrix requires ordered type");

    for (int r = 0; r < rows; ++r) {
        T row_norm = T{0};
        for (int c = 0; c < cols; ++c) {
            row_norm += matrix(r, c) * matrix(r, c);
        }
        if (row_norm > static_cast<T>(1e-10)) {
            T factor = std::sqrt(static_cast<T>(1e20) / std::abs(effective_det));
            if (factor < T{1}) {
                factor = T{1};
            }
            matrix.multiply_row(r, factor);
            effective_det *= factor;
            break;
        }
    }
}
