template<typename T> void Matrix<T>::alloc_matrix_() {
    if (rows_ <= 0 || cols_ <= 0) {
        matrix_ = nullptr;
        return;
    }

    matrix_ = std::make_unique<std::unique_ptr<T[]>[]>(rows_);
    for (int i = 0; i < rows_; ++i) {
        matrix_[i] = std::make_unique<T[]>(cols_);
    }
}

template<typename T> void Matrix<T>::init_zero_() {
    for (int i = 0; i < rows_; i++)
        for (int j = 0; j < cols_; j++)
            matrix_[i][j] = T{};
}

template<typename T> void Matrix<T>::fill_upper_triangle(T min_val, T max_val) {
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

template<typename T> std::optional<T> &Matrix<T>::get_determinant_() {
    return determinant_;
}

template<typename T> template<typename U> Matrix<U> Matrix<T>::cast_to() const {
    using NonRefU = std::remove_reference_t<U>;
    static_assert(!std::is_reference_v<NonRefU>, "Cannot cast to reference type");

    if constexpr (detail::is_matrix_v<T> && detail::is_matrix_v<NonRefU>) {
        Matrix<NonRefU> result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result(i, j) =
                    (*this)(i, j).template cast_to<typename NonRefU::value_type>();
            }
        }
        return result;
    } else if constexpr (detail::is_matrix_v<T> && !detail::is_matrix_v<NonRefU>) {
        throw std::runtime_error("Cannot cast block matrix to scalar matrix type");
    } else {
        Matrix<NonRefU> result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result(i, j) = static_cast<NonRefU>((*this)(i, j));
            }
        }
        return result;
    }
}

template<typename T>
template<typename U>
bool Matrix<T>::is_equal(const U &a, const U &b) {
    if constexpr (detail::is_matrix_v<U>) {
        if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) {
            return false;
        }
        for (int i = 0; i < a.get_rows(); ++i) {
            for (int j = 0; j < a.get_cols(); ++j) {
                if (!is_equal(a(i, j), b(i, j))) {
                    return false;
                }
            }
            return true;
        }
    } else {
        if constexpr (std::is_floating_point_v<U>) {
            return std::abs(a - b) < Epsilon;
        } else {
            return a == b;
        }
    }
}

template<typename T> template<typename U> bool Matrix<T>::is_zero(const U &value) {
    if constexpr (detail::is_matrix_v<U>) {
        auto det_opt = value.det();
        return !det_opt.has_value() || is_zero(*det_opt);
    } else if constexpr (std::is_same_v<U, float> || std::is_same_v<U, double>) {
        return std::abs(value) < Epsilon;
    } else {
        return value == U{};
    }
}

template<typename T> bool Matrix<T>::is_zero(int i, int j) const {
    return is_zero((*this)(i, j));
}

template<typename T>
template<typename U>
std::optional<int> Matrix<T>::find_pivot_in_subcol(int row, int col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        DEBUG_PRINTF("ERROR: index out of range\n");
        return std::nullopt;
    }

    if (rows_ == 0)
        return std::nullopt;

    int max_val_index = row;

    if constexpr (detail::has_abs_v<U> && !detail::is_matrix_v<U>) {
        using std::abs;
        auto max_abs = abs((*this)(row, col));

        for (int i = row + 1; i < rows_; ++i) {
            auto current_abs = abs((*this)(i, col));
            if (current_abs > max_abs) {
                max_val_index = i;
                max_abs = current_abs;
            }
        }
        return max_val_index;
    } else if constexpr (detail::is_matrix_v<U>) {
        double max_norm = compute_block_norm((*this)(row, col));

        for (int i = row + 1; i < rows_; ++i) {
            double current_norm = compute_block_norm((*this)(i, col));
            if (current_norm > max_norm) {
                max_val_index = i;
                max_norm = current_norm;
            }
        }
        return max_val_index;
    } else {
        for (int i = row; i < rows_; ++i) {
            if (!is_zero((*this)(i, col))) {
                return i;
            }
        }
        return std::nullopt;
    }
}

template<typename T> void Matrix<T>::swap_rows(int i, int j) {
    if (i != j) {
        std::swap(matrix_[i], matrix_[j]);

        if (determinant_)
            determinant_ = -*determinant_;
    }
}

template<typename T> void Matrix<T>::multiply_row(int target_row, T scalar) {
    // there is no scalar null check here intentionally
    // user must keep in mind that
    // it would't be an equivalent transformation
    for (int j = 0; j < cols_; ++j)
        matrix_[target_row][j] = matrix_[target_row][j] * scalar;

    if (determinant_)
        determinant_ = scalar * *determinant_;
}

template<typename T>
void Matrix<T>::add_row_scaled(int target_row, int source_row, T scalar) {
    for (int j = 0; j < cols_; ++j)
        matrix_[target_row][j] =
            matrix_[target_row][j] + matrix_[source_row][j] * scalar;
}

template<typename T> void Matrix<T>::print() const {
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++)
            std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                      << std::defaultfloat << matrix_[i][j] << ' ';

        std::cout << "\n\n";
    }
}

template<typename T> void Matrix<T>::print(int max_size) const {
    if (rows_ <= max_size && cols_ <= max_size) {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++)
                std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                          << std::defaultfloat << matrix_[i][j] << ' ';
            std::cout << "\n\n";
        }
    } else {
        std::cout << "Skipped printing (dimensions " << rows_ << "x" << cols_
                  << " exceed " << max_size << "x" << max_size << ").\n";
    }
}

template<typename T> void Matrix<T>::precise_print(int precision) const {
    int field_width = precision + 8;
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if constexpr (detail::is_matrix_v<T>) {
                if (rows_ <= 5 && cols_ <= 5 && (*this)(i, j).get_rows() <= 3
                    && (*this)(i, j).get_cols() <= 3) {
                    std::cout << "[" << (*this)(i, j).get_rows() << "x"
                              << (*this)(i, j).get_cols() << "] ";
                } else {
                    std::cout << "[SubMatrix] ";
                }
            } else {
                std::cout << std::setw(field_width) << std::scientific
                          << std::setprecision(precision) << matrix_[i][j] << " ";
            }
        }
        std::cout << "\n";
    }
}

template<typename T> void Matrix<T>::detailed_print() const {
    if constexpr (detail::is_matrix_v<T>) {
        std::cout << "Block Matrix " << rows_ << "x" << cols_ << " of "
                  << (*this)(0, 0).get_rows() << "x" << (*this)(0, 0).get_cols()
                  << " blocks:\n";

        std::cout << std::fixed << std::setprecision(2);

        for (int i = 0; i < rows_; ++i) {
            for (int inner_row = 0; inner_row < (*this)(0, 0).get_rows(); ++inner_row) {
                std::cout << "  ";
                for (int j = 0; j < cols_; ++j) {
                    const auto &block = (*this)(i, j);

                    std::cout << "[";
                    for (int inner_col = 0; inner_col < block.get_cols(); ++inner_col) {
                        std::cout << std::setw(6) << block(inner_row, inner_col);
                        if (inner_col < block.get_cols() - 1)
                            std::cout << " ";
                    }
                    std::cout << "]";

                    if (j < cols_ - 1)
                        std::cout << "  ";
                }
                std::cout << "\n";
            }
            if (i < rows_ - 1) {
                std::cout << "\n";
            }
        }
        std::cout << std::defaultfloat;
    } else {
        print(10);
    }
}

template<typename T>
template<typename U>
double Matrix<T>::compute_block_norm(const U &block) {
    using std::abs;

    if constexpr (detail::is_matrix_v<U>) {
        double norm = 0.0;
        for (int i = 0; i < block.get_rows(); ++i) {
            for (int j = 0; j < block.get_cols(); ++j) {
                double val = compute_block_norm(block(i, j));
                norm += val * val;
            }
        }
        return std::sqrt(norm);
    } else {
        try {
            return abs(block);
        } catch (...) {
            try {
                auto squared = block * block;
                if constexpr (std::is_convertible_v<decltype(squared), double>) {
                    return static_cast<double>(squared);
                } else {
                    return 0.0;
                }
            } catch (...) {
                return 0.0;
            }
        }
    }
}

template<typename T>
template<typename U>
bool Matrix<T>::is_element_zero(const U &elem) {
    using std::abs;

    if constexpr (detail::is_matrix_v<U>) {
        return compute_block_norm(elem) < Epsilon;
    } else {
        try {
            return abs(elem) < Epsilon;
        } catch (...) {
            try {
                return elem == U{};
            } catch (...) {
                return false;
            }
        }
    }
}

template<typename T>
template<typename U>
U Matrix<T>::identity_element(int rows, int cols) {
    if constexpr (detail::is_matrix_v<U>) {
        return U::Identity(rows, cols);
    } else {
        return U{1};
    }
}

template<typename T> template<typename U> U Matrix<T>::zero_element(int rows, int cols) {
    if constexpr (detail::is_matrix_v<U>) {
        return U::Zero(rows, cols);
    } else {
        return U{0};
    }
}
