#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include <map>
#include <iomanip>
#include <random>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include "Timer.hpp"
#include "Debug_printf.h"

enum class TransformationTypes {
    I_TYPE,
    II_TYPE,
    III_TYPE
};

template<typename T>
class Matrix_1d {
private:
    std::unique_ptr<T[]> matrix_;
    std::unique_ptr<T*[]> row_pointers_;
    int rows_, cols_, min_dim_;
    std::optional<T> determinant_ = std::nullopt;

    static constexpr auto epsilon = 1e-10;

    Matrix_1d(int rows, int cols) : rows_(rows), cols_(cols), min_dim_(std::min(rows, cols)) {
        alloc_matrix_();
    }

    bool alloc_matrix_() {
        if (rows_ <= 0 || cols_ <= 0) {
            DEBUG_PRINTF("ERROR: invalid dimensions\n");
            return false;
        }

        matrix_ = std::make_unique<T[]>(rows_ * cols_);
        row_pointers_ = std::make_unique<T*[]>(rows_);

        for (int i = 0; i < rows_; i++)
            row_pointers_[i] = &matrix_[i * cols_];

        return true;
    }

    void init_zero_() {
        for (int i = 0; i < rows_ * cols_; i++)
            matrix_[i] = T{};
    }

public:
    using value_type = T;

    std::optional<T> get_determinant() const { return determinant_; }
    // === Конструкторы ===
    static Matrix_1d Square(int size) {
        return Matrix_1d(size, size);
    }

    static Matrix_1d Rectangular(int rows, int cols) {
        return Matrix_1d(rows, cols);
    }

    static Matrix_1d Identity(int rows, int cols) {
        Matrix_1d result(rows, cols);
        int min_dim = result.get_min_dim();
        for (int i = 0; i < min_dim; i++)
            result(i, i) = T{1};

        return result;
    }

    static Matrix_1d Identity(int rows) {
        return Identity(rows, rows);
    }

    static Matrix_1d Diagonal(int rows, int cols, const std::vector<T>& diagonal) {
        Matrix_1d result = Matrix_1d::Zero(static_cast<int>(rows), static_cast<int>(cols));
        int diag_size = std::min(result.get_min_dim(), static_cast<int>(diagonal.size()));
        for (int i = 0; i < diag_size; i++)
            result(i, i) = diagonal[i];

        return result;
    }

    static Matrix_1d Diagonal(const std::vector<T>& diagonal) {
        int min_dim = diagonal.size();
        Matrix_1d result(min_dim, min_dim);
        for (int i = 0; i < min_dim; i++)
            result(i, i) = diagonal[i];

        return result;
    }

    static Matrix_1d From_vector(const std::vector<std::vector<T>>& input) {
        if (input.empty()) {
            return Matrix_1d(0, 0);
        }

        size_t max_cols = 0;
        for (const auto& row : input) {
            max_cols = std::max(max_cols, row.size());
        }

        size_t rows = input.size();
        Matrix_1d result = Matrix_1d::Zero(static_cast<int>(rows), static_cast<int>(max_cols));

        for (size_t i = 0; i < rows; i++) {
            const auto& current_row = input[i];
            size_t current_cols = current_row.size();

            for (size_t j = 0; j < current_cols; j++)
                result(static_cast<int>(i), static_cast<int>(j)) = current_row[j];
        }

        return result;
    }

    static Matrix_1d Zero(int rows, int cols) {
        Matrix_1d result(rows, cols);
        result.init_zero_();
        return result;
    }

    static Matrix_1d Zero(int rows) {
        return Zero(rows, rows);
    }

    static Matrix_1d Read_vector() {
        int n;
        std::cin >> n;

        Matrix_1d matrix(n, n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cin >> matrix(i, j);
            }
        }

        return matrix;
    }

    static Matrix_1d Generate_matrix(int rows, int cols,
        std::conditional_t<std::is_same_v<T, int>, int, T> min_val = {},
        std::conditional_t<std::is_same_v<T, int>, int, T> max_val = {},
        int iterations = 100,
        T target_determinant_magnitude = T{1}
    ) {
        if constexpr (std::is_floating_point_v<T>) {
            if (min_val == T{} && max_val == T{}) {
                min_val = T{-10};
                max_val = T{10};
            }
        }

        std::vector<T> diagonal = create_controlled_diagonal(std::min(rows, cols), min_val, max_val, target_determinant_magnitude);
        Matrix_1d result(rows, cols);
        result.fill_diagonal(diagonal);
        result.fill_upper_triangle(min_val, max_val);

        apply_controlled_transformations(result, rows, cols, min_val, max_val, iterations);

        return result;
    }

    static std::vector<T> create_controlled_diagonal(int size,
        std::conditional_t<std::is_same_v<T, int>, int, T> min_val,
        std::conditional_t<std::is_same_v<T, int>, int, T> max_val,
        T target_determinant_magnitude
    ) {
        std::vector<T> diagonal(size);
        T current_det = T{1};

        for (int i = 0; i < size; ++i) {
            T rand_element = generate_random(min_val, max_val);

            if constexpr (std::is_floating_point_v<T>) {
                T scale_factor = std::clamp(std::abs(rand_element), T{0.1}, T{10.0});
                T sign = rand_element >= T{0} ? T{1} : T{-1};
                rand_element = sign * scale_factor;
            }

            if (i == size - 1) {
                if (current_det != T{0}) {
                    rand_element = (target_determinant_magnitude / current_det);

                    if constexpr (std::is_floating_point_v<T>) {
                        T abs_elem = std::abs(rand_element);
                        if (abs_elem > T{1e50}) {
                            rand_element = std::copysign(T{1e50}, rand_element);
                        } else if (abs_elem < T{1e-50}) {
                            rand_element = std::copysign(T{1e-50}, rand_element);
                        }
                    }
                }
            }

            diagonal[i] = rand_element;
            current_det *= rand_element;
        }

        return diagonal;
    }

    static void apply_controlled_transformations(Matrix_1d& matrix, int rows, int cols,
        std::conditional_t<std::is_same_v<T, int>, int, T> min_val,
        std::conditional_t<std::is_same_v<T, int>, int, T> max_val,
        int iterations
    ) {
        T effective_det = T{1};
        T sign = T{1};

        for (int i = 0; i < iterations; ++i) {
            TransformationTypes rand_transformation = static_cast<TransformationTypes>(
                generate_random_int_(0, 2));

            switch (rand_transformation) {
                case TransformationTypes::I_TYPE: {
                    int row_1 = generate_random_int_(0, rows - 1);
                    int row_2 = generate_random_int_(0, rows - 1);
                    if (row_1 != row_2) {
                        matrix.swap_rows(row_1, row_2);
                        sign = -sign;
                    }
                    break;
                }

                case TransformationTypes::II_TYPE: {
                    int row = generate_random_int_(0, rows - 1);
                    T scalar = generate_random(min_val, max_val);

                    if (!matrix.is_zero(scalar)) {
                        if constexpr (std::is_floating_point_v<T>) {
                            scalar = std::clamp(scalar, T{-10.0}, T{10.0});
                            if (std::abs(scalar) < T{1e-10}) {
                                scalar = std::copysign(T{1e-10}, scalar);
                            }
                        }

                        matrix.multiply_row(row, scalar);
                        effective_det *= scalar;

                        if constexpr (std::is_floating_point_v<T>) {
                            T abs_det = std::abs(effective_det);
                            if (abs_det > T{1e100}) {
                                matrix.stabilize_for_large_determinant(rows, cols, effective_det);
                            }
                        }
                    }
                    break;
                }

                case TransformationTypes::III_TYPE: {
                    int row_1 = generate_random_int_(0, rows - 1);
                    int row_2 = generate_random_int_(0, rows - 1);
                    if (row_1 != row_2) {
                        T scalar = generate_random(min_val, max_val);

                        if constexpr (std::is_floating_point_v<T>) {
                            scalar = std::clamp(scalar, T{-10.0}, T{10.0});
                        }

                        matrix.add_row_scaled(row_1, row_2, scalar);
                    }
                    break;
                }
            }
        }

        matrix.determinant_ = sign * effective_det;
    }

    void fill_upper_triangle(
        std::conditional_t<std::is_same_v<T, int>, int, T> min_val = {},
        std::conditional_t<std::is_same_v<T, int>, int, T> max_val = {}
    ) {
        for (int i = 0; i < min_dim_; ++i) {
            for (int j = i + 1; j < cols_; ++j) {
                T val = generate_random(min_val, max_val);

                if constexpr (std::is_floating_point_v<T>) {
                    val = std::clamp(val, T{-10.0}, T{10.0});
                }

                (*this)(i, j) = val;
            }
        }
    }

    void stabilize_for_large_determinant(int rows, int cols, T& effective_det) {
        for (int r = 0; r < rows; ++r) {
            T row_norm = T{0};
            for (int c = 0; c < cols; ++c) {
                row_norm += (*this)(r, c) * (*this)(r, c);
            }
            if (row_norm > T{1e-10}) {
                T factor = std::sqrt(T{1e20} / std::abs(effective_det));
                if (factor < T{1}) factor = T{1};
                multiply_row(r, factor);
                effective_det *= factor;
                break;
            }
        }
    }

    Matrix_1d(const Matrix_1d& rhs) : rows_(rhs.rows_), cols_(rhs.cols_), min_dim_(rhs.min_dim_) {
        alloc_matrix_();

        for (int i = 0; i < rows_ * cols_; i++)
            matrix_[i] = rhs.matrix_[i];
    }

    Matrix_1d& operator=(const Matrix_1d& rhs) {
        if (this != &rhs) {
            rows_ = rhs.get_rows();
            cols_ = rhs.get_cols();
            min_dim_ = rhs.get_min_dim();
            alloc_matrix_();

            for (int i = 0; i < rows_ * cols_; i++)
                matrix_[i] = rhs.matrix_[i];
        }

        return *this;
    }

    Matrix_1d(Matrix_1d&&) = default;
    Matrix_1d& operator=(Matrix_1d&&) = default;

    T& operator()(int i, int j) { return row_pointers_[i][j]; }
    const T& operator()(int i, int j) const { return row_pointers_[i][j]; }

    int get_rows() const { return rows_; }
    int get_cols() const { return cols_; }
    int get_min_dim() const { return min_dim_; }

    // === Исправленные is_zero ===
    static bool is_zero(T val) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(val) < epsilon;
        } else {
            return val == T{0};
        }
    }

    void fill_diagonal(const std::vector<T>& diag) {
        for (int i = 0; i < min_dim_; ++i) {
            (*this)(i, i) = diag[i];
        }
    }

    void swap_rows(int i, int j) {
        if (i != j) {
            for (int c = 0; c < cols_; ++c) {
                std::swap((*this)(i, c), (*this)(j, c));
            }
        }
    }

    void multiply_row(int target_row, T scalar) {
        for (int j = 0; j < cols_; ++j)
            (*this)(target_row, j) = (*this)(target_row, j) * scalar;
    }

    void add_row_scaled(int target_row, int source_row, T scalar = T{1}) {
        for (int j = 0; j < cols_; ++j)
            (*this)(target_row, j) = (*this)(target_row, j) + (*this)(source_row, j) * scalar;
    }

    void print() const {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                T value = (*this)(i, j);
                if (value == static_cast<long long>(value)) {
                    std::cout << std::setw(8) << static_cast<long long>(value);
                } else {
                    std::cout << std::setw(8) << std::defaultfloat << value;
                }
                std::cout << ' ';
            }
            std::cout << std::endl;
        }
    }

    void precise_print(int precision = 15) const {
        int field_width = precision + 8;
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                std::cout << std::setw(field_width) << std::scientific
                          << std::setprecision(precision) << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
    }

    std::optional<int> find_max_in_subcol(int row, int col) const {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            DEBUG_PRINTF("ERROR: index out of range\n");
            return std::nullopt;
        }

        if (rows_ == 0)
            return std::nullopt;

        int max_val_index = row;
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
    }

    std::optional<int> find_max_in_subrow(int row, int col) const {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            DEBUG_PRINTF("ERROR: index out of range\n");
            return std::nullopt;
        }

        if (cols_ == 0)
            return std::nullopt;

        int max_val_index = col;
        using std::abs;
        auto max_abs = abs((*this)(row, col));

        for (int j = col + 1; j < cols_; ++j) {  
            auto current_abs = abs((*this)(row, j)); 
            if (current_abs > max_abs) {
                max_val_index = j; 
                max_abs = current_abs;
            }
        }

        return max_val_index;
    }

    // Вспомогательная функция для подсчёта инверсий через merge sort
    int merge_and_count(std::vector<int>& arr, int left, int mid, int right) {
        std::vector<int> temp(right - left + 1);
        int i = left, j = mid + 1, k = 0, inv_count = 0;

        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
                inv_count += (mid - i + 1);
            }
        }

        while (i <= mid) temp[k++] = arr[i++];
        while (j <= right) temp[k++] = arr[j++];

        for (int idx = left; idx <= right; ++idx) {
            arr[idx] = temp[idx - left];
        }

        return inv_count;
    }

    int merge_sort_and_count(std::vector<int>& arr, int left, int right) {
        int inv_count = 0;
        if (left < right) {
            int mid = left + (right - left) / 2;
            inv_count += merge_sort_and_count(arr, left, mid);
            inv_count += merge_sort_and_count(arr, mid + 1, right);
            inv_count += merge_and_count(arr, left, mid, right);
        }
        return inv_count;
    }

    int count_inversions(std::vector<int> perm) {
        return merge_sort_and_count(perm, 0, perm.size() - 1);
    }

    std::optional<T> det(int row, int col, int size) {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_ ||
            row + size > rows_ || col + size > cols_ || size <= 0) {
            return std::nullopt;
        }

        T determinant = T{1};
        Matrix_1d matrix_cpy = *this;

        std::vector<bool> used_rows(size, false);
        std::vector<int> pivot_rows;
        pivot_rows.reserve(size);

        for (int j = 0; j < size; ++j) {
            int current_col = col + j;

            int best_row = -1;
            T max_val = T{};

            for (int r = row; r < row + size; ++r) {
                if (!used_rows[r - row]) {
                    T abs_val = std::abs(matrix_cpy(r, current_col));
                    if (abs_val > max_val) {
                        max_val = abs_val;
                        best_row = r;
                    }
                }
            }

            if (best_row == -1) {
                return T{};
            }

            pivot_rows.push_back(best_row);
            used_rows[best_row - row] = true;

            T pivot = matrix_cpy(best_row, current_col);
            determinant = determinant * pivot;

            if constexpr (std::is_floating_point_v<T>) {
                T abs_det = std::abs(determinant);
                if (abs_det > T{1e50} || abs_det < T{1e-50}) {
                    int order = static_cast<int>(std::floor(std::log10(abs_det)));
                    T normalized = determinant / std::pow(T{10}, T(order));
                    determinant = normalized;
                }
            }

            if (j == size - 1) continue;

            for (int i = row; i < row + size; ++i) {
                if (used_rows[i - row]) continue;

                T coeff = matrix_cpy(i, current_col) / pivot;

                for (int c = current_col + 1; c < col + size; ++c) {
                    matrix_cpy(i, c) = matrix_cpy(i, c) - coeff * matrix_cpy(best_row, c);
                }
            }
        }

        int inversions = count_inversions(pivot_rows);
        if (inversions % 2 == 1) {
            determinant = determinant * T{-1};
        }

        return determinant;
    }

    std::optional<T> det() {
        return det(0, 0, min_dim_);
    }

private:
    static int generate_random_int_(int min = 1, int max = 100) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(min, max);
        return dis(gen);
    }

    static T generate_random(
        std::conditional_t<std::is_same_v<T, int>, int, T> min_val = {},
        std::conditional_t<std::is_same_v<T, int>, int, T> max_val = {}
    ) {
        if constexpr (std::is_same_v<T, int>) {
            int actual_min = static_cast<int>(min_val);
            int actual_max = static_cast<int>(max_val);
            if (actual_min == 0 && actual_max == 0) {
                actual_min = 1;
                actual_max = 100;
            }
            return generate_random_int_(actual_min, actual_max);
        }
        else if constexpr (std::is_floating_point_v<T>) {
            double actual_min = static_cast<double>(min_val);
            double actual_max = static_cast<double>(max_val);
            if (actual_min == 0.0 && actual_max == 0.0) {
                actual_min = -10.0;
                actual_max = 10.0;
            }
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(actual_min, actual_max);
            return static_cast<T>(dis(gen));
        }
        else {
            static_assert(std::is_constructible_v<T, int> || std::is_constructible_v<T, double>,
                          "T must be constructible from int or double");
        }
    }
};
