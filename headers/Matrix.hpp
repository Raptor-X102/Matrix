#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include <iomanip>
#include <random>
#include <type_traits>
#include <algorithm>
#include <limits>
#include <cmath>

#ifdef __GNUC__
#   include <x86intrin.h>
#elif defined(_MSC_VER)
#   include <intrin.h>
#else
#   include <immintrin.h>
#endif

#include "Debug_printf.h"

namespace detail {

template <typename T, typename = void>
struct has_division : std::false_type {};
template <typename T>
struct has_division<T, std::void_t<decltype(std::declval<T>() / std::declval<T>())>>
    : std::true_type {};
template <typename T>
constexpr bool has_division_v = has_division<T>::value;

template <typename T, typename = void>
struct has_abs : std::false_type {};
template <typename T>
struct has_abs<T, std::void_t<decltype(std::abs(std::declval<T>()))>>
    : std::true_type {};
template <typename T>
constexpr bool has_abs_v = has_abs<T>::value;

template <typename T>
constexpr bool is_ordered_v = std::is_arithmetic_v<T>;

template <typename T>
constexpr bool is_builtin_integral_v =
    std::is_same_v<T, int> || std::is_same_v<T, long> ||
    std::is_same_v<T, long long> || std::is_same_v<T, unsigned int> ||
    std::is_same_v<T, unsigned long> || std::is_same_v<T, unsigned long long> ||
    std::is_same_v<T, short> || std::is_same_v<T, unsigned short> ||
    std::is_same_v<T, char> || std::is_same_v<T, signed char> ||
    std::is_same_v<T, unsigned char>;

}

const int Default_iterations = 100;

struct BasicAlgorithm {};
struct AvxAlgorithm {};

template <typename T> class Matrix {
private:
    enum class Tranformation_types { I_TYPE = 0, II_TYPE = 1, III_TYPE = 2 };

    std::unique_ptr<std::unique_ptr<T[]>[]> matrix_;
    int rows_, cols_, min_dim_;
    std::optional<T> determinant_ = std::nullopt;

    static constexpr auto epsilon = 1e-10;

public:
    using value_type = T;

    Matrix(int rows, int cols) : rows_(rows), cols_(cols), min_dim_(std::min(rows, cols)) {
        alloc_matrix_();
    }

    // Submatrix constructor
    Matrix(const Matrix& other, int start_row, int start_col, int num_rows, int num_cols)
        : rows_(num_rows), cols_(num_cols), min_dim_(std::min(num_rows, num_cols)) {
        alloc_matrix_();
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                (*this)(i, j) = other(start_row + i, start_col + j);
            }
        }
    }

    static Matrix Square(int size) { return Matrix(size, size); }

    static Matrix Rectangular(int rows, int cols) { return Matrix(rows, cols); }

    static Matrix Identity(int rows, int cols) {
        Matrix result(rows, cols);
        int min_dim = result.get_min_dim();
        for (int i = 0; i < min_dim; i++)
            result(i, i) = T{1};

        return result;
    }

    static Matrix Identity(int rows) { return Identity(rows, rows); }

    static Matrix Diagonal(int rows, int cols, const std::vector<T> &diagonal) {
        Matrix result = Matrix::Zero(static_cast<int>(rows), static_cast<int>(cols));
        int diag_size = std::min(result.get_min_dim(), static_cast<int>(diagonal.size()));
        for (int i = 0; i < diag_size; i++)
            result.matrix_[i][i] = diagonal[i];

        return result;
    }

    static Matrix Diagonal(const std::vector<T> &diagonal) {
        int min_dim = diagonal.size();
        Matrix result(min_dim);
        for (int i = 0; i < min_dim; i++)
            result.matrix_[i][i] = diagonal[i];

        return result;
    }

    static Matrix Diagonal(int rows, int cols, T diagonal_value) {
        Matrix result(rows, cols);
        int min_dim = std::min(rows, cols);
        for (int i = 0; i < min_dim; i++)
            result(i, i) = diagonal_value;

        return result;
    }

    static Matrix Diagonal(int size, T diagonal_value) {
        return Diagonal(size, size, diagonal_value);
    }

    static Matrix From_vector(const std::vector<std::vector<T>> &input) {
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

    static Matrix Zero(int rows, int cols) {
        Matrix result(rows, cols);
        result.init_zero_();
        return result;
    }

    static Matrix Zero(int rows) { return Zero(rows, rows); }

    static Matrix Read_vector() {
        int n;
        std::cin >> n;

        Matrix matrix(n, n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cin >> matrix(i, j);
            }
        }

        return matrix;
    }

    static Matrix Generate_matrix(int rows, int cols,
        T min_val = {},
        T max_val = {},
        int iterations = Default_iterations, 
        T target_determinant_magnitude = T{1},
        T max_condition_number = {}) {
        
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

        std::vector<T> diagonal = create_controlled_diagonal(
            std::min(rows, cols), min_val, max_val, target_determinant_magnitude);
        Matrix result = Matrix::Diagonal(rows, cols, diagonal);
        result.fill_upper_triangle(min_val, max_val);

        apply_controlled_transformations(result, rows, cols, min_val, max_val, iterations);

        return result;
    }

    static Matrix Generate_binary_matrix(int rows, int cols,
        T target_determinant_magnitude = T{1},
        int iterations = Default_iterations / 10) {
        
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

    Matrix(const Matrix &rhs) : rows_(rhs.rows_), cols_(rhs.cols_) {
        alloc_matrix_();

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                matrix_[i][j] = rhs.matrix_[i][j];
    }

    Matrix &operator=(const Matrix &rhs) {
        if (this != &rhs) {
            rows_ = rhs.get_rows(); 
            cols_ = rhs.get_cols();
            min_dim_ = rhs.get_min_dim();
            alloc_matrix_();

            for (int i = 0; i < rows_; i++)
                for (int j = 0; j < cols_; j++)
                    matrix_[i][j] = rhs.matrix_[i][j];
        }

        return *this;
    }

    Matrix(Matrix &&) = default;
    Matrix &operator=(Matrix &&) = default;

    T &operator()(int i, int j) { return matrix_[i][j]; }
    const T &operator()(int i, int j) const { return matrix_[i][j]; }

    friend Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
        return lhs.binary_operation(rhs, std::plus<T>{});
    }

    friend Matrix operator-(const Matrix& lhs, const Matrix& rhs) {
        return lhs.binary_operation(rhs, std::minus<T>{});
    }

    Matrix& operator+=(const Matrix& other) {
        *this = binary_operation(other, std::plus<T>{});
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        *this = binary_operation(other, std::minus<T>{});
        return *this;
    }
    
    friend Matrix operator+(const Matrix& lhs, const T& scalar) {
        Matrix result = lhs;
        result += scalar;
        return result;
    }

    friend Matrix operator+(const T& scalar, const Matrix& rhs) {
        Matrix result = rhs;
        result += scalar;
        return result;
    }

    Matrix& operator+=(const T scalar) {
        int min_dim = std::min(rows_, cols_);
        for (int i = 0; i < min_dim; ++i) {
            (*this)(i, i) += scalar;
        }
        return *this;
    }
    
    Matrix& operator-=(const T scalar) {
        int min_dim = std::min(rows_, cols_);
        for (int i = 0; i < min_dim; ++i) {
            (*this)(i, i) -= scalar;
        }
        return *this;
    }
    
    friend Matrix operator-(const Matrix& lhs, const T& scalar) {
        Matrix result = lhs;
        result -= scalar;
        return result;
    }

    friend Matrix operator*(const Matrix& A, const Matrix& B) {
        #ifdef __AVX__
            if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
                 return multiply_impl<AvxAlgorithm>(A, B);
            } else {
                 return multiply_impl<BasicAlgorithm>(A, B);
            }
        #else
            return multiply_impl<BasicAlgorithm>(A, B);
        #endif
    }

    friend Matrix multiply_basic(const Matrix& A, const Matrix& B) {
        return multiply_impl<BasicAlgorithm>(A, B);
    }

    #ifdef __AVX__
    friend Matrix multiply_avx(const Matrix& A, const Matrix& B) {
        return multiply_impl<AvxAlgorithm>(A, B);
    }
    #endif

    friend Matrix operator*(const Matrix& lhs, const T& scalar) {
        Matrix result = lhs;
        result *= scalar;
        return result;
    }

    friend Matrix operator*(const T& scalar, const Matrix& rhs) {
        return rhs * scalar;
    }

    Matrix& operator*=(const T& scalar) {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                (*this)(i, j) *= scalar;
            }
        }
        return *this;
    }

    int get_rows() const { return rows_; }
    int get_cols() const { return cols_; }
    int get_min_dim() const { return min_dim_; }
    std::optional<T> get_determinant() const { return determinant_; }

    static bool is_zero(const T &value) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return std::abs(value) < epsilon;
        } else {
            return value == T{};
        }
    }

    bool is_zero(int i, int j) const { return is_zero((*this)(i, j)); }

    /********** 3 types of matrix transformation ***********/
    void swap_rows(int i, int j) { // I.

        if (i != j) {
            std::swap(matrix_[i], matrix_[j]);

            if (determinant_)
                determinant_ = T{-1} * *determinant_;
        }
    }

    void multiply_row(int target_row, T scalar) { // II.
        // there is no scalar null check here intentionally
        // user must keep in mind that
        // it would't be an equivalent transformation
        for (int j = 0; j < cols_; ++j)
            matrix_[target_row][j] = matrix_[target_row][j] * scalar;

        if (determinant_)
            determinant_ = scalar * *determinant_;
    }

    void add_row_scaled(int target_row, int source_row, T scalar = T{1}) { // III.

        for (int j = 0; j < cols_; ++j)
            matrix_[target_row][j] = matrix_[target_row][j] + matrix_[source_row][j] * scalar;
    }

    void print() const {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++)
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << std::defaultfloat
                          << matrix_[i][j] << ' ';

            std::cout << "\n\n";
        }
    }

    void print(int max_size) const {
        if (rows_ <= max_size && cols_ <= max_size) {
            for (int i = 0; i < rows_; i++) {
                for (int j = 0; j < cols_; j++)
                    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << std::defaultfloat
                              << matrix_[i][j] << ' ';
                std::cout << "\n\n";
            }
        } else {
            std::cout << "Skipped printing (dimensions " << rows_ << "x" << cols_ << " exceed " << max_size << "x" << max_size << ").\n";
        }
    }

    void precise_print(int precision = 15) const {
        int field_width = precision + 8;

        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                std::cout << std::setw(field_width) << std::scientific
                          << std::setprecision(precision) << matrix_[i][j] << " ";
            }

            std::cout << "\n";
        }
    }

    void fill_upper_triangle(
        T min_val = {},
        T max_val = {}
    ) {
        for (int i = 0; i < min_dim_; ++i) {
            for (int j = i + 1; j < cols_; ++j) {
                T val = generate_random(min_val, max_val);
                if constexpr (detail::is_ordered_v<T>) {
                    if constexpr (std::is_floating_point_v<T>) {
                        val = std::clamp(val, T{-10.0}, T{10.0});
                    } else {
                        val = std::clamp(val, static_cast<T>(min_val), static_cast<T>(max_val));
                    }
                }
                (*this)(i, j) = val;
            }
        }
    }

    std::optional<int> find_max_in_subcol(int row, int col) {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            DEBUG_PRINTF("ERROR: index out of range\n");
            return std::nullopt;
        }

        if (rows_ == 0)
            return std::nullopt;

        int max_val_index = row;
        using std::abs;
        auto max_abs = abs(matrix_[row][col]);

        for (int i = row + 1; i < rows_; ++i) {
            auto current_abs = abs(matrix_[i][col]);
            if (current_abs > max_abs) {
                max_val_index = i;
                max_abs = current_abs;
            }
        }

        return max_val_index;
    }

    std::optional<T> det(int row, int col, int size) {
        if constexpr (detail::is_builtin_integral_v<T>) {
            return det_integer_algorithm(row, col, size);
        } else {
            static_assert(detail::has_division_v<T>,
                "Numeric determinant requires operator/ for type T.");
            return det_numeric_impl(row, col, size);
        }
    }

    std::optional<T> det() {
        return det(0, 0, min_dim_);
    }

    std::optional<T> det_numeric_algorithm() {
        return det_numeric_impl(0, 0, min_dim_);
    }

    template<typename U = T>
    using inverse_return_type = typename std::conditional<
        std::is_integral<U>::value,
        double,
        U
    >::type;

    Matrix<inverse_return_type<>> inverse() const {
        if (rows_ != cols_) throw std::invalid_argument("Matrix must be square");

        int n = rows_;
        using ComputeType = inverse_return_type<>;
        Matrix<ComputeType> lu(n, n);
        
        if constexpr (detail::has_abs_v<ComputeType>) {
            return inverse_impl_with_abs(lu);
        } else {
            return inverse_impl_generic(lu);
        }
    }

    Matrix<inverse_return_type<>> inverse_gauss_jordan() const {
        if (rows_ != cols_) {
            throw std::invalid_argument("Matrix must be square to compute inverse");
        }

        int n = rows_;
        using ComputeType = inverse_return_type<>;
        
        Matrix<ComputeType> augmented(n, 2 * n);
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented(i, j) = static_cast<ComputeType>((*this)(i, j));
            }
            augmented(i, n + i) = ComputeType{1};
        }
        
        for (int k = 0; k < n; ++k) {
            int max_row = k;
            
            auto get_abs = [](const ComputeType& val) {
                if constexpr (detail::has_abs_v<ComputeType>) {
                    return std::abs(val);
                } else {
                    return val;
                }
            };
            
            auto compare = [&](int i, int j) -> bool {
                auto abs_i = get_abs(augmented(i, k));
                auto abs_j = get_abs(augmented(j, k));
                return abs_i < abs_j;
            };
            
            for (int i = k + 1; i < n; ++i) {
                if (compare(max_row, i)) {
                    max_row = i;
                }
            }
            
            if (augmented(max_row, k) == ComputeType{0}) {
                throw std::runtime_error("Matrix is singular");
            }
            
            if (max_row != k) {
                augmented.swap_rows(k, max_row);
            }
            
            ComputeType pivot = augmented(k, k);
            for (int j = k; j < 2 * n; ++j) {
                augmented(k, j) /= pivot;
            }
            
            for (int i = 0; i < n; ++i) {
                if (i != k) {
                    ComputeType factor = augmented(i, k);
                    if (factor != ComputeType{0}) {
                        for (int j = k; j < 2 * n; ++j) {
                            augmented(i, j) -= factor * augmented(k, j);
                        }
                    }
                }
            }
        }
        
        Matrix<ComputeType> inv(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                inv(i, j) = augmented(i, n + j);
            }
        }
        
        return inv;
    }

private:
    
    bool alloc_matrix_() {
        if (rows_ <= 0 || cols_ <= 0) {
            DEBUG_PRINTF("ERROR: invalid dimensions");
            return false;
        }

        matrix_ = std::make_unique<std::unique_ptr<T[]>[]>(rows_);

        for (int i = 0; i < rows_; i++)
            matrix_[i] = std::make_unique<T[]>(cols_);

        return true;
    }

    void init_zero_() {
        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                matrix_[i][j] = T{};
    }

    static std::vector<T> create_controlled_diagonal(int size, T min_val, T max_val, T target_determinant_magnitude) {
        std::vector<T> diagonal(size);
        
        if (size == 0) return diagonal;

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
                    if (test_val == T{0}) continue;
                    
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
                        if (abs_elem > T{1e50}) {
                            rand_element = (rand_element >= T{0}) ? T{1e50} : T{-1e50};
                        } else if (abs_elem < T{1e-50}) {
                            rand_element = (rand_element >= T{0}) ? T{1e-50} : T{-1e-50};
                        }
                    }
                }

                diagonal[i] = rand_element;
                current_det *= rand_element;
            }
        }
        
        return diagonal;
    }

    static void apply_transformation_type_I(Matrix &matrix, int rows) {
        int row_1 = generate_random_int_(0, rows - 1);
        int row_2 = generate_random_int_(0, rows - 1);
        if (row_1 != row_2) {
            matrix.swap_rows(row_1, row_2);
        }
    }

    static void apply_transformation_type_II(Matrix &matrix, int rows, int cols,
        T min_val, T max_val, T &effective_det) {

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
                    if (abs_det > T{1e100}) {
                        stabilize_matrix(matrix, rows, cols, effective_det);
                    }
                }
            }
        }
    }

    static void apply_transformation_type_III(Matrix &matrix, int rows, int cols,
        T min_val, T max_val) {

        int row_1 = generate_random_int_(0, rows - 1);
        int row_2 = generate_random_int_(0, rows - 1);
        if (row_1 == row_2) return;

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

    static void apply_controlled_transformations(Matrix &matrix, int rows, int cols,
        T min_val,
        T max_val, int iterations) {
        
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
                    apply_transformation_type_II(matrix, rows, cols, min_val, max_val, effective_det);
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

    static void stabilize_matrix(Matrix &matrix, int rows, int cols, T &effective_det) {
        static_assert(detail::is_ordered_v<T>, "stabilize_matrix requires ordered type");

        for (int r = 0; r < rows; ++r) {
            T row_norm = T{0};
            for (int c = 0; c < cols; ++c) {
                row_norm += matrix(r, c) * matrix(r, c);
            }
            if (row_norm > T{1e-10}) {
                T factor = std::sqrt(T{1e20} / std::abs(effective_det));
                if (factor < T{1}) {
                    factor = T{1};
                }
                matrix.multiply_row(r, factor);
                effective_det *= factor;
                break;
            }
        }
    }

    static T gcd(T a, T b) {
        while (b != T{0}) {
            T temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    static T lcm(T a, T b) {
        T g = gcd(a, b);
        return (g == T{0}) ? T{0} : (a / g) * b;
    }

    std::optional<T> det_integer_algorithm(int row, int col, int size) {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_ || 
            row + size > rows_ || col + size > cols_ || size <= 0) {
            return std::nullopt;
        }

        if (size == 1) {
            determinant_ = (*this)(row, col);
            return determinant_;
        }

        Matrix<T> work_matrix(*this, row, col, size, size);
        long long previous_pivot = 1;
        int row_swaps = 0;

        for (int step = 0; step < size - 1; ++step) {
            int pivot_row = -1;
            for (int i = step; i < size; ++i) {
                if (work_matrix(i, step) != T{0}) {
                    pivot_row = i;
                    break;
                }
            }

            if (pivot_row == -1) {
                determinant_ = T{0};
                return determinant_;
            }

            if (pivot_row != step) {
                work_matrix.swap_rows(step, pivot_row);
                ++row_swaps;
            }

            long long current_pivot = static_cast<long long>(work_matrix(step, step));

            for (int i = step + 1; i < size; ++i) {
                for (int j = step + 1; j < size; ++j) {
                    long long numerator = static_cast<long long>(work_matrix(i, j)) * current_pivot 
                                        - static_cast<long long>(work_matrix(i, step)) * 
                                          static_cast<long long>(work_matrix(step, j));
                    
                    if (step > 0 && previous_pivot != 0) {
                        long long new_val = numerator / previous_pivot;
                        work_matrix(i, j) = static_cast<T>(new_val);
                    } else {
                        work_matrix(i, j) = static_cast<T>(numerator);
                    }
                }
                work_matrix(i, step) = T{0};
            }

            previous_pivot = current_pivot;
        }

        long long final_det = static_cast<long long>(work_matrix(size - 1, size - 1));
        
        if (row_swaps % 2 != 0) {
            final_det = -final_det;
        }

        if (final_det > std::numeric_limits<T>::max() || 
            final_det < std::numeric_limits<T>::min()) {
            return std::nullopt;
        }

        determinant_ = static_cast<T>(final_det);
        return determinant_;
    }

    std::optional<T> det_numeric_impl(int row, int col, int size) {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_ || row + size > rows_ ||
            col + size > cols_ || size <= 0) {
            return std::nullopt;
        }

        T determinant = T{1};
        Matrix matrix_cpy = *this;
        int sign = 1;

        for (int j = 0; j < size; ++j) {
            int current_col = col + j;
            int current_row = row + j;

            std::optional<int> max_index_opt =
                matrix_cpy.find_max_in_subcol(current_row, current_col);
            if (!max_index_opt) {
                return T{};
            }

            int max_index = *max_index_opt;

            if (max_index != current_row) {
                matrix_cpy.swap_rows(max_index, current_row);
                sign = -sign;
            }

            T pivot = matrix_cpy(current_row, current_col);
            if (is_zero(pivot)) {
                determinant_.emplace(T{0});
                return T{};
            }

            determinant *= pivot;

            if constexpr (std::is_floating_point_v<T>) {
                T abs_det = std::abs(determinant);
                if (abs_det > T{1e50} || abs_det < T{1e-50}) {
                    int order = static_cast<int>(std::floor(std::log10(abs_det)));
                    T normalized = determinant / std::pow(T{10}, T(order));
                    determinant = normalized;
                }
            }

            for (int i = current_row + 1; i < row + size; ++i) {
                T scalar = matrix_cpy(i, current_col) / pivot;

                for (int k = current_col + 1; k < col + size; ++k) {
                    matrix_cpy(i, k) = matrix_cpy(i, k) - scalar * matrix_cpy(current_row, k);
                }
            }
        }

        determinant_.emplace(determinant * T(sign));
        return determinant_;
    }

    static int generate_random_int_(int min = 1, int max = 100) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(min, max);

        return dis(gen);
    }

    static double generate_random_double_(double min = 0.0, double max = 1.0) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(min, max);

        return dis(gen);
    }

    static T generate_random(T min_val = {}, T max_val = {}) {
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
                actual_min = 0.0;
                actual_max = 1.0;
            }
            return generate_random_double_(actual_min, actual_max);
        }
        else {
            double actual_min = 0.0;
            double actual_max = 1.0;
            
            if constexpr (detail::is_ordered_v<T>) {
                actual_min = static_cast<double>(min_val);
                actual_max = static_cast<double>(max_val);
            }
            
            return T{generate_random_double_(actual_min, actual_max)};
        }
    }

    std::optional<T> &get_determinant_() { return determinant_; }

    template<typename BinaryOp>
    Matrix binary_operation(const Matrix& other, BinaryOp op) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for operation");
        }

        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result(i, j) = op((*this)(i, j), other(i, j));
            }
        }
        return result;
    }

    static void matrix_multiply_basic_impl(const Matrix& A, const Matrix& B, Matrix& C) {
        const int rows_A = A.rows_;
        const int cols_A = A.cols_;
        const int cols_B = B.cols_;

        for (int i = 0; i < rows_A; ++i) {
            for (int k = 0; k < cols_A; ++k) {
                T temp_a_ik = A(i, k);
                for (int j = 0; j < cols_B; ++j) {
                    C(i, j) += temp_a_ik * B(k, j);
                }
            }
        }
    }

    #ifdef __AVX__
    static void matrix_multiply_avx_impl(const Matrix& A, const Matrix& B, Matrix& C) {
        if constexpr (!std::is_same_v<T, double>) {
             // Только для double
             matrix_multiply_basic_impl(A, B, C);
             return;
        }

        const int rows_A = A.rows_;
        const int cols_A = A.cols_;
        const int cols_B = B.cols_;

        const int cols_B_simd = (cols_B / 4) * 4;

        for (int i = 0; i < rows_A; ++i) {
            for (int k = 0; k < cols_A; ++k) {
                __m256d temp_a_ik_vec = _mm256_broadcast_sd(&A(i, k));

                int j = 0;
                for (; j < cols_B_simd; j += 4) {
                    __m256d c_vec = _mm256_loadu_pd(&C(i, j));
                    __m256d b_vec = _mm256_loadu_pd(&B(k, j));
                    __m256d mul_res = _mm256_mul_pd(temp_a_ik_vec, b_vec);
                    c_vec = _mm256_add_pd(c_vec, mul_res);
                    _mm256_storeu_pd(&C(i, j), c_vec);
                }

                for (; j < cols_B; ++j) {
                    C(i, j) += A(i, k) * B(k, j);
                }
            }
        }
    }
    #endif

    #ifdef __AVX__
    static void matrix_multiply_avx_impl_float(const Matrix& A, const Matrix& B, Matrix& C) {
        if constexpr (!std::is_same_v<T, float>) {
             matrix_multiply_basic_impl(A, B, C);
             return;
        }

        const int rows_A = A.rows_;
        const int cols_A = A.cols_;
        const int cols_B = B.cols_;

        const int cols_B_simd = (cols_B / 8) * 8;

        for (int i = 0; i < rows_A; ++i) {
            for (int k = 0; k < cols_A; ++k) {
                __m256 temp_a_ik_vec = _mm256_broadcast_ss(&A(i, k));

                int j = 0;
                for (; j < cols_B_simd; j += 8) {
                    __m256 c_vec = _mm256_loadu_ps(&C(i, j));
                    __m256 b_vec = _mm256_loadu_ps(&B(k, j));
                    __m256 mul_res = _mm256_mul_ps(temp_a_ik_vec, b_vec);
                    c_vec = _mm256_add_ps(c_vec, mul_res);
                    _mm256_storeu_ps(&C(i, j), c_vec);
                }

                for (; j < cols_B; ++j) {
                    C(i, j) += A(i, k) * B(k, j);
                }
            }
        }
    }
    #endif

    template<typename AlgorithmTag>
    static Matrix multiply_impl(const Matrix& A, const Matrix& B) {
        if (A.cols_ != B.rows_) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication (A.cols != B.rows)");
        }

        Matrix result(A.rows_, B.cols_);

        if constexpr (std::is_same_v<AlgorithmTag, BasicAlgorithm>) {
            matrix_multiply_basic_impl(A, B, result);
        }
        #ifdef __AVX__
        else if constexpr (std::is_same_v<AlgorithmTag, AvxAlgorithm>) {
            if constexpr (std::is_same_v<T, double>) {
                 matrix_multiply_avx_impl(A, B, result);
            } else if constexpr (std::is_same_v<T, float>) {
                 matrix_multiply_avx_impl_float(A, B, result);
            } else {
                 matrix_multiply_basic_impl(A, B, result);
            }
        }
        #endif 
        else {
            matrix_multiply_basic_impl(A, B, result);
        }

        return result;
    }

    template<typename ComputeType>
    Matrix<ComputeType> inverse_impl_with_abs(Matrix<ComputeType>& A) const {
        int n = rows_;
        
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A(i, j) = static_cast<ComputeType>((*this)(i, j));
        
        std::vector<int> perm(n);
        std::vector<double> scale(n);
        
        for (int i = 0; i < n; ++i) {
            perm[i] = i;
            double max_val = 0.0;
            for (int j = 0; j < n; ++j) {
                double abs_val = std::abs(A(i, j));
                if (abs_val > max_val) max_val = abs_val;
            }
            if (max_val == 0.0) throw std::runtime_error("Matrix is singular");
            scale[i] = 1.0 / max_val;
        }
        
        for (int k = 0; k < n; ++k) {
            int max_row = k;
            double max_val = std::abs(A(k, k)) * scale[k];
            
            for (int i = k + 1; i < n; ++i) {
                double abs_val = std::abs(A(i, k)) * scale[i];
                if (abs_val > max_val) {
                    max_val = abs_val;
                    max_row = i;
                }
            }
            
            if (max_val == 0.0) throw std::runtime_error("Matrix is singular");
            
            if (max_row != k) {
                A.swap_rows(k, max_row);
                std::swap(scale[k], scale[max_row]);
                std::swap(perm[k], perm[max_row]);
            }
            
            ComputeType pivot = A(k, k);
            for (int i = k + 1; i < n; ++i) {
                ComputeType factor = A(i, k) / pivot;
                A(i, k) = factor;
                for (int j = k + 1; j < n; ++j) {
                    A(i, j) -= factor * A(k, j);
                }
            }
        }
        
        Matrix<ComputeType> inv(n, n);
        Matrix<ComputeType> B(n, n);
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                B(i, j) = (perm[i] == j) ? ComputeType{1} : ComputeType{0};
            }
        }
        
        for (int i = 0; i < n; ++i) {
            ComputeType* B_i = &B(i, 0);
            ComputeType* A_i = &A(i, 0);
            
            for (int j = 0; j < n; ++j) {
                ComputeType sum = B_i[j];
                for (int k = 0; k < i; ++k) {
                    ComputeType* B_k = &B(k, 0);
                    sum -= A_i[k] * B_k[j];
                }
                B_i[j] = sum;
            }
        }
        
        for (int i = n - 1; i >= 0; --i) {
            ComputeType* inv_i = &inv(i, 0);
            ComputeType* B_i = &B(i, 0);
            ComputeType* A_i = &A(i, 0);
            ComputeType pivot = A_i[i];
            
            for (int j = 0; j < n; ++j) {
                ComputeType sum = B_i[j];
                for (int k = i + 1; k < n; ++k) {
                    ComputeType* inv_k = &inv(k, 0);
                    sum -= A_i[k] * inv_k[j];
                }
                inv_i[j] = sum / pivot;
            }
        }
        
        return inv;
    }

    template<typename ComputeType>
    Matrix<ComputeType> inverse_impl_generic(Matrix<ComputeType>& A) const {
        int n = rows_;

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A(i, j) = static_cast<ComputeType>((*this)(i, j));

        std::vector<int> perm(n);
        for (int i = 0; i < n; ++i)
            perm[i] = i;

        for (int k = 0; k < n; ++k) {
            int max_row = k;
            while (max_row < n && A(max_row, k) == ComputeType{0})
                ++max_row;

            if (max_row == n)
                throw std::runtime_error("Matrix is singular");

            if (max_row != k) {
                A.swap_rows(k, max_row);
                std::swap(perm[k], perm[max_row]);
            }

            ComputeType pivot = A(k, k);
            for (int i = k + 1; i < n; ++i) {
                ComputeType factor = A(i, k) / pivot;
                A(i, k) = factor;
                for (int j = k + 1; j < n; ++j) {
                    A(i, j) -= factor * A(k, j);
                }
            }
        }

        Matrix<ComputeType> inv(n, n);
        
        for (int col = 0; col < n; ++col) {
            std::vector<ComputeType> b(n, ComputeType{0});
            
            for (int i = 0; i < n; ++i) {
                b[i] = (perm[i] == col) ? ComputeType{1} : ComputeType{0};
            }
            
            for (int i = 0; i < n; ++i) {
                ComputeType sum = b[i];
                for (int j = 0; j < i; ++j) {
                    sum -= A(i, j) * b[j];
                }
                b[i] = sum;
            }
            
            for (int i = n - 1; i >= 0; --i) {
                ComputeType sum = b[i];
                for (int j = i + 1; j < n; ++j) {
                    sum -= A(i, j) * inv(j, col);
                }
                inv(i, col) = sum / A(i, i);
            }
        }

        return inv;
    }
};
