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

template <typename T> class Matrix;

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

template<typename T>
struct is_matrix : std::false_type {};

template<typename T>
struct is_matrix<Matrix<T>> : std::true_type {};

template<typename T>
struct has_abs<Matrix<T>> : has_abs<typename Matrix<T>::value_type> {};

template<typename T>
struct has_division<Matrix<T>> : has_division<typename Matrix<T>::value_type> {};

template<typename T>
constexpr bool is_matrix_v = is_matrix<T>::value;
}

const int Default_iterations = 100;

struct BasicAlgorithm {};
struct AvxAlgorithm {};

template <typename T> class Matrix {
private:
    enum class Tranformation_types { I_TYPE = 0, II_TYPE = 1, III_TYPE = 2 };

    std::unique_ptr<std::unique_ptr<T[]>[]> matrix_;
    int rows_, cols_, min_dim_;
    mutable std::optional<T> determinant_ = std::nullopt;

    static constexpr auto epsilon = 1e-10;

public:
    using value_type = T;

    Matrix() : rows_(0), cols_(0), min_dim_(0), matrix_(nullptr) {}

    Matrix(int rows, int cols) : rows_(rows), cols_(cols), min_dim_(std::min(rows, cols)) {
        if (rows > 0 && cols > 0) {
            alloc_matrix_();
        }
    }

    Matrix(T value) : rows_(0), cols_(0), min_dim_(0), matrix_(nullptr) {
        if constexpr (detail::is_matrix_v<T>) {
            throw std::runtime_error("Cannot create Matrix<Matrix<T>> from single value");
        }
    }

    ~Matrix() {
        if constexpr (detail::is_matrix_v<T>) {
            if (matrix_) {
                for (int i = 0; i < rows_; ++i) {
                    if (matrix_[i]) {
                        T* raw_ptr = matrix_[i].get();
                        for (int j = 0; j < cols_; ++j) {
                            raw_ptr[j].~T();  // Явный вызов деструктора
                        }
                    }
                }
            }
        }
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
        int min_dim = std::min(rows, cols);
        for (int i = 0; i < min_dim; i++) {
            if constexpr (detail::is_matrix_v<T>) {
                using BlockType = typename T::value_type;
                int block_rows = result(0, 0).get_rows();
                int block_cols = result(0, 0).get_cols();
                result(i, i) = Matrix<BlockType>::Identity(block_rows, block_cols);
            } else {
                result(i, i) = T{1};
            }
        }
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
        T max_condition_number = {}) 
    {
        
        if constexpr (detail::is_matrix_v<T>) {
            // Generate matrix of matrices
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result(i, j) = T::Generate_matrix(
                        min_val.get_rows(), min_val.get_cols(),
                        typename T::value_type{-10}, typename T::value_type{10}
                    );
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

            std::vector<T> diagonal = create_controlled_diagonal(
                std::min(rows, cols), min_val, max_val, target_determinant_magnitude);
            Matrix result = Matrix::Diagonal(rows, cols, diagonal);
            result.fill_upper_triangle(min_val, max_val);

            apply_controlled_transformations(result, rows, cols, min_val, max_val, iterations);

            return result;
        }
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

    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), min_dim_(other.min_dim_) {
        if (rows_ > 0 && cols_ > 0) {
            alloc_matrix_();
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    (*this)(i, j) = other(i, j);
                }
            }
        }
    }

    // Оператор присваивания
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            // Освобождаем старую память
            if constexpr (detail::is_matrix_v<T>) {
                if (matrix_) {
                    for (int i = 0; i < rows_; ++i) {
                        if (matrix_[i]) {
                            T* raw_ptr = matrix_[i].get();
                            for (int j = 0; j < cols_; ++j) {
                                raw_ptr[j].~T();
                            }
                        }
                    }
                }
            }
            
            rows_ = other.rows_;
            cols_ = other.cols_;
            min_dim_ = other.min_dim_;
            
            if (rows_ > 0 && cols_ > 0) {
                alloc_matrix_();
                for (int i = 0; i < rows_; ++i) {
                    for (int j = 0; j < cols_; ++j) {
                        (*this)(i, j) = other(i, j);
                    }
                }
            } else {
                matrix_.reset();
            }
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

    Matrix& operator*=(const Matrix& other) {
        *this = *this * other;
        return *this;
    }

    bool operator==(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) return false;
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (!is_equal((*this)(i, j), other(i, j))) return false;
            }
        }
        return true;
    }

    bool operator!=(const Matrix& other) const {
        return !(*this == other);
    }

    static bool is_equal(const T& a, const T& b) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(a - b) < epsilon;
        } else {
            return a == b;
        }
    }

    Matrix operator-() const {
        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result(i, j) = -(*this)(i, j);
            }
        }
        return result;
    }

    int get_rows() const { return rows_; }
    int get_cols() const { return cols_; }
    int get_min_dim() const { return min_dim_; }
    std::optional<T> get_determinant() const { return determinant_; }

    template <typename U = T>
    static bool is_zero(const U &value) {
        if constexpr (detail::is_matrix_v<U>) {
            auto det_opt = value.det();
            return !det_opt.has_value() || is_zero(*det_opt);
        } else if constexpr (std::is_same_v<U, float> || std::is_same_v<U, double>) {
            return std::abs(value) < epsilon;
        } else {
            return value == U{};
        }
    }

    bool is_zero(int i, int j) const { return is_zero((*this)(i, j)); }

    /********** 3 types of matrix transformation ***********/
    void swap_rows(int i, int j) { // I.

        if (i != j) {
            std::swap(matrix_[i], matrix_[j]);

            if (determinant_)
                determinant_ = -*determinant_;
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
                if constexpr (detail::is_matrix_v<T>) {
                    if (rows_ <= 5 && cols_ <= 5 && (*this)(i, j).get_rows() <= 3 && (*this)(i, j).get_cols() <= 3) {
                        std::cout << "[" << (*this)(i, j).get_rows() << "x" << (*this)(i, j).get_cols() << "] ";
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

    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
        if constexpr (detail::is_matrix_v<T>) {
            if (matrix.rows_ <= 3 && matrix.cols_ <= 3) {
                for (int i = 0; i < matrix.rows_; ++i) {
                    os << "[";
                    for (int j = 0; j < matrix.cols_; ++j) {
                        if (j > 0) os << ", ";
                        if (matrix(i, j).get_rows() <= 3 && matrix(i, j).get_cols() <= 3) {
                            os << matrix(i, j);
                        } else {
                            os << "[Matrix " << matrix(i, j).get_rows()
                               << "x" << matrix(i, j).get_cols() << "]";
                        }
                    }
                    os << "]";
                    if (i < matrix.rows_ - 1) os << "\n";
                }
            } else {
                os << "[BlockMatrix " << matrix.rows_ << "x" << matrix.cols_
                   << " of " << matrix(0, 0).get_rows()
                   << "x" << matrix(0, 0).get_cols() << "]";
            }
        }
        else {
            if (matrix.rows_ <= 5 && matrix.cols_ <= 5) {
                for (int i = 0; i < matrix.rows_; ++i) {
                    for (int j = 0; j < matrix.cols_; ++j) {
                        os << matrix(i, j);
                        if (j < matrix.cols_ - 1) os << " ";
                    }
                    if (i < matrix.rows_ - 1) os << "\n";
                }
            } else {
                os << "[Matrix " << matrix.rows_ << "x" << matrix.cols_ << "]";
            }
        }
        return os;
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
        auto max_abs = std::abs(matrix_[row][col]);
        for (int i = row + 1; i < rows_; ++i) {
            auto current_abs = std::abs(matrix_[i][col]);
            if (current_abs > max_abs) {
                max_val_index = i;
                max_abs = current_abs;
            }
        }

        return max_val_index;
    }

    std::optional<T> det(int row, int col, int size) const {
        if constexpr (detail::is_builtin_integral_v<T>) {
            return det_integer_algorithm(row, col, size);
        } else {
            static_assert(detail::has_division_v<T>,
                "Numeric determinant requires operator/ for type T.");
            return det_numeric_impl(row, col, size);
        }
    }

    std::optional<T> det() const {
        return det(0, 0, min_dim_);
    }

    std::optional<T> det_numeric_algorithm() const {
        return det_numeric_impl(0, 0, min_dim_);
    }

    template<typename U = T>
using inverse_return_type = typename std::conditional<
    std::is_integral<U>::value && !detail::is_matrix<U>::value,
    double,
    U
>::type;

  friend auto operator/(const Matrix& A, const Matrix& B) {
    DEBUG_PRINTF("operator/: A=%dx%d, B=%dx%d\n", A.rows_, A.cols_, B.rows_, B.cols_);
    
    if (A.cols_ != B.rows_) {
        DEBUG_PRINTF("ERROR: A.cols=%d != B.rows=%d\n", A.cols_, B.rows_);
        throw std::invalid_argument("Matrix dimensions don't match for division (A.cols != B.rows)");
    }
    
    try {
        using ResultType = inverse_return_type<T>;
        DEBUG_PRINTF("ResultType: %s\n", typeid(ResultType).name());
        
        Matrix<ResultType> A_cast = A;
        Matrix<ResultType> B_cast = B;
        
        DEBUG_PRINTF("Computing inverse...\n");
        Matrix<ResultType> B_inv = B_cast.inverse();
        DEBUG_PRINTF("B_inv computed: %dx%d\n", B_inv.get_rows(), B_inv.get_cols());
        
        DEBUG_PRINTF("Multiplying...\n");
        auto result = A_cast * B_inv;
        DEBUG_PRINTF("Division completed\n");
        return result;
    } catch (const std::exception& e) {
        DEBUG_PRINTF("Error in division: %s\n", e.what());
        throw std::runtime_error("Cannot divide by singular matrix B");
    }
} 

template<typename U>
Matrix<U> cast_to() const {
    DEBUG_PRINTF("Casting Matrix %dx%d to different type\n", rows_, cols_);
    Matrix<U> result(rows_, cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result(i, j) = static_cast<U>((*this)(i, j));
        }
    }
    return result;
}

    template<typename U>
    operator Matrix<U>() const {
        Matrix<U> result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result(i, j) = static_cast<U>((*this)(i, j));
            }
        }
        return result;
    }

   template<typename ComputeType>
Matrix<ComputeType> inverse_impl_with_abs() const {
    int n = rows_;
    if (n == 0) return Matrix<ComputeType>();
    
    Matrix<ComputeType> augmented(n, 2 * n);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented(i, j) = (*this)(i, j);
        }
        
        if constexpr (detail::is_matrix_v<ComputeType>) {
            using BlockType = typename ComputeType::value_type;
            int block_rows = (*this)(0, 0).get_rows();
            int block_cols = (*this)(0, 0).get_cols();
            augmented(i, n + i) = Matrix<BlockType>::Identity(block_rows, block_cols);
        } else {
            augmented(i, n + i) = ComputeType(1);
        }
    }
    
    for (int k = 0; k < n; ++k) {
        int max_row = k;
        double max_val = 0.0;
        
        if constexpr (detail::is_matrix_v<ComputeType>) {
            max_val = compute_block_norm(augmented(k, k));
            for (int i = k + 1; i < n; ++i) {
                double current_val = compute_block_norm(augmented(i, k));
                if (current_val > max_val) {
                    max_val = current_val;
                    max_row = i;
                }
            }
        } else if constexpr (detail::has_abs_v<ComputeType>) {
            max_val = std::abs(augmented(k, k));
            for (int i = k + 1; i < n; ++i) {
                double current_val = std::abs(augmented(i, k));
                if (current_val > max_val) {
                    max_val = current_val;
                    max_row = i;
                }
            }
        } else {
            max_val = static_cast<double>(augmented(k, k) * augmented(k, k));
            for (int i = k + 1; i < n; ++i) {
                double current_val = static_cast<double>(augmented(i, k) * augmented(i, k));
                if (current_val > max_val) {
                    max_val = current_val;
                    max_row = i;
                }
            }
        }
        
        if (max_val < epsilon) throw std::runtime_error("Matrix is singular");
        
        if (max_row != k) {
            augmented.swap_rows(k, max_row);
        }
        
        ComputeType pivot = augmented(k, k);
        
        for (int j = k; j < 2 * n; ++j) {
            if constexpr (detail::is_matrix_v<ComputeType>) {
                auto inv_pivot = pivot.inverse();
                augmented(k, j) = augmented(k, j) * inv_pivot;
            } else if constexpr (detail::has_division_v<ComputeType>) {
                augmented(k, j) = augmented(k, j) / pivot;
            } else {
                throw std::runtime_error("Cannot divide for this type");
            }
        }
        
        for (int i = 0; i < n; ++i) {
            if (i != k) {
                ComputeType factor = augmented(i, k);
                bool factor_is_zero = false;
                
                if constexpr (detail::is_matrix_v<ComputeType>) {
                    factor_is_zero = compute_block_norm(factor) < epsilon;
                } else if constexpr (detail::has_abs_v<ComputeType>) {
                    factor_is_zero = std::abs(factor) < epsilon;
                } else {
                    factor_is_zero = factor == ComputeType{};
                }
                
                if (!factor_is_zero) {
                    for (int j = k; j < 2 * n; ++j) {
                        augmented(i, j) = augmented(i, j) - factor * augmented(k, j);
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

    template<typename ComputeType = inverse_return_type<T>>
Matrix<ComputeType> inverse() const {
    if (rows_ != cols_) throw std::invalid_argument("Matrix must be square");
    
    if constexpr (detail::has_abs_v<ComputeType>) {
        return inverse_impl_with_abs<ComputeType>();
    } else {
        return inverse_impl_generic<ComputeType>();
    }
}

template <typename U = T>
    std::optional<int> find_pivot_in_subcol(int row, int col) const {
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
            DEBUG_PRINTF("MATRIX_TYPE!");
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

private:
    bool alloc_matrix_() {
        if (rows_ <= 0 || cols_ <= 0) {
            return false;
        }

        if constexpr (detail::is_matrix_v<T>) {
            matrix_ = std::unique_ptr<std::unique_ptr<T[]>[]>(new std::unique_ptr<T[]>[rows_]);
            
            for (int i = 0; i < rows_; ++i) {
                T* raw_row = new T[cols_];
                matrix_[i].reset(raw_row);
                
                for (int j = 0; j < cols_; ++j) {
                    new (&raw_row[j]) T();
                }
            }
        } else {
            matrix_ = std::make_unique<std::unique_ptr<T[]>[]>(rows_);
            for (int i = 0; i < rows_; ++i) {
                matrix_[i] = std::make_unique<T[]>(cols_);
            }
        }
        
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

    

    std::optional<T> det_integer_algorithm(int row, int col, int size) const {
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

    template <typename U = T>
    static U identity_element(int rows = 1, int cols = 1) {
        if constexpr (detail::is_matrix_v<U>) {
            return U::Identity(rows, cols);
        } else {
            return U{1};
        }
    }

    template <typename U = T>
    static U zero_element(int rows = 1, int cols = 1) {
        if constexpr (detail::is_matrix_v<U>) {
            return U::Zero(rows, cols);
        } else {
            return U{0};
        }
    }

template <typename U = T>
std::optional<T> det_numeric_impl(int row, int col, int size) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_ || 
        row + size > rows_ || col + size > cols_ || size <= 0) {
        return std::nullopt;
    }

    if (size == 1) {
        determinant_.emplace((*this)(row, col));
        return determinant_;
    }

    // Копируем подматрицу
    Matrix<T> matrix_cpy = Matrix<T>(*this, row, col, size, size);

    T determinant = identity_element<T>();
    int sign = 1;

    for (int j = 0; j < size; ++j) {
        // Ищем опорный элемент
        std::optional<int> max_index_opt = matrix_cpy.template find_pivot_in_subcol<T>(j, j);
        if (!max_index_opt) {
            determinant_.emplace(zero_element<T>());
            return zero_element<T>();
        }

        int max_index = *max_index_opt;
        if (max_index != j) {
            matrix_cpy.swap_rows(max_index, j);
            sign = -sign;
        }

        T pivot = matrix_cpy(j, j);
        
        if (is_zero(pivot)) {
            determinant_.emplace(zero_element<T>());
            return zero_element<T>();
        }

        determinant = determinant * pivot;

        // Исключение Гаусса
        for (int i = j + 1; i < size; ++i) {
            T scalar = matrix_cpy(i, j) / pivot;
            
            for (int k = j + 1; k < size; ++k) {
                matrix_cpy(i, k) = matrix_cpy(i, k) - scalar * matrix_cpy(j, k);
            }
        }
    }

    if (sign == -1) {
        determinant = -determinant;
    }
    
    determinant_ = determinant;
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
    DEBUG_PRINTF("multiply_impl: A=%dx%d, B=%dx%d\n", A.rows_, A.cols_, B.rows_, B.cols_);
    
    if (A.cols_ != B.rows_) {
        DEBUG_PRINTF("ERROR: A.cols=%d != B.rows=%d\n", A.cols_, B.rows_);
        throw std::invalid_argument("Matrix dimensions don't match for multiplication (A.cols != B.rows)");
    }

    Matrix result(A.rows_, B.cols_);
    DEBUG_PRINTF("Result matrix: %dx%d\n", result.rows_, result.cols_);

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

    DEBUG_PRINTF("Multiplication completed\n");
    return result;
} 

template<typename ComputeType>
static bool is_element_zero_impl(const ComputeType& elem) {
    if constexpr (detail::is_matrix_v<ComputeType>) {
        // Для блочных матриц проверяем каждый элемент
        for (int i = 0; i < elem.get_rows(); ++i) {
            for (int j = 0; j < elem.get_cols(); ++j) {
                if (!is_element_zero_impl(elem(i, j))) return false;
            }
        }
        return true;
    } else if constexpr (detail::has_abs_v<ComputeType>) {
        return std::abs(elem) < epsilon;
    } else {
        return elem == ComputeType{};
    }
}

template<typename U>
    static double compute_block_norm(const U& block) {
        if constexpr (detail::is_matrix_v<U>) {
            double norm = 0.0;
            for (int i = 0; i < block.get_rows(); ++i) {
                for (int j = 0; j < block.get_cols(); ++j) {
                    double val = compute_block_norm(block(i, j));
                    norm += val * val;
                }
            }
            return std::sqrt(norm);
        } else if constexpr (detail::has_abs_v<U>) {
            return std::abs(block);
        } else {
            return static_cast<double>(block * block);
        }
    }

template<typename ComputeType>
void initialize_augmented_matrix(Matrix<ComputeType>& augmented) const {
    int n = rows_;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented(i, j) = (*this)(i, j);
        }
    }
    
    for (int i = 0; i < n; ++i) {
        if constexpr (detail::is_matrix_v<ComputeType>) {
            using BlockType = typename ComputeType::value_type;
            int block_rows = (*this)(0, 0).get_rows();
            int block_cols = (*this)(0, 0).get_cols();
            Matrix<BlockType> identity = Matrix<BlockType>::Identity(block_rows, block_cols);
            augmented(i, n + i) = identity;
        } else {
            augmented(i, n + i) = ComputeType(1);
        }
    }
}

template<typename ComputeType>
int find_pivot_row(const Matrix<ComputeType>& augmented, int col) const {
    int n = augmented.get_rows();
    int pivot_row = col;
    
    if constexpr (detail::is_matrix_v<ComputeType>) {
        double pivot_norm = compute_block_norm(augmented(col, col));
        
        for (int i = col + 1; i < n; ++i) {
            double current_norm = compute_block_norm(augmented(i, col));
            if (current_norm > pivot_norm) {
                pivot_norm = current_norm;
                pivot_row = i;
            }
        }
    } else if constexpr (detail::has_abs_v<ComputeType>) {
        auto pivot_norm = std::abs(augmented(col, col));
        
        for (int i = col + 1; i < n; ++i) {
            auto current_norm = std::abs(augmented(i, col));
            if (current_norm > pivot_norm) {
                pivot_norm = current_norm;
                pivot_row = i;
            }
        }
    } else {
        auto pivot_norm = augmented(col, col) * augmented(col, col);
        
        for (int i = col + 1; i < n; ++i) {
            auto current_norm = augmented(i, col) * augmented(i, col);
            if (current_norm > pivot_norm) {
                pivot_norm = current_norm;
                pivot_row = i;
            }
        }
    }
    
    return pivot_row;
}

template<typename ComputeType>
static void normalize_row(Matrix<ComputeType>& augmented, int row) {
    int n = augmented.get_rows();
    ComputeType pivot = augmented(row, row);
    
    if constexpr (detail::is_matrix_v<ComputeType>) {
        auto inv_pivot = pivot.inverse();
        for (int j = row; j < 2 * n; ++j) {
            augmented(row, j) = augmented(row, j) * inv_pivot;
        }
    } else if constexpr (detail::has_division_v<ComputeType>) {
        for (int j = row; j < 2 * n; ++j) {
            augmented(row, j) = augmented(row, j) / pivot;
        }
    } else {
        throw std::runtime_error("Cannot normalize row for this type");
    }
}

template<typename ComputeType>
static void eliminate_rows(Matrix<ComputeType>& augmented, int pivot_row) {
    int n = augmented.get_rows();
    
    for (int i = 0; i < n; ++i) {
        if (i != pivot_row) {
            ComputeType factor = augmented(i, pivot_row);
            
            bool factor_is_zero = false;
            if constexpr (detail::is_matrix_v<ComputeType>) {
                factor_is_zero = compute_block_norm(factor) < epsilon;
            } else if constexpr (detail::has_abs_v<ComputeType>) {
                factor_is_zero = std::abs(factor) < epsilon;
            } else {
                factor_is_zero = factor == ComputeType{};
            }
            
            if (!factor_is_zero) {
                for (int j = pivot_row; j < 2 * n; ++j) {
                    augmented(i, j) = augmented(i, j) - factor * augmented(pivot_row, j);
                }
            }
        }
    }
}

    template<typename ComputeType>
    static Matrix<ComputeType> extract_inverse(const Matrix<ComputeType>& augmented) {
        int n = augmented.get_rows();
        Matrix<ComputeType> inv(n, n);
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                inv(i, j) = augmented(i, n + j);
            }
        }
        
        return inv;
    }

    template<typename ComputeType>
Matrix<ComputeType> inverse_impl_generic() const {
    DEBUG_PRINTF("inverse_impl_generic: matrix %dx%d\n", rows_, cols_);
    
    if (rows_ != cols_) throw std::invalid_argument("Matrix must be square");
    
    int n = rows_;
    if (n == 0) return Matrix<ComputeType>();
    
    Matrix<ComputeType> augmented(n, 2 * n);
    
    DEBUG_PRINTF("Initializing augmented matrix\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented(i, j) = (*this)(i, j);
        }
        
        if constexpr (detail::is_matrix_v<ComputeType>) {
            using BlockType = typename ComputeType::value_type;
            int block_rows = (*this)(0, 0).get_rows();
            int block_cols = (*this)(0, 0).get_cols();
            augmented(i, n + i) = Matrix<BlockType>::Identity(block_rows, block_cols);
        } else {
            augmented(i, n + i) = ComputeType(1);
        }
    }
    
    DEBUG_PRINTF("Starting Gauss-Jordan elimination\n");
    for (int k = 0; k < n; ++k) {
        DEBUG_PRINTF("Step %d\n", k);
        
        int pivot_row = k;
        double max_val = 0.0;
        
        if constexpr (detail::is_matrix_v<ComputeType>) {
            max_val = compute_block_norm(augmented(k, k));
            for (int i = k + 1; i < n; ++i) {
                double current_val = compute_block_norm(augmented(i, k));
                if (current_val > max_val) {
                    max_val = current_val;
                    pivot_row = i;
                }
            }
        } else if constexpr (detail::has_abs_v<ComputeType>) {
            max_val = std::abs(augmented(k, k));
            for (int i = k + 1; i < n; ++i) {
                double current_val = std::abs(augmented(i, k));
                if (current_val > max_val) {
                    max_val = current_val;
                    pivot_row = i;
                }
            }
        } else {
            max_val = static_cast<double>(augmented(k, k) * augmented(k, k));
            for (int i = k + 1; i < n; ++i) {
                double current_val = static_cast<double>(augmented(i, k) * augmented(i, k));
                if (current_val > max_val) {
                    max_val = current_val;
                    pivot_row = i;
                }
            }
        }
        
        DEBUG_PRINTF("Pivot row: %d, max_val: %f\n", pivot_row, max_val);
        
        if (max_val < epsilon) throw std::runtime_error("Matrix is singular");
        
        if (pivot_row != k) {
            DEBUG_PRINTF("Swapping rows %d and %d\n", k, pivot_row);
            augmented.swap_rows(k, pivot_row);
        }
        
        ComputeType pivot = augmented(k, k);
        DEBUG_PRINTF("Pivot element type: %s\n", typeid(pivot).name());
        
        if constexpr (detail::is_matrix_v<ComputeType>) {
            DEBUG_PRINTF("Pivot matrix size: %dx%d\n", pivot.get_rows(), pivot.get_cols());
            auto inv_pivot = pivot.inverse();
            DEBUG_PRINTF("inv_pivot size: %dx%d\n", inv_pivot.get_rows(), inv_pivot.get_cols());
            for (int j = k; j < 2 * n; ++j) {
                DEBUG_PRINTF("Normalizing column %d\n", j);
                augmented(k, j) = augmented(k, j) * inv_pivot;
            }
        } else if constexpr (detail::has_division_v<ComputeType>) {
            for (int j = k; j < 2 * n; ++j) {
                augmented(k, j) = augmented(k, j) / pivot;
            }
        } else {
            throw std::runtime_error("Cannot divide for this type");
        }
        
        DEBUG_PRINTF("Eliminating other rows\n");
        for (int i = 0; i < n; ++i) {
            if (i != k) {
                ComputeType factor = augmented(i, k);
                bool factor_is_zero = false;
                
                if constexpr (detail::is_matrix_v<ComputeType>) {
                    factor_is_zero = compute_block_norm(factor) < epsilon;
                } else if constexpr (detail::has_abs_v<ComputeType>) {
                    factor_is_zero = std::abs(factor) < epsilon;
                } else {
                    factor_is_zero = factor == ComputeType{};
                }
                
                if (!factor_is_zero) {
                    DEBUG_PRINTF("Eliminating row %d using row %d\n", i, k);
                    for (int j = k; j < 2 * n; ++j) {
                        augmented(i, j) = augmented(i, j) - factor * augmented(k, j);
                    }
                }
            }
        }
    }
    
    DEBUG_PRINTF("Extracting inverse matrix\n");
    Matrix<ComputeType> inv(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inv(i, j) = augmented(i, n + j);
        }
    }
    
    return inv;
}

};
