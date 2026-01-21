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
#include <x86intrin.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include "Debug_printf.h"

template<typename T> class Matrix;

namespace detail {

    template<typename T, typename = void> struct has_division : std::false_type {};
    template<typename T>
    struct has_division<T, std::void_t<decltype(std::declval<T>() / std::declval<T>())>>
        : std::true_type {};
    template<typename T> constexpr bool has_division_v = has_division<T>::value;

    template<typename T, typename = void> struct has_abs : std::false_type {};

    template<typename T>
    struct has_abs<T, std::void_t<decltype(abs(std::declval<T>()))>> : std::true_type {};

    template<typename T> constexpr bool has_abs_v = has_abs<T>::value;

    template<typename T>
    struct has_abs<Matrix<T>> : has_abs<typename Matrix<T>::value_type> {};

    template<typename T> constexpr bool is_ordered_v = std::is_arithmetic_v<T>;

    template<typename T>
    constexpr bool is_builtin_integral_v =
        std::is_same_v<T, int> || std::is_same_v<T, long> || std::is_same_v<T, long long>
        || std::is_same_v<T, unsigned int> || std::is_same_v<T, unsigned long>
        || std::is_same_v<T, unsigned long long> || std::is_same_v<T, short>
        || std::is_same_v<T, unsigned short> || std::is_same_v<T, char>
        || std::is_same_v<T, signed char> || std::is_same_v<T, unsigned char>;

    template<typename T> struct is_matrix : std::false_type {};

    template<typename T> struct is_matrix<Matrix<T>> : std::true_type {};

    template<typename T>
    struct has_division<Matrix<T>> : has_division<typename Matrix<T>::value_type> {};

    template<typename T> constexpr bool is_matrix_v = is_matrix<T>::value;

    template<typename T> constexpr bool is_avx_float = std::is_same_v<T, float>;

    template<typename T> constexpr bool is_avx_double = std::is_same_v<T, double>;

    template<typename T>
    constexpr bool is_avx_compatible = is_avx_float<T> || is_avx_double<T>;

    template<typename T> struct remove_all_ref {
        using type = T;
    };

    template<typename T> struct remove_all_ref<T &> {
        using type = typename remove_all_ref<T>::type;
    };

    template<typename T> struct remove_all_ref<T &&> {
        using type = typename remove_all_ref<T>::type;
    };

    template<typename T> using remove_all_ref_t = typename remove_all_ref<T>::type;

    template<typename T, typename U> struct matrix_common_type {
    private:
        static auto test() -> decltype(true ? std::declval<remove_all_ref_t<T>>()
                                            : std::declval<remove_all_ref_t<U>>());
        using test_type = decltype(test());

    public:
        using type = remove_all_ref_t<test_type>;
    };

    template<typename T, typename U> struct matrix_common_type<Matrix<T>, Matrix<U>> {
        using type = Matrix<typename matrix_common_type<T, U>::type>;
    };

    template<typename T, typename U>
    using matrix_common_type_t = typename matrix_common_type<T, U>::type;

    template<typename T, typename U, typename = void> struct result_type_mul_impl {
        using type = void;
    };

    template<typename T, typename U>
    struct result_type_mul_impl<
        T,
        U,
        std::void_t<decltype(std::declval<remove_all_ref_t<T>>()
                             * std::declval<remove_all_ref_t<U>>())>> {
        using type = remove_all_ref_t<decltype(std::declval<remove_all_ref_t<T>>()
                                               * std::declval<remove_all_ref_t<U>>())>;
    };

    template<typename T, typename U>
    struct result_type_mul : result_type_mul_impl<T, U> {};

    template<typename T, typename U>
    using result_type_mul_t = typename result_type_mul<T, U>::type;

    template<typename T, typename U, typename = void> struct result_type_add_impl {
        using type = void;
    };

    template<typename T, typename U>
    struct result_type_add_impl<
        T,
        U,
        std::void_t<decltype(std::declval<remove_all_ref_t<T>>()
                             + std::declval<remove_all_ref_t<U>>())>> {
        using type = remove_all_ref_t<decltype(std::declval<remove_all_ref_t<T>>()
                                               + std::declval<remove_all_ref_t<U>>())>;
    };

    template<typename T, typename U>
    struct result_type_add : result_type_add_impl<T, U> {};

    template<typename T, typename U>
    using result_type_add_t = typename result_type_add<T, U>::type;

    template<typename T, typename U, typename = void> struct result_type_sub_impl {
        using type = void;
    };

    template<typename T, typename U>
    struct result_type_sub_impl<
        T,
        U,
        std::void_t<decltype(std::declval<remove_all_ref_t<T>>()
                             - std::declval<remove_all_ref_t<U>>())>> {
        using type = remove_all_ref_t<decltype(std::declval<remove_all_ref_t<T>>()
                                               - std::declval<remove_all_ref_t<U>>())>;
    };

    template<typename T, typename U>
    struct result_type_sub : result_type_sub_impl<T, U> {};

    template<typename T, typename U>
    using result_type_sub_t = typename result_type_sub<T, U>::type;

    template<typename T, typename U, typename = void> struct result_type_div_impl {
        using type = void;
    };

    template<typename T, typename U>
    struct result_type_div_impl<
        T,
        U,
        std::void_t<decltype(std::declval<remove_all_ref_t<T>>()
                             / std::declval<remove_all_ref_t<U>>())>> {
        using type = remove_all_ref_t<decltype(std::declval<remove_all_ref_t<T>>()
                                               / std::declval<remove_all_ref_t<U>>())>;
    };

    template<typename T, typename U>
    struct result_type_div : result_type_div_impl<T, U> {};

    template<typename T, typename U>
    using result_type_div_t = typename result_type_div<T, U>::type;

    template<typename T> struct inverse_return_type_impl {
        using type = T;
    };

    template<typename T> struct inverse_return_type_impl<Matrix<T>> {
        using type = Matrix<typename inverse_return_type_impl<T>::type>;
    };

    template<> struct inverse_return_type_impl<int> {
        using type = double;
    };

    template<> struct inverse_return_type_impl<long> {
        using type = double;
    };

    template<> struct inverse_return_type_impl<long long> {
        using type = double;
    };

    template<> struct inverse_return_type_impl<short> {
        using type = float;
    };

    template<> struct inverse_return_type_impl<char> {
        using type = float;
    };

    template<> struct inverse_return_type_impl<signed char> {
        using type = float;
    };

    template<> struct inverse_return_type_impl<unsigned int> {
        using type = double;
    };

    template<> struct inverse_return_type_impl<unsigned long> {
        using type = double;
    };

    template<> struct inverse_return_type_impl<unsigned long long> {
        using type = double;
    };

    template<> struct inverse_return_type_impl<unsigned short> {
        using type = float;
    };

    template<> struct inverse_return_type_impl<unsigned char> {
        using type = float;
    };

} // namespace detail

const int Default_iterations = 100;

template<typename T> class Matrix {
private:
    enum class Tranformation_types { I_TYPE = 0, II_TYPE = 1, III_TYPE = 2 };

    std::unique_ptr<std::unique_ptr<T[]>[]> matrix_;
    int rows_, cols_, min_dim_;
    mutable std::optional<T> determinant_ = std::nullopt;

    static constexpr auto Epsilon = 1e-10;

public:
    using value_type = T;

    Matrix()
        : rows_(0)
        , cols_(0)
        , min_dim_(0)
        , matrix_(nullptr) {}

    Matrix(int rows, int cols)
        : rows_(rows)
        , cols_(cols)
        , min_dim_(std::min(rows, cols)) {
        alloc_matrix_();
    }

    ~Matrix() = default;

    static Matrix Square(int size) { return Matrix(size, size); }

    static Matrix Rectangular(int rows, int cols) { return Matrix(rows, cols); }

    static Matrix Identity(int rows, int cols) {
        if constexpr (detail::is_matrix_v<T>) {
            throw std::runtime_error(
                "For block matrices use BlockIdentity() with block dimensions");
        }

        Matrix result(rows, cols);
        int min_dim = std::min(rows, cols);
        for (int i = 0; i < min_dim; i++) {
            result(i, i) = T{1};
        }
        return result;
    }

    static Matrix Identity(int rows) { return Identity(rows, rows); }

    static Matrix BlockMatrix(int rows, int cols, int block_rows, int block_cols) {
        static_assert(detail::is_matrix_v<T>,
                      "BlockMatrix can only be used with Matrix<Matrix<U>> types");

        using InnerType = typename T::value_type;
        Matrix result(rows, cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = T::Zero(block_rows, block_cols);
            }
        }
        return result;
    }

    static Matrix BlockIdentity(int rows, int cols, int block_rows, int block_cols) {
        static_assert(detail::is_matrix_v<T>,
                      "BlockIdentity can only be used with Matrix<Matrix<U>> types");

        using InnerType = typename T::value_type;
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

    static Matrix BlockZero(int rows, int cols, int block_rows, int block_cols) {
        return BlockMatrix(rows, cols, block_rows, block_cols);
    }

    static Matrix Diagonal(int rows, int cols, const std::vector<T> &diagonal) {
        Matrix result = Matrix::Zero(static_cast<int>(rows), static_cast<int>(cols));
        int diag_size =
            std::min(result.get_min_dim(), static_cast<int>(diagonal.size()));
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
        if constexpr (detail::is_matrix_v<T>) {
            throw std::runtime_error(
                "For block matrices use BlockZero() with block dimensions");
        }

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

    static Matrix Submatrix(const Matrix &source,
                            int start_row,
                            int start_col,
                            int num_rows,
                            int num_cols) {
        if (start_row < 0 || start_row >= source.rows_ || start_col < 0
            || start_col >= source.cols_ || num_rows <= 0 || num_cols <= 0
            || start_row + num_rows > source.rows_
            || start_col + num_cols > source.cols_) {
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

    Matrix
    get_submatrix(int start_row, int start_col, int num_rows, int num_cols) const {
        return Submatrix(*this, start_row, start_col, num_rows, num_cols);
    }

    static Matrix
    Submatrix(const Matrix &source, int start_row, int start_col, int size) {
        return Submatrix(source, start_row, start_col, size, size);
    }

    static Matrix Generate_matrix(int rows,
                                  int cols,
                                  T min_val = {},
                                  T max_val = {},
                                  int iterations = Default_iterations,
                                  T target_determinant_magnitude = T{1},
                                  T max_condition_number = {}) {
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

    static Matrix Generate_binary_matrix(int rows,
                                         int cols,
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

    Matrix(const Matrix &other)
        : rows_(other.rows_)
        , cols_(other.cols_)
        , min_dim_(other.min_dim_) {
        alloc_matrix_();
        if (matrix_) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    (*this)(i, j) = other(i, j);
                }
            }
        }
    }

    Matrix &operator=(const Matrix &other) {
        if (this != &other) {
            matrix_.reset();
            rows_ = other.rows_;
            cols_ = other.cols_;
            min_dim_ = other.min_dim_;

            alloc_matrix_();
            if (matrix_) {
                for (int i = 0; i < rows_; ++i) {
                    for (int j = 0; j < cols_; ++j) {
                        (*this)(i, j) = other(i, j);
                    }
                }
            }
        }
        return *this;
    }

    Matrix(Matrix &&) = default;
    Matrix &operator=(Matrix &&) = default;

    T &operator()(int i, int j) { return matrix_[i][j]; }
    const T &operator()(int i, int j) const { return matrix_[i][j]; }

    template<typename U> Matrix &operator+=(const Matrix<U> &other) {
        *this = *this + other;
        return *this;
    }

    template<typename U> Matrix &operator-=(const Matrix<U> &other) {
        *this = *this - other;
        return *this;
    }

    Matrix &operator*=(const Matrix &other) {
        *this = *this * other;
        return *this;
    }

    template<typename U> Matrix &operator*=(const U &scalar) {
        *this = *this * scalar;
        return *this;
    }

    template<typename U> Matrix &operator/=(const U &scalar) {
        *this = *this / scalar;
        return *this;
    }

    bool operator==(const Matrix &other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            return false;
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (!is_equal((*this)(i, j), other(i, j)))
                    return false;
            }
        }
        return true;
    }

    bool operator!=(const Matrix &other) const { return !(*this == other); }

    template<typename U = T>
    static bool is_equal(const U& a, const U& b) {
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
            }
            return true;
        } else {
            if constexpr (std::is_floating_point_v<U>) {
                return std::abs(a - b) < Epsilon;
            } else {
                return a == b;
            }
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

    template<typename U = T> static bool is_zero(const U &value) {
        if constexpr (detail::is_matrix_v<U>) {
            auto det_opt = value.det();
            return !det_opt.has_value() || is_zero(*det_opt);
        } else if constexpr (std::is_same_v<U, float> || std::is_same_v<U, double>) {
            return std::abs(value) < Epsilon;
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
            matrix_[target_row][j] =
                matrix_[target_row][j] + matrix_[source_row][j] * scalar;
    }

    void print() const {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++)
                std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                          << std::defaultfloat << matrix_[i][j] << ' ';

            std::cout << "\n\n";
        }
    }

    void print(int max_size) const {
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

    void precise_print(int precision = 15) const {
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

    void detailed_print() const {
        if constexpr (detail::is_matrix_v<T>) {
            std::cout << "Block Matrix " << rows_ << "x" << cols_ << " of "
                      << (*this)(0, 0).get_rows() << "x" << (*this)(0, 0).get_cols()
                      << " blocks:\n";

            std::cout << std::fixed << std::setprecision(2);

            for (int i = 0; i < rows_; ++i) {
                for (int inner_row = 0; inner_row < (*this)(0, 0).get_rows();
                     ++inner_row) {
                    std::cout << "  ";
                    for (int j = 0; j < cols_; ++j) {
                        const auto &block = (*this)(i, j);

                        std::cout << "[";
                        for (int inner_col = 0; inner_col < block.get_cols();
                             ++inner_col) {
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

    friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix) {
        if constexpr (detail::is_matrix_v<T>) {
            if (matrix.rows_ <= 3 && matrix.cols_ <= 3) {
                os << "BlockMatrix " << matrix.rows_ << "x" << matrix.cols_ << ":\n";
                for (int i = 0; i < matrix.rows_; ++i) {
                    for (int inner_i = 0; inner_i < matrix(0, 0).get_rows(); ++inner_i) {
                        os << "  ";
                        for (int j = 0; j < matrix.cols_; ++j) {
                            const auto &block = matrix(i, j);
                            os << "[";
                            for (int inner_j = 0; inner_j < block.get_cols();
                                 ++inner_j) {
                                os << block(inner_i, inner_j);
                                if (inner_j < block.get_cols() - 1)
                                    os << " ";
                            }
                            os << "]";
                            if (j < matrix.cols_ - 1)
                                os << " ";
                        }
                        os << "\n";
                    }
                    if (i < matrix.rows_ - 1)
                        os << "\n";
                }
            } else {
                os << "[BlockMatrix " << matrix.rows_ << "x" << matrix.cols_ << " of "
                   << matrix(0, 0).get_rows() << "x" << matrix(0, 0).get_cols() << "]";
            }
        } else {
            if (matrix.rows_ <= 5 && matrix.cols_ <= 5) {
                for (int i = 0; i < matrix.rows_; ++i) {
                    for (int j = 0; j < matrix.cols_; ++j) {
                        os << matrix(i, j);
                        if (j < matrix.cols_ - 1)
                            os << " ";
                    }
                    if (i < matrix.rows_ - 1)
                        os << "\n";
                }
            } else {
                os << "[Matrix " << matrix.rows_ << "x" << matrix.cols_ << "]";
            }
        }
        return os;
    }

    void fill_upper_triangle(T min_val = {}, T max_val = {}) {
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

    std::optional<T> det(int row, int col, int size) const {
        if constexpr (detail::is_builtin_integral_v<T>) {
            return det_integer_algorithm(row, col, size);
        } else {
            static_assert(detail::has_division_v<T>,
                          "Numeric determinant requires operator/ for type T.");
            return det_numeric_impl(row, col, size);
        }
    }

    std::optional<T> det() const { return det(0, 0, min_dim_); }

    template<typename U = T>
    using inverse_return_type = typename detail::inverse_return_type_impl<U>::type;

    friend auto operator/(const Matrix &A, const Matrix &B) {
        if (A.cols_ != B.rows_) {
            throw std::invalid_argument(
                "Matrix dimensions don't match for division (A.cols != B.rows)");
        }

        try {
            using ResultType = inverse_return_type<T>;

            Matrix<ResultType> A_cast = A.template cast_to<ResultType>();
            Matrix<ResultType> B_cast = B.template cast_to<ResultType>();

            Matrix<ResultType> B_inv = B_cast.inverse();

            return A_cast * B_inv;
        } catch (const std::exception &e) {
            throw std::runtime_error("Cannot divide by singular matrix B");
        }
    }

    template<typename U> Matrix<U> cast_to() const {
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

    template<typename U> operator Matrix<U>() const {
        using NonRefU = std::remove_reference_t<U>;
        static_assert(!std::is_reference_v<NonRefU>, "Cannot cast to reference type");

        Matrix<NonRefU> result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result(i, j) = static_cast<NonRefU>((*this)(i, j));
            }
        }
        return result;
    }

    template<typename ComputeType = inverse_return_type<T>>
    Matrix<ComputeType> inverse() const {
        if (rows_ != cols_) {
            throw std::invalid_argument("Matrix must be square");
        }

        if constexpr (std::is_integral_v<T> && !detail::is_matrix_v<T>) {
            using RealType = std::conditional_t<sizeof(T) >= 8, double, float>;
            Matrix<RealType> real_matrix = this->template cast_to<RealType>();
            return real_matrix.inverse();
        }

        constexpr bool is_block_matrix = detail::is_matrix_v<T>;
        constexpr bool use_abs = detail::has_abs_v<ComputeType>;

        return inverse_impl<ComputeType, is_block_matrix, use_abs>();
    }

    template<typename U = T>
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
    void alloc_matrix_() {
        if (rows_ <= 0 || cols_ <= 0) {
            matrix_ = nullptr;
            return;
        }

        matrix_ = std::make_unique<std::unique_ptr<T[]>[]>(rows_);
        for (int i = 0; i < rows_; ++i) {
            matrix_[i] = std::make_unique<T[]>(cols_);
        }
    }

    void init_zero_() {
        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                matrix_[i][j] = T{};
    }

    static std::vector<T> create_controlled_diagonal(int size,
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

    static void apply_transformation_type_II(Matrix &matrix,
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
                    if (abs_det > T{1e100}) {
                        stabilize_matrix(matrix, rows, cols, effective_det);
                    }
                }
            }
        }
    }

    static void apply_transformation_type_III(Matrix &matrix,
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

    static void apply_controlled_transformations(Matrix &matrix,
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

    std::optional<T> det_integer_algorithm(int row, int col, int size) const {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_ || row + size > rows_
            || col + size > cols_ || size <= 0) {
            return std::nullopt;
        }

        if (size == 1) {
            determinant_ = (*this)(row, col);
            return determinant_;
        }

        Matrix<T> work_matrix = Submatrix(*this, row, col, size, size);
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
                    long long numerator =
                        static_cast<long long>(work_matrix(i, j)) * current_pivot
                        - static_cast<long long>(work_matrix(i, step))
                              * static_cast<long long>(work_matrix(step, j));

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

        if (final_det > std::numeric_limits<T>::max()
            || final_det < std::numeric_limits<T>::min()) {
            return std::nullopt;
        }

        determinant_ = static_cast<T>(final_det);
        return determinant_;
    }

    template<typename U = T> static U identity_element(int rows, int cols) {
        if constexpr (detail::is_matrix_v<U>) {
            return U::Identity(rows, cols);
        } else {
            return U{1};
        }
    }

    template<typename U = T> static U zero_element(int rows, int cols) {
        if constexpr (detail::is_matrix_v<U>) {
            return U::Zero(rows, cols);
        } else {
            return U{0};
        }
    }

    template<typename U = T>
    std::optional<T> det_numeric_impl(int row, int col, int size) const {
        int block_rows = 1, block_cols = 1;
        if constexpr (detail::is_matrix_v<T>) {
            if (size > 0) {
                block_rows = (*this)(row, col).get_rows();
                block_cols = (*this)(row, col).get_cols();
            }
        }

        if (row < 0 || row >= rows_ || col < 0 || col >= cols_ || row + size > rows_
            || col + size > cols_ || size <= 0) {
            DEBUG_PRINTF("Invalid parameters!\n");
            return std::nullopt;
        }

        if (size == 1) {
            determinant_.emplace((*this)(row, col));
            return determinant_;
        }

        Matrix<T> matrix_cpy = Submatrix(*this, row, col, size, size);
        T determinant = identity_element<T>(block_rows, block_cols);
        int sign = 1;

        for (int j = 0; j < size; ++j) {
            std::optional<int> max_index_opt =
                matrix_cpy.template find_pivot_in_subcol<T>(j, j);
            if (!max_index_opt) {
                DEBUG_PRINTF("No pivot found, determinant is zero\n");
                determinant_.emplace(zero_element<T>(block_rows, block_cols));
                return zero_element<T>(block_rows, block_cols);
            }

            int max_index = *max_index_opt;

            if (max_index != j) {
                matrix_cpy.swap_rows(max_index, j);
                sign = -sign;
            }

            T pivot = matrix_cpy(j, j);

            if (is_element_zero(pivot)) {
                DEBUG_PRINTF("Pivot is zero, determinant is zero\n");
                determinant_.emplace(zero_element<T>(block_rows, block_cols));
                return zero_element<T>(block_rows, block_cols);
            }

            determinant = determinant * pivot;

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

    std::optional<T> &get_determinant_() { return determinant_; }

    template<typename ComputeType, bool IsBlockMatrix>
    Matrix<ComputeType> create_augmented_matrix() const {
        int n = rows_;

        if constexpr (IsBlockMatrix) {
            using InnerType = typename T::value_type;
            int block_rows = (*this)(0, 0).get_rows();
            int block_cols = (*this)(0, 0).get_cols();

            Matrix<ComputeType> augmented(n, 2 * n);

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    augmented(i, j) = (*this)(i, j);
                }

                for (int j = 0; j < n; ++j) {
                    if (i == j) {
                        augmented(i, n + j) = T::Identity(block_rows, block_cols);
                    } else {
                        augmented(i, n + j) = T::Zero(block_rows, block_cols);
                    }
                }
            }

            return augmented;
        } else {
            Matrix<ComputeType> augmented(n, 2 * n);

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    augmented(i, j) = static_cast<ComputeType>((*this)(i, j));
                }
                augmented(i, n + i) = ComputeType(1);
            }

            return augmented;
        }
    }

    template<typename ComputeType>
    Matrix<ComputeType> extract_inverse(const Matrix<ComputeType> &augmented) const {
        int n = augmented.get_rows();
        Matrix<ComputeType> inv(n, n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                inv(i, j) = augmented(i, n + j);
            }
        }

        return inv;
    }

    template<typename ComputeType, bool IsBlockMatrix, bool UseAbs>
    Matrix<ComputeType> inverse_impl() const {
        int n = rows_;

        if (n == 0) {
            return Matrix<ComputeType>();
        }

        Matrix<ComputeType> augmented =
            create_augmented_matrix<ComputeType, IsBlockMatrix>();

        for (int k = 0; k < n; ++k) {
            int pivot_row = -1;

            if constexpr (UseAbs) {
                double max_norm = 0.0;

                for (int i = k; i < n; ++i) {
                    double norm = 0.0;

                    if constexpr (IsBlockMatrix) {
                        norm = compute_block_norm(augmented(i, k));
                    } else {
                        using std::abs;
                        norm = abs(augmented(i, k));
                    }

                    if (norm > max_norm) {
                        max_norm = norm;
                        pivot_row = i;
                    }
                }

                if (max_norm < Epsilon) {
                    throw std::runtime_error("Matrix is singular - all zero column");
                }
            } else {
                for (int i = k; i < n; ++i) {
                    bool is_non_zero = false;

                    if constexpr (IsBlockMatrix) {
                        is_non_zero = (compute_block_norm(augmented(i, k)) >= Epsilon);
                    } else {
                        is_non_zero = !is_element_zero(augmented(i, k));
                    }

                    if (is_non_zero) {
                        pivot_row = i;
                        break;
                    }
                }

                if (pivot_row == -1) {
                    throw std::runtime_error("Matrix is singular - all zero column");
                }
            }

            if (pivot_row != k) {
                augmented.swap_rows(k, pivot_row);
            }

            try {
                normalize_row<ComputeType, IsBlockMatrix>(augmented, k);
            } catch (const std::exception &e) {
                std::string error = "Failed to normalize row: ";
                error += e.what();
                throw std::runtime_error(error);
            }

            eliminate_other_rows<ComputeType, IsBlockMatrix, UseAbs>(augmented, k);
        }

        return extract_inverse<ComputeType>(augmented);
    }

    template<typename ComputeType, bool IsBlockMatrix>
    void normalize_row(Matrix<ComputeType> &augmented, int row) const {
        int n = augmented.get_rows();
        ComputeType pivot = augmented(row, row);

        if (is_element_zero(pivot)) {
            throw std::runtime_error("Matrix is singular - zero pivot");
        }

        if constexpr (IsBlockMatrix) {
            ComputeType pivot_inv = pivot.inverse();
            for (int j = row; j < 2 * n; ++j) {
                augmented(row, j) = pivot_inv * augmented(row, j); // УМНОЖЕНИЕ СЛЕВА!
            }
        } else {
            for (int j = row; j < 2 * n; ++j) {
                augmented(row, j) = augmented(row, j) / pivot;
            }
        }
    }

    template<typename ComputeType, bool IsBlockMatrix, bool UseAbs>
    void eliminate_other_rows(Matrix<ComputeType> &augmented, int pivot_row) const {
        int n = augmented.get_rows();

        for (int i = 0; i < n; ++i) {
            if (i != pivot_row) {
                ComputeType factor;

                if constexpr (IsBlockMatrix) {
                    factor = augmented(i, pivot_row)
                             * augmented(pivot_row, pivot_row).inverse();
                } else {
                    factor = augmented(i, pivot_row) / augmented(pivot_row, pivot_row);
                }

                if (!is_element_zero(factor)) {
                    for (int j = pivot_row; j < 2 * n; ++j) {
                        augmented(i, j) =
                            augmented(i, j) - factor * augmented(pivot_row, j);
                    }
                }
            }
        }
    }

    template<typename U> static double compute_block_norm(const U &block) {
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

    template<typename U> static bool is_element_zero(const U &elem) {
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
}; // class Matrix

namespace matrix_ops_detail {
    template<typename T, typename U>
    using CommonType = std::remove_reference_t<detail::matrix_common_type_t<T, U>>;

    template<typename T, typename U>
    using AddResult = std::remove_reference_t<
        detail::result_type_add_t<CommonType<T, U>, CommonType<T, U>>>;

    template<typename T, typename U>
    using SubResult = std::remove_reference_t<
        detail::result_type_sub_t<CommonType<T, U>, CommonType<T, U>>>;

    template<typename T, typename U>
    using MulResult = std::remove_reference_t<detail::result_type_mul_t<T, U>>;

    template<typename T, typename U>
    using DivResult = std::remove_reference_t<detail::result_type_div_t<T, U>>;

    template<typename ResultType, typename T>
    void init_block_matrix(Matrix<ResultType> &result, const Matrix<T> &lhs) {
        if constexpr (detail::is_matrix_v<ResultType>) {
            if (lhs.get_rows() > 0 && lhs.get_cols() > 0) {
                int inner_rows = lhs(0, 0).get_rows();
                int inner_cols = lhs(0, 0).get_cols();
                for (int i = 0; i < result.get_rows(); ++i) {
                    for (int j = 0; j < result.get_cols(); ++j) {
                        result(i, j) = ResultType::Zero(inner_rows, inner_cols);
                    }
                }
            }
        }
    }

    template<typename T, typename U, typename BinaryOp>
    Matrix<typename std::invoke_result<BinaryOp, T, U>::type>
    perform_elementwise(const Matrix<T> &lhs, const Matrix<U> &rhs, BinaryOp op) {
        if (lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        using ResultType = typename std::invoke_result<BinaryOp, T, U>::type;
        Matrix<ResultType> result(lhs.get_rows(), lhs.get_cols());
        init_block_matrix<ResultType, T>(result, lhs);

        for (int i = 0; i < lhs.get_rows(); ++i) {
            for (int j = 0; j < lhs.get_cols(); ++j) {
                result(i, j) = op(lhs(i, j), rhs(i, j));
            }
        }

        return result;
    }

    template<typename T, typename U, typename BinaryOp>
    Matrix<typename std::invoke_result<BinaryOp, T, U>::type>
    perform_scalar_op(const Matrix<T> &matrix, const U &scalar, BinaryOp op) {
        using ResultType = typename std::invoke_result<BinaryOp, T, U>::type;
        Matrix<ResultType> result(matrix.get_rows(), matrix.get_cols());

        for (int i = 0; i < matrix.get_rows(); ++i) {
            for (int j = 0; j < matrix.get_cols(); ++j) {
                result(i, j) = op(matrix(i, j), scalar);
            }
        }

        return result;
    }
} // namespace matrix_ops_detail

template<typename T, typename U> auto operator*(const Matrix<T> &A, const Matrix<U> &B) {
    if (A.get_cols() != B.get_rows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    using CommonT = std::remove_reference_t<detail::matrix_common_type_t<T, U>>;
    using ResultType =
        std::remove_reference_t<detail::result_type_mul_t<CommonT, CommonT>>;

    Matrix<ResultType> result(A.get_rows(), B.get_cols());

    if constexpr (detail::is_matrix_v<ResultType>) {
        if (A.get_rows() > 0 && A.get_cols() > 0 && B.get_cols() > 0) {
            int inner_rows = A(0, 0).get_rows();
            int inner_cols = B(0, 0).get_cols();

            for (int i = 0; i < result.get_rows(); ++i) {
                for (int j = 0; j < result.get_cols(); ++j) {
                    result(i, j) = ResultType::Zero(inner_rows, inner_cols);
                }
            }
        }
    }

#ifdef __AVX__
    if constexpr (std::is_same_v<T, U> && detail::is_avx_double<T>) {
        for (int i = 0; i < A.get_rows(); ++i) {
            for (int k = 0; k < A.get_cols(); ++k) {
                double a_ik = A(i, k);
                __m256d a_vec = _mm256_set1_pd(a_ik);
                int j = 0;
                for (; j + 4 <= B.get_cols(); j += 4) {
                    double b_temp[4] = {B(k, j), B(k, j + 1), B(k, j + 2), B(k, j + 3)};
                    double c_temp[4] = {result(i, j),
                                        result(i, j + 1),
                                        result(i, j + 2),
                                        result(i, j + 3)};

                    __m256d b_vec = _mm256_loadu_pd(b_temp);
                    __m256d c_vec = _mm256_loadu_pd(c_temp);
                    c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec));
                    _mm256_storeu_pd(c_temp, c_vec);

                    result(i, j) = c_temp[0];
                    result(i, j + 1) = c_temp[1];
                    result(i, j + 2) = c_temp[2];
                    result(i, j + 3) = c_temp[3];
                }
                for (; j < B.get_cols(); ++j) {
                    result(i, j) += A(i, k) * B(k, j);
                }
            }
        }
    } else if constexpr (std::is_same_v<T, U> && detail::is_avx_float<T>) {
        for (int i = 0; i < A.get_rows(); ++i) {
            for (int k = 0; k < A.get_cols(); ++k) {
                float a_ik = A(i, k);
                __m256 a_vec = _mm256_set1_ps(a_ik);
                int j = 0;
                for (; j + 8 <= B.get_cols(); j += 8) {
                    float b_temp[8] = {B(k, j),
                                       B(k, j + 1),
                                       B(k, j + 2),
                                       B(k, j + 3),
                                       B(k, j + 4),
                                       B(k, j + 5),
                                       B(k, j + 6),
                                       B(k, j + 7)};
                    float c_temp[8] = {result(i, j),
                                       result(i, j + 1),
                                       result(i, j + 2),
                                       result(i, j + 3),
                                       result(i, j + 4),
                                       result(i, j + 5),
                                       result(i, j + 6),
                                       result(i, j + 7)};

                    __m256 b_vec = _mm256_loadu_ps(b_temp);
                    __m256 c_vec = _mm256_loadu_ps(c_temp);
                    c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));
                    _mm256_storeu_ps(c_temp, c_vec);

                    for (int n = 0; n < 8; ++n) {
                        result(i, j + n) = c_temp[n];
                    }
                }
                for (; j < B.get_cols(); ++j) {
                    result(i, j) += A(i, k) * B(k, j);
                }
            }
        }
    } else {
        for (int i = 0; i < A.get_rows(); ++i) {
            for (int k = 0; k < A.get_cols(); ++k) {
                CommonT a_ik = static_cast<CommonT>(A(i, k));
                for (int j = 0; j < B.get_cols(); ++j) {
                    result(i, j) += a_ik * static_cast<CommonT>(B(k, j));
                }
            }
        }
    }
#else
    for (int i = 0; i < A.get_rows(); ++i) {
        for (int k = 0; k < A.get_cols(); ++k) {
            CommonT a_ik = static_cast<CommonT>(A(i, k));
            for (int j = 0; j < B.get_cols(); ++j) {
                result(i, j) += a_ik * static_cast<CommonT>(B(k, j));
            }
        }
    }
#endif

    return result;
}

template<typename T, typename U>
auto operator+(const Matrix<T> &lhs, const Matrix<U> &rhs) {
    using namespace matrix_ops_detail;

    if (lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    using CommonT = CommonType<T, U>;
    using ResultType = AddResult<T, U>;

    Matrix<ResultType> result(lhs.get_rows(), lhs.get_cols());
    init_block_matrix<ResultType, T>(result, lhs);

#ifdef __AVX__
    if constexpr (std::is_same_v<T, U> && detail::is_avx_double<T>) {
        for (int i = 0; i < lhs.get_rows(); ++i) {
            int j = 0;
            for (; j + 4 <= lhs.get_cols(); j += 4) {
                double lhs_temp[4] = {lhs(i, j),
                                      lhs(i, j + 1),
                                      lhs(i, j + 2),
                                      lhs(i, j + 3)};
                double rhs_temp[4] = {rhs(i, j),
                                      rhs(i, j + 1),
                                      rhs(i, j + 2),
                                      rhs(i, j + 3)};

                __m256d lhs_vec = _mm256_loadu_pd(lhs_temp);
                __m256d rhs_vec = _mm256_loadu_pd(rhs_temp);
                __m256d res_vec = _mm256_add_pd(lhs_vec, rhs_vec);
                _mm256_storeu_pd(lhs_temp, res_vec);

                result(i, j) = lhs_temp[0];
                result(i, j + 1) = lhs_temp[1];
                result(i, j + 2) = lhs_temp[2];
                result(i, j + 3) = lhs_temp[3];
            }
            for (; j < lhs.get_cols(); ++j) {
                result(i, j) = lhs(i, j) + rhs(i, j);
            }
        }
        return result;
    } else if constexpr (std::is_same_v<T, U> && detail::is_avx_float<T>) {
        for (int i = 0; i < lhs.get_rows(); ++i) {
            int j = 0;
            for (; j + 8 <= lhs.get_cols(); j += 8) {
                float lhs_temp[8] = {lhs(i, j),
                                     lhs(i, j + 1),
                                     lhs(i, j + 2),
                                     lhs(i, j + 3),
                                     lhs(i, j + 4),
                                     lhs(i, j + 5),
                                     lhs(i, j + 6),
                                     lhs(i, j + 7)};
                float rhs_temp[8] = {rhs(i, j),
                                     rhs(i, j + 1),
                                     rhs(i, j + 2),
                                     rhs(i, j + 3),
                                     rhs(i, j + 4),
                                     rhs(i, j + 5),
                                     rhs(i, j + 6),
                                     rhs(i, j + 7)};

                __m256 lhs_vec = _mm256_loadu_ps(lhs_temp);
                __m256 rhs_vec = _mm256_loadu_ps(rhs_temp);
                __m256 res_vec = _mm256_add_ps(lhs_vec, rhs_vec);
                _mm256_storeu_ps(lhs_temp, res_vec);

                for (int n = 0; n < 8; ++n) {
                    result(i, j + n) = lhs_temp[n];
                }
            }
            for (; j < lhs.get_cols(); ++j) {
                result(i, j) = lhs(i, j) + rhs(i, j);
            }
        }
        return result;
    }
#endif

    return perform_elementwise(lhs, rhs, [](auto a, auto b) {
        using ResultType = decltype(a + b);
        return static_cast<ResultType>(a) + static_cast<ResultType>(b);
    });
}

template<typename T, typename U>
auto operator-(const Matrix<T> &lhs, const Matrix<U> &rhs) {
    using namespace matrix_ops_detail;

    if (lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    using CommonT = CommonType<T, U>;
    using ResultType = SubResult<T, U>;

    Matrix<ResultType> result(lhs.get_rows(), lhs.get_cols());
    init_block_matrix<ResultType, T>(result, lhs);

#ifdef __AVX__
    if constexpr (std::is_same_v<T, U> && detail::is_avx_double<T>) {
        for (int i = 0; i < lhs.get_rows(); ++i) {
            int j = 0;
            for (; j + 4 <= lhs.get_cols(); j += 4) {
                double lhs_temp[4] = {lhs(i, j),
                                      lhs(i, j + 1),
                                      lhs(i, j + 2),
                                      lhs(i, j + 3)};
                double rhs_temp[4] = {rhs(i, j),
                                      rhs(i, j + 1),
                                      rhs(i, j + 2),
                                      rhs(i, j + 3)};

                __m256d lhs_vec = _mm256_loadu_pd(lhs_temp);
                __m256d rhs_vec = _mm256_loadu_pd(rhs_temp);
                __m256d res_vec = _mm256_sub_pd(lhs_vec, rhs_vec);
                _mm256_storeu_pd(lhs_temp, res_vec);

                result(i, j) = lhs_temp[0];
                result(i, j + 1) = lhs_temp[1];
                result(i, j + 2) = lhs_temp[2];
                result(i, j + 3) = lhs_temp[3];
            }
            for (; j < lhs.get_cols(); ++j) {
                result(i, j) = lhs(i, j) - rhs(i, j);
            }
        }
        return result;
    } else if constexpr (std::is_same_v<T, U> && detail::is_avx_float<T>) {
        for (int i = 0; i < lhs.get_rows(); ++i) {
            int j = 0;
            for (; j + 8 <= lhs.get_cols(); j += 8) {
                float lhs_temp[8] = {lhs(i, j),
                                     lhs(i, j + 1),
                                     lhs(i, j + 2),
                                     lhs(i, j + 3),
                                     lhs(i, j + 4),
                                     lhs(i, j + 5),
                                     lhs(i, j + 6),
                                     lhs(i, j + 7)};
                float rhs_temp[8] = {rhs(i, j),
                                     rhs(i, j + 1),
                                     rhs(i, j + 2),
                                     rhs(i, j + 3),
                                     rhs(i, j + 4),
                                     rhs(i, j + 5),
                                     rhs(i, j + 6),
                                     rhs(i, j + 7)};

                __m256 lhs_vec = _mm256_loadu_ps(lhs_temp);
                __m256 rhs_vec = _mm256_loadu_ps(rhs_temp);
                __m256 res_vec = _mm256_sub_ps(lhs_vec, rhs_vec);
                _mm256_storeu_ps(lhs_temp, res_vec);

                for (int n = 0; n < 8; ++n) {
                    result(i, j + n) = lhs_temp[n];
                }
            }
            for (; j < lhs.get_cols(); ++j) {
                result(i, j) = lhs(i, j) - rhs(i, j);
            }
        }
        return result;
    }
#endif

    return perform_elementwise(lhs, rhs, [](auto a, auto b) {
        using ResultType = decltype(a - b);
        return static_cast<ResultType>(a) - static_cast<ResultType>(b);
    });
}

template<typename T, typename U>
auto operator*(const Matrix<T> &matrix, const U &scalar) {
    using namespace matrix_ops_detail;
    using ResultType = MulResult<T, U>;

    Matrix<ResultType> result(matrix.get_rows(), matrix.get_cols());

#ifdef __AVX__
    if constexpr (std::is_same_v<T, U> && detail::is_avx_double<T>) {
        __m256d scalar_vec = _mm256_set1_pd(scalar);
        for (int i = 0; i < matrix.get_rows(); ++i) {
            int j = 0;
            for (; j + 4 <= matrix.get_cols(); j += 4) {
                double temp[4] = {matrix(i, j),
                                  matrix(i, j + 1),
                                  matrix(i, j + 2),
                                  matrix(i, j + 3)};
                __m256d vec = _mm256_loadu_pd(temp);
                vec = _mm256_mul_pd(vec, scalar_vec);
                _mm256_storeu_pd(temp, vec);

                result(i, j) = temp[0];
                result(i, j + 1) = temp[1];
                result(i, j + 2) = temp[2];
                result(i, j + 3) = temp[3];
            }
            for (; j < matrix.get_cols(); ++j) {
                result(i, j) = matrix(i, j) * scalar;
            }
        }
        return result;
    } else if constexpr (std::is_same_v<T, U> && detail::is_avx_float<T>) {
        __m256 scalar_vec = _mm256_set1_ps(scalar);
        for (int i = 0; i < matrix.get_rows(); ++i) {
            int j = 0;
            for (; j + 8 <= matrix.get_cols(); j += 8) {
                float temp[8] = {matrix(i, j),
                                 matrix(i, j + 1),
                                 matrix(i, j + 2),
                                 matrix(i, j + 3),
                                 matrix(i, j + 4),
                                 matrix(i, j + 5),
                                 matrix(i, j + 6),
                                 matrix(i, j + 7)};
                __m256 vec = _mm256_loadu_ps(temp);
                vec = _mm256_mul_ps(vec, scalar_vec);
                _mm256_storeu_ps(temp, vec);

                for (int k = 0; k < 8; ++k) {
                    result(i, j + k) = temp[k];
                }
            }
            for (; j < matrix.get_cols(); ++j) {
                result(i, j) = matrix(i, j) * scalar;
            }
        }
        return result;
    }
#endif

    return perform_scalar_op(matrix, scalar, [](auto a, auto b) {
        using ResultType = decltype(a * b);
        return static_cast<ResultType>(a) * static_cast<ResultType>(b);
    });
}

template<typename T, typename U>
auto operator/(const Matrix<T> &matrix, const U &scalar) {
    using namespace matrix_ops_detail;
    using ResultType = DivResult<T, U>;

    Matrix<ResultType> result(matrix.get_rows(), matrix.get_cols());

#ifdef __AVX__
    if constexpr (std::is_same_v<T, U> && detail::is_avx_double<T>) {
        __m256d inv_vec = _mm256_set1_pd(1.0 / scalar);
        for (int i = 0; i < matrix.get_rows(); ++i) {
            int j = 0;
            for (; j + 4 <= matrix.get_cols(); j += 4) {
                double temp[4] = {matrix(i, j),
                                  matrix(i, j + 1),
                                  matrix(i, j + 2),
                                  matrix(i, j + 3)};
                __m256d vec = _mm256_loadu_pd(temp);
                vec = _mm256_mul_pd(vec, inv_vec);
                _mm256_storeu_pd(temp, vec);

                result(i, j) = temp[0];
                result(i, j + 1) = temp[1];
                result(i, j + 2) = temp[2];
                result(i, j + 3) = temp[3];
            }
            for (; j < matrix.get_cols(); ++j) {
                result(i, j) = matrix(i, j) / scalar;
            }
        }
        return result;
    } else if constexpr (std::is_same_v<T, U> && detail::is_avx_float<T>) {
        __m256 inv_vec = _mm256_set1_ps(1.0f / scalar);
        for (int i = 0; i < matrix.get_rows(); ++i) {
            int j = 0;
            for (; j + 8 <= matrix.get_cols(); j += 8) {
                float temp[8] = {matrix(i, j),
                                 matrix(i, j + 1),
                                 matrix(i, j + 2),
                                 matrix(i, j + 3),
                                 matrix(i, j + 4),
                                 matrix(i, j + 5),
                                 matrix(i, j + 6),
                                 matrix(i, j + 7)};
                __m256 vec = _mm256_loadu_ps(temp);
                vec = _mm256_mul_ps(vec, inv_vec);
                _mm256_storeu_ps(temp, vec);

                for (int k = 0; k < 8; ++k) {
                    result(i, j + k) = temp[k];
                }
            }
            for (; j < matrix.get_cols(); ++j) {
                result(i, j) = matrix(i, j) / scalar;
            }
        }
        return result;
    }
#endif

    return perform_scalar_op(matrix, scalar, [](auto a, auto b) {
        using ResultType = decltype(a / b);
        return static_cast<ResultType>(a) / static_cast<ResultType>(b);
    });
}

template<typename T, typename U>
auto operator*(const U &scalar, const Matrix<T> &matrix) {
    return matrix * scalar;
}

template<typename T, typename U>
auto operator+(const Matrix<T> &matrix, const U &scalar) {
    using ResultType = detail::result_type_add_t<T, U>;
    Matrix<ResultType> result(matrix.get_rows(), matrix.get_cols());

    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int j = 0; j < matrix.get_cols(); ++j) {
            if (i == j) {
                result(i, j) = matrix(i, j) + scalar;
            } else {
                result(i, j) = matrix(i, j);
            }
        }
    }

    return result;
}

template<typename T, typename U>
auto operator+(const U &scalar, const Matrix<T> &matrix) {
    return matrix + scalar;
}

template<typename T, typename U>
auto operator-(const U &scalar, const Matrix<T> &matrix) {
    using ResultType = detail::result_type_sub_t<U, T>;
    Matrix<ResultType> result(matrix.get_rows(), matrix.get_cols());

    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int j = 0; j < matrix.get_cols(); ++j) {
            if (i == j) {
                result(i, j) = scalar - matrix(i, j);
            } else {
                result(i, j) = -matrix(i, j);
            }
        }
    }

    return result;
}

template<typename T, typename U>
auto operator-(const Matrix<T> &matrix, const U &scalar) {
    using ResultType = detail::result_type_sub_t<T, U>;
    Matrix<ResultType> result(matrix.get_rows(), matrix.get_cols());

    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int j = 0; j < matrix.get_cols(); ++j) {
            if (i == j) {
                result(i, j) = matrix(i, j) - scalar;
            } else {
                result(i, j) = matrix(i, j);
            }
        }
    }

    return result;
}
