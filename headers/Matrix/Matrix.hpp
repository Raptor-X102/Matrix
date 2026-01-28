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
#include <complex>

#ifdef __GNUC__
#include <x86intrin.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include "Debug_printf.h"
#include "Matrix_detail.hpp"

const int Default_iterations = 100;

template<typename T> class Matrix;
template<typename T> class Vector;

namespace detail {
    template<typename T, typename U>
    using matrix_common_type_t = typename matrix_common_type<T, U>::type;
}

template<typename T> class Matrix {
private:
    enum class Tranformation_types { I_TYPE = 0, II_TYPE = 1, III_TYPE = 2 };

    std::unique_ptr<std::unique_ptr<T[]>[]> matrix_;
    int rows_, cols_, min_dim_;
    mutable std::optional<T> determinant_ = std::nullopt;

    static constexpr auto Epsilon = 1e-10;

public:
    using value_type = T;

    Matrix();
    Matrix(int rows, int cols);
    Matrix(const Matrix &other);
    Matrix(Matrix &&) = default;
    ~Matrix() = default;

    Matrix &operator=(const Matrix &other);
    Matrix &operator=(Matrix &&) = default;

    T &operator()(int i, int j);
    const T &operator()(int i, int j) const;

    template<typename U> Matrix &operator+=(const Matrix<U> &other);
    template<typename U> Matrix &operator-=(const Matrix<U> &other);
    Matrix &operator*=(const Matrix &other);
    template<typename U> Matrix &operator*=(const U &scalar);
    template<typename U> Matrix &operator/=(const U &scalar);

    bool operator==(const Matrix &other) const;
    bool operator!=(const Matrix &other) const;
    Matrix operator-() const;

    static Matrix Square(int size);
    static Matrix Rectangular(int rows, int cols);
    static Matrix Identity(int rows, int cols);
    static Matrix Identity(int rows);
    static Matrix BlockMatrix(int rows, int cols, int block_rows, int block_cols);
    static Matrix BlockIdentity(int rows, int cols, int block_rows, int block_cols);
    static Matrix BlockZero(int rows, int cols, int block_rows, int block_cols);
    static Matrix Diagonal(int rows, int cols, const std::vector<T> &diagonal);
    static Matrix Diagonal(const std::vector<T> &diagonal);
    static Matrix Diagonal(int rows, int cols, T diagonal_value);
    static Matrix Diagonal(int size, T diagonal_value);
    static Matrix From_vector(const std::vector<std::vector<T>> &input);
    static Matrix Zero(int rows, int cols);
    static Matrix Zero(int rows);
    static Matrix Read_vector();
    static Matrix Submatrix(const Matrix &source,
                            int start_row,
                            int start_col,
                            int num_rows,
                            int num_cols);
    static Matrix
    Submatrix(const Matrix &source, int start_row, int start_col, int size);

    static Matrix Generate_matrix(int rows,
                                  int cols,
                                  T min_val = {},
                                  T max_val = {},
                                  int iterations = Default_iterations,
                                  T target_determinant_magnitude = T{1},
                                  T max_condition_number = {});

    static Matrix Generate_binary_matrix(int rows,
                                         int cols,
                                         T target_determinant_magnitude = T{1},
                                         int iterations = Default_iterations / 10);

    Matrix get_submatrix(int start_row, int start_col, int num_rows, int num_cols) const;

    std::optional<T> det(int row, int col, int size) const;
    std::optional<T> det() const;

    template<typename U = T>
    using inverse_return_type = typename detail::inverse_return_type_impl<U>::type;

    template<typename ComputeType = inverse_return_type<T>>
    Matrix<ComputeType> inverse() const;

    template<typename U> Matrix<U> cast_to() const;
    template<typename U> operator Matrix<U>() const;

    int get_rows() const { return rows_; }
    int get_cols() const { return cols_; }
    int get_min_dim() const { return min_dim_; }
    std::optional<T> get_determinant() const { return determinant_; }

    template<typename U = T> static bool is_equal(const U &a, const U &b);

    template<typename U = T> 
    static bool is_zero(const U &value);

    bool is_zero(int i, int j) const;

    template<typename U = T>
    std::optional<int> find_pivot_in_subcol(int row, int col) const;

    void swap_rows(int i, int j);
    void multiply_row(int target_row, T scalar);
    void add_row_scaled(int target_row, int source_row, T scalar = T{1});

    void print() const;
    void print(int max_size) const;
    void precise_print(int precision = 15) const;
    void detailed_print() const;

    Matrix<T> transpose() const;
    Matrix<T> &transpose_in_place();
    Matrix<T> transpose_deep() const;
    Matrix<T> &transpose_deep_in_place();

    template<typename U>
    friend std::ostream &operator<<(std::ostream &os, const Matrix<U> &matrix);

    using norm_return_type = typename detail::norm_return_type_impl<T>::type;

    norm_return_type frobenius_norm() const;
    norm_return_type frobenius_norm_squared() const;
    T trace() const;

    template<typename U = T>
    using sqrt_return_type = typename detail::sqrt_return_type_impl<U>::type;

    Matrix<sqrt_return_type<T>> sqrt() const;
    bool has_square_root() const;

    template<typename ResultType> Matrix<ResultType> sqrt_2x2_impl() const;

    template<typename ResultType>
    Matrix<ResultType> sqrt_newton_impl(int max_iter = 100,
                                        ResultType tolerance = ResultType(1e-10)) const;

    template<typename ResultType> bool has_square_root_impl() const;

    template<typename ResultType> bool has_square_root_direct_impl() const;

    template<typename ResultType> bool has_square_root_via_eigen_impl() const;

    template<typename ResultType> Matrix<ResultType> sqrt_impl() const;

    std::pair<Matrix<typename Matrix<T>::template sqrt_return_type<T>>, bool> 
    safe_sqrt() const;

    static T generate_random(T min_val, T max_val);

    bool is_symmetric() const;

    template<typename U = T>
    using eigen_return_type = typename detail::eigen_return_type_impl<U>::type;
    
    std::vector<eigen_return_type<T>> eigenvalues(int max_iterations = 1000) const;
    Matrix<eigen_return_type<T>> eigenvectors(int max_iterations = 1000) const;
    std::pair<std::vector<eigen_return_type<T>>, Matrix<eigen_return_type<T>>>
    eigen(int max_iterations = Default_max_iterations) const;
    
    // Вспомогательные публичные методы
    template<typename ComputeType = eigen_return_type<T>>
    std::vector<ComputeType> eigenvalues_qr(int max_iterations = 1000) const;
    
    template<typename ComputeType = eigen_return_type<T>>
    Matrix<ComputeType> eigenvectors_qr(int max_iterations = 1000) const;
    
    template<typename ComputeType = eigen_return_type<T>>
    std::pair<std::vector<ComputeType>, Matrix<ComputeType>>
    eigen_qr(int max_iterations = 1000) const;

    template<typename ComputeType = T>
    std::pair<Matrix<ComputeType>, Matrix<ComputeType>> qr_decomposition() const;

    template<typename ComputeType>
    Matrix<ComputeType> hessenberg_form() const;

    template<typename ComputeType>
    Vector<ComputeType> householder_vector(const Vector<ComputeType>& x) const;

    template<typename ExampleType, typename ValueType>
    static auto create_scalar(const ExampleType& example, ValueType value);

    template<typename ComputeType>
    bool is_norm_zero(const ComputeType& norm_value) const;

protected:
    template<typename U = T> static U identity_element(int rows, int cols);
    template<typename U = T> static U zero_element(int rows, int cols);

private:
    void alloc_matrix_();
    void init_zero_();
    void fill_upper_triangle(T min_val = {}, T max_val = {});

    std::optional<T> &get_determinant_();

    template<typename ComputeType, bool IsBlockMatrix>
    Matrix<ComputeType> create_augmented_matrix() const;

    template<typename ComputeType>
    Matrix<ComputeType> extract_inverse(const Matrix<ComputeType> &augmented) const;

    template<typename ComputeType, bool IsBlockMatrix, bool UseAbs>
    Matrix<ComputeType> inverse_impl() const;

    template<typename ComputeType, bool IsBlockMatrix>
    void normalize_row(Matrix<ComputeType> &augmented, int row) const;

    template<typename ComputeType, bool IsBlockMatrix, bool UseAbs>
    void eliminate_other_rows(Matrix<ComputeType> &augmented, int pivot_row) const;

    static void apply_transformation_type_I(Matrix &matrix, int rows);
    static void apply_transformation_type_II(Matrix &matrix,
                                             int rows,
                                             int cols,
                                             T min_val,
                                             T max_val,
                                             T &effective_det);
    static void apply_transformation_type_III(Matrix &matrix,
                                              int rows,
                                              int cols,
                                              T min_val,
                                              T max_val);
    static void apply_controlled_transformations(Matrix &matrix,
                                                 int rows,
                                                 int cols,
                                                 T min_val,
                                                 T max_val,
                                                 int iterations);
    static void stabilize_matrix(Matrix &matrix, int rows, int cols, T &effective_det);

    std::optional<T> det_integer_algorithm(int row, int col, int size) const;

    template<typename U = T>
    std::optional<T> det_numeric_impl(int row, int col, int size) const;

    static int generate_random_int_(int min = 1, int max = 100);
    static double generate_random_double_(double min = 0.0, double max = 1.0);

    template<typename U> static double compute_block_norm(const U &block);

    template<typename U> static bool is_element_zero(const U &elem);

    enum class TransposeAlgorithm { SIMPLE, BLOCKED, SIMD_BLOCKED };

    TransposeAlgorithm select_transpose_algorithm() const;
    bool is_small_matrix() const;
    bool should_use_blocking() const;
    int compute_optimal_block_size() const;
    void transpose_impl(Matrix<T> &result) const;
    void transpose_simple(Matrix<T> &result) const;
    void transpose_blocked(Matrix<T> &result) const;
    void transpose_blocked_impl(Matrix<T> &result, int block_size) const;
    Matrix<T> transpose_deep_impl() const;

    template<typename U, int SIMD_WIDTH>
    void transpose_simd_blocked(Matrix<U> &result) const;
    void transpose_block_simd_float(Matrix<float> &result,
                                    int start_i,
                                    int start_j,
                                    int rows_in_block,
                                    int cols_in_block) const;
    void transpose_block_simd_double(Matrix<double> &result,
                                     int start_i,
                                     int start_j,
                                     int rows_in_block,
                                     int cols_in_block) const;

    template<typename ComputeType>
    void apply_householder_left(Matrix<ComputeType>& A,
                                const Vector<ComputeType>& v,
                                int k) const;

    template<typename ComputeType>
    void apply_householder_right(Matrix<ComputeType>& A,
                                 const Vector<ComputeType>& v,
                                 int k) const;

    template<typename ComputeType>
    Vector<ComputeType> back_substitution(const Matrix<ComputeType>& R,
                                          const Vector<ComputeType>& y) const;

    template<typename ComputeType>
    Vector<ComputeType> inverse_iteration(const Matrix<ComputeType>& A,
                                          const ComputeType& lambda,
                                          int max_iterations = 50) const;

    template<typename ComputeType>
    std::vector<ComputeType> extract_eigenvalues_2x2(const Matrix<ComputeType>& H, int i) const;
    
    template<typename ComputeType>
    typename Matrix<ComputeType>::norm_return_type off_diagonal_norm(const Matrix<ComputeType>& H) const;

    static constexpr int Default_max_iterations = 10000;

    static std::vector<T> create_controlled_diagonal(int size,
                                                     T min_val,
                                                     T max_val,
                                                     T target_determinant_magnitude);
};

template<typename T, typename U>
auto operator+(const Matrix<T> &lhs, const Matrix<U> &rhs);

template<typename T, typename U>
auto operator-(const Matrix<T> &lhs, const Matrix<U> &rhs);

template<typename T, typename U> auto operator*(const Matrix<T> &A, const Matrix<U> &B);

template<typename T, typename U> auto operator/(const Matrix<T> &A, const Matrix<U> &B);

template<typename T, typename U>
auto operator*(const Matrix<T> &matrix, const U &scalar);

template<typename T, typename U>
auto operator*(const U &scalar, const Matrix<T> &matrix);

template<typename T, typename U>
auto operator/(const Matrix<T> &matrix, const U &scalar);

template<typename T, typename U>
auto operator+(const Matrix<T> &matrix, const U &scalar);

template<typename T, typename U>
auto operator+(const U &scalar, const Matrix<T> &matrix);

template<typename T, typename U>
auto operator-(const Matrix<T> &matrix, const U &scalar);

template<typename T, typename U>
auto operator-(const U &scalar, const Matrix<T> &matrix);

#include "Matrix_constructors.ipp"
#include "Matrix_generators.ipp"
#include "Matrix_operators.ipp"
#include "Matrix_determinant.ipp"
#include "Matrix_inverse.ipp"
#include "Matrix_helpers.ipp"
#include "Matrix_transpose.ipp"
#include "Matrix_properties.ipp"
#include "Matrix_sqrt.ipp"
#include "Matrix_eigen.ipp"
