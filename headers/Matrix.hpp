#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include <iomanip>
#include <random>
#include <type_traits>
#include "Timer.hpp"
#include "Debug_printf.h"

const int Default_iterations = 100;

template<typename T>
class Matrix {
private:
    enum class Tranformation_types { I_TYPE = 0, II_TYPE = 1, III_TYPE = 2 };

    std::unique_ptr<std::unique_ptr<T[]>[]> matrix_;
    int rows_, cols_, min_dim_;
    std::optional<T> determinant_ = std::nullopt;

    static constexpr auto epsilon = 1e-10;

    Matrix(int rows, int cols) : rows_(rows), cols_(cols), min_dim_(std::min(rows, cols)) {

        alloc_matrix_();
    }

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

public:
    static Matrix Square(int size) {
        return Matrix(size, size);
    }

    static Matrix Rectangular(int rows, int cols) {
        return Matrix(rows, cols);
    }

    static Matrix Identity(int rows, int cols) {

        Matrix result(rows, cols);
        int min_dim = result.get_min_dim();
        for (int i = 0; i < min_dim; i++)
            result(i, i) = T{1};
        
        return result;
    }

    static Matrix Identity(int rows) {

        return Identity(rows, rows);
    }
    
    static Matrix Diagonal(int rows, int cols, const std::vector<T>& diagonal) {

        Matrix result = Matrix::Zero(static_cast<int>(rows), static_cast<int>(cols));
        int diag_size = std::min(result.get_min_dim(), static_cast<int>(diagonal.size()));
            for (int i = 0; i < diag_size; i++)
                result.matrix_[i][i] = diagonal[i]; 
        
        return result;
    }

    static Matrix Diagonal(const std::vector<T>& diagonal) {

        int min_dim = diagonal.size();
        Matrix result(min_dim);
        for (int i = 0; i < min_dim; i++)
            result.matrix_[i][i] = diagonal[i]; 
        
        return result;
    }

    static Matrix From_vector(const std::vector<std::vector<T>>& input) {

        if (input.empty()) {
            return Matrix(0, 0);
        }

        size_t max_cols = 0;
        for (const auto& row : input) {
            max_cols = std::max(max_cols, row.size());
        }

        size_t rows = input.size();
        Matrix result = Matrix::Zero(static_cast<int>(rows), static_cast<int>(max_cols));
        
        for (size_t i = 0; i < rows; i++) {

            const auto& current_row = input[i];
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

    static Matrix Zero(int rows) {
        return Zero(rows, rows);
    }

    static Matrix Read_vector() {

        int n;
        //std::cout << "Введите размер матрицы n: ";
        std::cin >> n;

        Matrix matrix(n, n);

        //std::cout << "Введите элементы матрицы " << n << "x" << n << ":\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cin >> matrix(i, j);
            }
        }

        return matrix;
    }

    static Matrix Generate_matrix(int rows, int cols, 
        std::conditional_t<std::is_same_v<T, int>, int, T> min_val = {},
        std::conditional_t<std::is_same_v<T, int>, int, T> max_val = {},
        int iterations = Default_iterations
    ) {

        if constexpr (std::is_floating_point_v<T>) {

            if (min_val == T{} && max_val == T{}) {

                min_val = T{-10};
                max_val = T{10};
            }
        }

        int min_dim = std::min(rows, cols);
        std::vector<T> diagonal(min_dim);
        T determinant = T{1};
        for (int i = 0; i < min_dim; ++i) {
            
            T rand_element = generate_random(min_val, max_val);
            diagonal[i] = rand_element;
            determinant = determinant * rand_element; 
        }

        Matrix result = Matrix::Diagonal(rows, cols, diagonal);
        result.generate_upper_triangle(min_val, max_val);
        
        T sign = T{1};

        for (int i = 0; i < iterations; ++i) {

            Tranformation_types rand_transformation = static_cast<Tranformation_types>
                                                      (generate_random_int_(0, 2));

            switch (rand_transformation) {

                case Tranformation_types::I_TYPE : {
                    
                    int row_1 = generate_random_int_(0, rows - 1);
                    int row_2 = generate_random_int_(0, rows - 1);
                    if (row_1 != row_2) {

                        result.swap_rows(row_1, row_2);
                        sign = -sign;
                    }
                    break;
                }

                case Tranformation_types::II_TYPE : {

                    int row = generate_random_int_(0, rows - 1);
                    T scalar = generate_random(min_val, max_val);
                    if(!is_zero(scalar)) {

                        result.multiply_row(row, scalar);
                        determinant = determinant * scalar;
                    }
                    break;
                }
                
                case Tranformation_types::III_TYPE : {
                    
                    int row_1 = generate_random_int_(0, rows - 1);
                    int row_2 = generate_random_int_(0, rows - 1);
                    if (row_1 != row_2) {
                       
                        T scalar = generate_random(min_val, max_val);
                        result.add_row_scaled(row_1, row_2, scalar);
                    }
                    break;
                }
            }

            //result.print();
        }

        //auto det_ref = result.get_determinant_();
        result.determinant_ = T{sign} * determinant;
        return result;
    }

    Matrix(const Matrix& rhs) : rows_(rhs.rows_), cols_(rhs.cols_) {

        alloc_matrix_();
        
        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                matrix_[i][j] = rhs.matrix_[i][j];
    }

    Matrix& operator=(const Matrix& rhs) {

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

    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;

    T& operator()(int i, int j) { return matrix_[i][j]; }
    const T& operator()(int i, int j) const { return matrix_[i][j]; }
    
    int get_rows() const { return rows_; }
    int get_cols() const { return cols_; }
    int get_min_dim() { return min_dim_; }
    std::optional<T> get_determinant() { return determinant_; }

    static bool is_zero(const T& value) {

        if constexpr (std::is_floating_point_v<T>)
            return std::abs(value) < epsilon;

        else
            return value == T{};
    }

    bool is_zero(int i, int j) const {

        return is_zero((*this)(i, j)); 
    }

/********** 3 types of matrix transformation ***********/
    void swap_rows(int i, int j) {        // I.

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
            matrix_[target_row][j] = matrix_[target_row][j] +  matrix_[source_row][j] * scalar;
    }

    void print() const {

        for (int i = 0; i < rows_; i++) {

            for (int j = 0; j < cols_; j++) 
                std::cout << std::setw(8) << std::fixed << 
                std::setprecision(3) << std::defaultfloat << matrix_[i][j] << ' ';
            
            std::cout << "\n\n";
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

    void generate_upper_triangle(
        std::conditional_t<std::is_same_v<T, int>, int, T> min_val = {},
        std::conditional_t<std::is_same_v<T, int>, int, T> max_val = {}
    ) {
        for (int i = 0; i < min_dim_; ++i)
            for (int j = i + 1; j < cols_; ++j)
               (*this)(i, j) = generate_random(min_val, max_val); 
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

        if (row < 0 || row >= rows_ || col < 0 || col >= cols_ ||
            row + size > rows_ || col + size > cols_ || size <= 0) {
            return std::nullopt;
        }

        T determinant = T{1};
        Matrix matrix_cpy = *this;
        int sign = 1;

        for (int j = 0; j < size; ++j) {

            int current_col = col + j;
            int current_row = row + j;

            std::optional<int> max_index_opt = matrix_cpy.find_max_in_subcol(current_row, current_col);
            if (!max_index_opt) {
                return T{};
            }
            
            int max_index = *max_index_opt;

            if (max_index != current_row) {
                matrix_cpy.swap_rows(max_index, current_row);
                sign = -sign;
            }

            if (matrix_cpy(current_row, current_col) == T{}) {
                return T{};
            }

            determinant = determinant * matrix_cpy(current_row, current_col);

            for (int i = current_row + 1; i < row + size; ++i) {
                T scalar = matrix_cpy(i, current_col) / matrix_cpy(current_row, current_col);
                
                for (int k = current_col; k < col + size; ++k) {
                    matrix_cpy(i, k) = matrix_cpy(i, k) - scalar * matrix_cpy(current_row, k);
                }
            }
        }
        
        determinant_.emplace(determinant * T(sign));

        return determinant_;
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

    static double generate_random_double_(double min = 0.0, double max = 1.0) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(min, max);

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

                actual_min = 0.0;
                actual_max = 1.0;
            }

            return generate_random_double_(actual_min, actual_max);
        }

        else if constexpr (std::is_constructible_v<T, int>) {

            int actual_min = static_cast<int>(min_val);
            int actual_max = static_cast<int>(max_val);
            if (actual_min == 0 && actual_max == 0) {

                actual_min = 0;
                actual_max = 1;
            }

            return T{generate_random_int_(actual_min, actual_max)};
        }

        else if constexpr (std::is_constructible_v<T, double>) {

            double actual_min = static_cast<double>(min_val);
            double actual_max = static_cast<double>(max_val);
            if (actual_min == 0.0 && actual_max == 0.0) {

                actual_min = 0.0;
                actual_max = 1.0;
            }

            return T{generate_random_double_(actual_min, actual_max)};
        }

        else {

            static_assert(std::is_constructible_v<T, int> || std::is_constructible_v<T, double>,
                          "T must be constructible from int or double");
        }
    }

    std::optional<T>& get_determinant_() { return determinant_; }
};
