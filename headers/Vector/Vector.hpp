#pragma once

template<typename T> class Matrix;
template<typename T> class Vector;

#ifndef MATRIX_IMPLEMENTATION_INCLUDED
#define MATRIX_IMPLEMENTATION_INCLUDED
#include "Matrix.hpp"
#endif

template<typename T> class Vector : public Matrix<T> {
public:
    /*============================ Vector_constructors.ipp ============================*/
    Vector();
    explicit Vector(int size);
    Vector(int size, T initial_value);
    Vector(const std::vector<T> &data);
    explicit Vector(const Matrix<T> &matrix);

    static Vector from_row(const Matrix<T> &matrix);
    static Vector<T> from_column(const Matrix<T> &matrix, int col);

    /*============================= Vector_operators.ipp =============================*/
    T &operator()(int i);
    const T &operator()(int i) const;
    T &operator[](int i);
    const T &operator[](int i) const;

    Vector<T> &operator+=(const Vector<T> &other);
    Vector<T> &operator-=(const Vector<T> &other);
    Vector<T> &operator*=(T scalar);
    Vector<T> &operator/=(T scalar);

    template<typename U> Vector<U> cast_to() const;
    template<typename U> operator Vector<U>() const;

    template<typename U>
    friend std::ostream &operator<<(std::ostream &os, const Vector<U> &vec);

    /*============================== Vector_helpers.ipp ==============================*/
    int size() const;

    static Vector<T> zero(int size);
    static Vector<T> ones(int size);
    static Vector<T> basis(int size, int k);
    static Vector<T> random(int size, T min_val, T max_val);
    static Vector<T> random(int size);
    static Vector<T> random_unit(int size);

    void print() const;

    T *begin();
    T *end();
    const T *begin() const;
    const T *end() const;
    const T *cbegin() const;
    const T *cend() const;

    /*============================== Vector_geometry.ipp ==============================*/
    T dot(const Vector<T> &other) const;
    Vector<T> cross(const Vector<T> &other) const;
    using norm_return_type = typename detail::norm_return_type_impl<T>::type;

    norm_return_type norm() const;
    T norm_squared() const;
    Vector<T> normalized() const;
    void normalize();
    T angle(const Vector<T> &other) const;
    Vector<T> projection(const Vector<T> &other) const;
    Vector<T> orthogonal(const Vector<T> &other) const;
    bool is_orthogonal(const Vector<T> &other, T tolerance = T{}) const;
    bool is_collinear(const Vector<T> &other, T tolerance = T{}) const;

private:
    /*============================== Vector_geometry.ipp ==============================*/
    T avx_dot_impl(const Vector<T> &other) const;
#if defined(__AVX__) || defined(__AVX2__)
    float avx_dot_impl_float(const Vector<float> &other) const;
    double avx_dot_impl_double(const Vector<double> &other) const;
#endif

protected:
    using Matrix<T>::identity_element;
    using Matrix<T>::zero_element;
    using Matrix<T>::generate_random;
    using Matrix<T>::is_equal;
    using Matrix<T>::is_zero;
}; // class Vector

/*============================= Vector_operators.ipp =============================*/
template<typename T, typename U>
auto operator+(const Vector<T> &lhs, const Vector<U> &rhs);

template<typename T, typename U>
auto operator-(const Vector<T> &lhs, const Vector<U> &rhs);

template<typename T, typename U> auto operator*(const Vector<T> &vec, const U &scalar);

template<typename T, typename U> auto operator*(const U &scalar, const Vector<T> &vec);

template<typename T, typename U> auto operator/(const Vector<T> &vec, const U &scalar);

template<typename T, typename U>
auto operator*(const Matrix<T> &matrix, const Vector<U> &vec);

// template<typename T, typename U>
// auto operator*(const Vector<T> &vec, const Matrix<U> &matrix);

#include "Vector_constructors.ipp"
#include "Vector_operators.ipp"
#include "Vector_geometry.ipp"
#include "Vector_helpers.ipp"
