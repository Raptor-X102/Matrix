template<typename T> T Vector<T>::dot(const Vector<T> &other) const {
    if (size() != other.size()) {
        throw std::invalid_argument("Vectors must have same size for dot product");
    }

    if (size() == 0) {
        if constexpr (detail::is_matrix_v<T>) {
            return T(0, 0);
        } else {
            return T{0};
        }
    }

#if defined(__AVX__) || defined(__AVX2__)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        return avx_dot_impl(other);
    }
#endif

    T result;
    if constexpr (detail::is_matrix_v<T>) {
        int block_rows = (*this)(0).get_rows();
        int block_cols = (*this)(0).get_cols();
        result = T::Zero(block_rows, block_cols);
    } else {
        result = T{0};
    }

    if constexpr (detail::is_complex_v<T>) {
        for (int i = 0; i < size(); ++i) {
            result += std::conj((*this)(i)) * other(i);
        }
    } else {
        for (int i = 0; i < size(); ++i) {
            result += (*this)(i)*other(i);
        }
    }
    return result;
}

template<typename T> Vector<T> Vector<T>::cross(const Vector<T> &other) const {
    if (size() != 3 || other.size() != 3) {
        throw std::invalid_argument("Cross product is only defined for 3D vectors");
    }

    Vector<T> result(3);
    result[0] = (*this)[1] * other[2] - (*this)[2] * other[1];
    result[1] = (*this)[2] * other[0] - (*this)[0] * other[2];
    result[2] = (*this)[0] * other[1] - (*this)[1] * other[0];
    return result;
}

template<typename T> typename Vector<T>::norm_return_type Vector<T>::norm() const {
    if constexpr (detail::is_matrix_v<T>) {
        if (size() == 0) {
            return norm_return_type{0};
        }

        using InnerType = typename T::value_type;
        using InnerNormType = typename Matrix<InnerType>::norm_return_type;

        InnerNormType sum = InnerNormType(0);

        for (int i = 0; i < size(); ++i) {
            auto block_norm_sq = (*this)(i).frobenius_norm_squared();
            sum = sum + block_norm_sq;
        }

        using std::sqrt;

        if constexpr (std::is_same_v<InnerNormType, int>) {
            return static_cast<int>(std::sqrt(static_cast<double>(sum)));
        } else if constexpr (std::is_floating_point_v<InnerNormType>) {
            return sqrt(sum);
        } else if constexpr (detail::is_complex_v<InnerNormType>) {
            return sqrt(sum);
        } else {
            return sqrt(sum);
        }
    } else {
        T dot_val = this->dot(*this);

        if constexpr (detail::is_complex_v<T>) {
            using std::sqrt;
            return sqrt(dot_val);
        } else if constexpr (std::is_floating_point_v<T>) {
            using std::sqrt;
            return sqrt(dot_val);
        } else if constexpr (std::is_same_v<T, int>) {
            return static_cast<int>(std::sqrt(static_cast<double>(dot_val)));
        } else if constexpr (detail::has_sqrt_v<T>) {
            return sqrt(dot_val);
        } else {
            return static_cast<norm_return_type>(dot_val);
        }
    }
}

template<typename T> T Vector<T>::norm_squared() const {
    return this->dot(*this);
}

template<typename T> T Vector<T>::angle(const Vector<T> &other) const {
    if constexpr (detail::is_complex_v<T>) {
        T dot_val = this->dot(other);
        T norm1 = this->norm();
        T norm2 = other.norm();

        if (norm1 == T{} || norm2 == T{}) {
            throw std::runtime_error("Cannot compute angle with zero vector");
        }

        using RealType = typename T::value_type;
        RealType abs_dot_val = std::abs(dot_val);
        RealType norms_product = std::abs(norm1) * std::abs(norm2);

        if (norms_product == RealType{}) {
            throw std::runtime_error("Cannot compute angle with zero norm");
        }

        RealType cos_angle = abs_dot_val / norms_product;
        cos_angle = std::max(RealType{-1}, std::min(RealType{1}, cos_angle));
        return T(std::acos(cos_angle), RealType(0));
    } else if constexpr (std::is_floating_point_v<T>) {
        T cos_angle = this->dot(other) / (this->norm() * other.norm());
        cos_angle = std::max(T{-1}, std::min(T{1}, cos_angle));
        return std::acos(cos_angle);
    } else {
        throw std::runtime_error(
            "Angle computation requires floating point or complex type");
    }
}

template<typename T>
bool Vector<T>::is_orthogonal(const Vector<T> &other, T tolerance) const {
    if constexpr (detail::is_complex_v<T>) {
        T dot_val = this->dot(other);
        using RealType = typename T::value_type;
        return std::abs(dot_val) < std::abs(tolerance);
    } else if constexpr (std::is_arithmetic_v<T>) {
        return std::abs(this->dot(other)) < std::abs(tolerance);
    } else {
        T zero_val = zero_element(1, 1);
        return this->is_equal(this->dot(other), zero_val);
    }
}

template<typename T>
bool Vector<T>::is_collinear(const Vector<T> &other, T tolerance) const {
    if constexpr (std::is_arithmetic_v<T>) {
        T zero_val = zero_element(1, 1);

        if (std::abs(this->norm()) < std::abs(tolerance)
            || std::abs(other.norm()) < std::abs(tolerance)) {
            return true;
        }

        if (size() == 3 && other.size() == 3) {
            return std::abs(this->cross(other).norm()) < std::abs(tolerance);
        }

        T cos_angle = std::abs(this->dot(other)) / (this->norm() * other.norm());
        return std::abs(cos_angle - identity_element(1, 1)) < tolerance;
    } else if constexpr (std::is_same_v<T, std::complex<float>>
                         || std::is_same_v<T, std::complex<double>>) {
        using RealType = typename T::value_type;

        RealType norm1 = std::abs(this->norm());
        RealType norm2 = std::abs(other.norm());

        if (norm1 < std::abs(tolerance) || norm2 < std::abs(tolerance)) {
            return true;
        }

        if (size() == 3 && other.size() == 3) {
            RealType cross_norm = std::abs(this->cross(other).norm());
            return cross_norm < std::abs(tolerance);
        }

        T dot_val = this->dot(other);
        RealType abs_dot = std::abs(dot_val);
        RealType cos_angle = abs_dot / (norm1 * norm2);
        return std::abs(cos_angle - RealType{1}) < std::abs(tolerance);
    } else {
        T zero_val = zero_element(1, 1);

        if (this->is_equal(this->norm(), zero_val)
            || this->is_equal(other.norm(), zero_val)) {
            return true;
        }

        if (size() == 3 && other.size() == 3) {
            return this->is_equal(this->cross(other).norm(), zero_val);
        }

        T cos_angle = this->dot(other) / (this->norm() * other.norm());
        return this->is_equal(cos_angle, identity_element(1, 1));
    }
}

template<typename T> Vector<T> Vector<T>::normalized() const {
    T n = norm();
    if (this->is_zero(n)) {
        throw std::runtime_error("Cannot normalize zero vector");
    }
    return (*this) / n;
}

template<typename T> void Vector<T>::normalize() {
    T n = norm();
    if (this->is_zero(n)) {
        throw std::runtime_error("Cannot normalize zero vector");
    }
    (*this) /= n;
}

template<typename T> Vector<T> Vector<T>::projection(const Vector<T> &other) const {
    T scalar = this->dot(other) / other.norm_squared();
    return other * scalar;
}

template<typename T> Vector<T> Vector<T>::orthogonal(const Vector<T> &other) const {
    return (*this) - this->projection(other);
}

#if defined(__AVX__) || defined(__AVX2__)
template<> float Vector<float>::avx_dot_impl_float(const Vector<float> &other) const {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < size(); i += 8) {
        __m256 a = _mm256_loadu_ps(&(*this)(i));
        __m256 b = _mm256_loadu_ps(&other(i));
        sum = _mm256_fmadd_ps(a, b, sum);
    }
    alignas(32) float temp[8];
    _mm256_store_ps(temp, sum);
    float result =
        temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    for (; i < size(); ++i) {
        result += (*this)(i)*other(i);
    }
    return result;
}

template<>
double Vector<double>::avx_dot_impl_double(const Vector<double> &other) const {
    __m256d sum = _mm256_setzero_pd();
    int i = 0;
    for (; i + 3 < size(); i += 4) {
        __m256d a = _mm256_loadu_pd(&(*this)(i));
        __m256d b = _mm256_loadu_pd(&other(i));
        sum = _mm256_fmadd_pd(a, b, sum);
    }
    alignas(32) double temp[4];
    _mm256_store_pd(temp, sum);
    double result = temp[0] + temp[1] + temp[2] + temp[3];
    for (; i < size(); ++i) {
        result += (*this)(i)*other(i);
    }
    return result;
}
#endif

template<typename T> T Vector<T>::avx_dot_impl(const Vector<T> &other) const {
#if defined(__AVX__) || defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
        return avx_dot_impl_float(other);
    } else if constexpr (std::is_same_v<T, double>) {
        return avx_dot_impl_double(other);
    } else
#endif
    {
        T result = this->zero_element(1, 1);
        for (int i = 0; i < size(); ++i) {
            result += (*this)(i)*other(i);
        }
        return result;
    }
}
