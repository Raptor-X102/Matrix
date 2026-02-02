template<typename T> int Vector<T>::size() const {
    return this->get_rows();
}

template<typename T> Vector<T> Vector<T>::zero(int size) {
    if (size < 0) {
        throw std::invalid_argument("Size cannot be negative");
    }
    return Vector<T>(size, Matrix<T>::zero_element(1, 1));
}

template<typename T> Vector<T> Vector<T>::ones(int size) {
    if (size < 0) {
        throw std::invalid_argument("Size cannot be negative");
    }
    return Vector<T>(size, Matrix<T>::identity_element(1, 1));
}

template<typename T> Vector<T> Vector<T>::basis(int size, int k) {
    if (size < 0) {
        throw std::invalid_argument("Size cannot be negative");
    }
    if (k < 0 || k >= size) {
        throw std::invalid_argument("Basis index out of bounds");
    }
    Vector<T> result(size, Matrix<T>::zero_element(1, 1));
    result[k] = Matrix<T>::identity_element(1, 1);
    return result;
}

template<typename T> Vector<T> Vector<T>::random(int size, T min_val, T max_val) {
    if (size < 0) {
        throw std::invalid_argument("Size cannot be negative");
    }
    Vector<T> result(size);
    for (int i = 0; i < size; ++i) {
        result[i] = generate_random(min_val, max_val);
    }
    return result;
}

template<typename T> Vector<T> Vector<T>::random(int size) {
    if (size < 0) {
        throw std::invalid_argument("Size cannot be negative");
    }
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return random(size,
                      std::complex<double>(-1.0, -1.0),
                      std::complex<double>(1.0, 1.0));
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return random(size,
                      std::complex<float>(-1.0f, -1.0f),
                      std::complex<float>(1.0f, 1.0f));
    } else {
        return random(size,
                      zero_element(1, 1) - identity_element(1, 1),
                      identity_element(1, 1));
    }
}

template<typename T> Vector<T> Vector<T>::random_unit(int size) {
    if (size < 0) {
        throw std::invalid_argument("Size cannot be negative");
    }

    if (size == 0) {
        return Vector<T>(0);
    }

    for (int attempt = 0; attempt < 10; ++attempt) {
        Vector<T> result = random(size);
        auto norm_val = result.norm();
        if (!result.is_zero(norm_val)) { // Use local variable instead of 'this'
            result.normalize();
            return result;
        }
    }

    // Final attempt with fallback
    Vector<T> result = random(size);
    auto norm_val = result.norm();
    if (result.is_zero(norm_val)) {
        throw std::runtime_error(
            "Failed to generate non-zero random unit vector after multiple attempts");
    }
    result.normalize();
    return result;
}

template<typename T> void Vector<T>::print() const {
    std::cout << "Vector(" << size() << "): [";
    for (int i = 0; i < size(); ++i) {
        std::cout << (*this)(i);
        if (i < size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

template<typename T> T *Vector<T>::begin() {
    if (size() == 0) {
        return nullptr;
    }
    return &operator()(0);
}

template<typename T> T *Vector<T>::end() {
    if (size() == 0) {
        return nullptr;
    }
    return &operator()(0) + size();
}

template<typename T> const T *Vector<T>::begin() const {
    if (size() == 0) {
        return nullptr;
    }
    return &operator()(0);
}

template<typename T> const T *Vector<T>::end() const {
    if (size() == 0) {
        return nullptr;
    }
    return &operator()(0) + size();
}

template<typename T> const T *Vector<T>::cbegin() const {
    return begin();
}

template<typename T> const T *Vector<T>::cend() const {
    return end();
}
