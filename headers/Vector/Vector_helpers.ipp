template<typename T> int Vector<T>::size() const {
    return this->get_rows();
}

template<typename T> Vector<T> Vector<T>::zero(int size) {
    return Vector<T>(size, Matrix<T>::zero_element(1, 1));
}

template<typename T> Vector<T> Vector<T>::ones(int size) {
    return Vector<T>(size, Matrix<T>::identity_element(1, 1));
}

template<typename T> Vector<T> Vector<T>::basis(int size, int k) {
    Vector<T> result(size, Matrix<T>::zero_element(1, 1));
    if (k >= 0 && k < size) {
        result[k] = Matrix<T>::identity_element(1, 1);
    }
    return result;
}

template<typename T> Vector<T> Vector<T>::random(int size, T min_val, T max_val) {
    Vector<T> result(size);
    for (int i = 0; i < size; ++i) {
        result[i] = generate_random(min_val, max_val);
    }
    return result;
}

template<typename T> Vector<T> Vector<T>::random(int size) {
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
    Vector<T> result = random(size);
    try {
        result.normalize();
    } catch (const std::exception &e) {
        result = random(size);
        for (int attempt = 0; attempt < 10; ++attempt) {
            try {
                result.normalize();
                break;
            } catch (...) {
                result = random(size);
            }
        }
    }
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
    return &operator()(0);
}

template<typename T> T *Vector<T>::end() {
    return &operator()(0) + size();
}

template<typename T> const T *Vector<T>::begin() const {
    return &operator()(0);
}

template<typename T> const T *Vector<T>::end() const {
    return &operator()(0) + size();
}

template<typename T> const T *Vector<T>::cbegin() const {
    return begin();
}

template<typename T> const T *Vector<T>::cend() const {
    return end();
}
