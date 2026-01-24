template<typename T, typename ValueType>
auto create_scalar(const T& example, ValueType value) {
    if constexpr (detail::is_matrix_v<T>) {
        // Для блочных матриц возвращаем value как есть
        // Пусть оператор умножения разбирается с преобразованием
        return value;
    } else {
        // Для скалярных типов используем общий тип
        using CommonType = typename detail::matrix_common_type<T, ValueType>::type;
        return static_cast<CommonType>(value);
    }
}

// Для получения вещественного типа из нормы
template<typename T>
struct real_type_from_norm {
    using type = T;
};

template<typename T>
struct real_type_from_norm<std::complex<T>> {
    using type = T;
};

template<typename T>
using real_type_from_norm_t = typename real_type_from_norm<T>::type;

// Основные функции

template<typename T>
Matrix<typename Matrix<T>::template sqrt_return_type<T>> sqrt(const Matrix<T>& m) {
    return m.sqrt();
}

template<typename T>
Matrix<typename Matrix<T>::template sqrt_return_type<T>> Matrix<T>::sqrt() const {
    using ResultType = sqrt_return_type<T>;
    
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix square root requires square matrix");
    }
    
    if constexpr (detail::is_matrix_v<T>) {
        return this->template sqrt_impl<ResultType>();
    } else {
        Matrix<ResultType> A = this->template cast_to<ResultType>();
        
        if (rows_ == 1) {
            Matrix<ResultType> result(1, 1);
            result(0, 0) = detail::sqrt_impl(A(0, 0));
            return result;
        }
        
        if (rows_ == 2) {
            return A.template sqrt_2x2_impl<ResultType>();
        }
        
        return A.template sqrt_newton_impl<ResultType>();
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::sqrt_2x2_impl() const {
    ResultType a = static_cast<ResultType>((*this)(0, 0));
    ResultType b = static_cast<ResultType>((*this)(0, 1));
    ResultType c = static_cast<ResultType>((*this)(1, 0));
    ResultType d = static_cast<ResultType>((*this)(1, 1));
    
    if (is_zero(b) && is_zero(c)) {
        Matrix<ResultType> result(2, 2);
        if constexpr (detail::is_builtin_integral_v<ResultType>) {
            result(0, 0) = static_cast<ResultType>(
                std::sqrt(static_cast<double>(a)));
            result(1, 1) = static_cast<ResultType>(
                std::sqrt(static_cast<double>(d)));
        } else if constexpr (detail::is_matrix_v<ResultType>) {
            result(0, 0) = a.sqrt();
            result(1, 1) = d.sqrt();
        } else {
            using std::sqrt;
            result(0, 0) = sqrt(a);
            result(1, 1) = sqrt(d);
        }
        return result;
    }
    
    ResultType det = a * d - b * c;
    ResultType tr = a + d;
    
    // Создаем скаляры - для блочных матриц просто числа
    auto four = create_scalar(a, 4);
    auto two = create_scalar(a, 2);
    
    ResultType delta = tr * tr - four * det;
    ResultType sqrt_delta;
    
    if constexpr (detail::is_builtin_integral_v<ResultType>) {
        sqrt_delta = static_cast<ResultType>(
            std::sqrt(static_cast<double>(delta)));
    } else if constexpr (detail::is_matrix_v<ResultType>) {
        sqrt_delta = delta.sqrt();
    } else {
        using std::sqrt;
        sqrt_delta = sqrt(delta);
    }
    
    ResultType s_plus;
    ResultType s_minus;
    
    if constexpr (detail::is_builtin_integral_v<ResultType>) {
        s_plus = static_cast<ResultType>(
            std::sqrt(static_cast<double>((tr + sqrt_delta) / two)));
        s_minus = static_cast<ResultType>(
            std::sqrt(static_cast<double>((tr - sqrt_delta) / two)));
    } else if constexpr (detail::is_matrix_v<ResultType>) {
        s_plus = ((tr + sqrt_delta) / two).sqrt();
        s_minus = ((tr - sqrt_delta) / two).sqrt();
    } else {
        using std::sqrt;
        s_plus = sqrt((tr + sqrt_delta) / two);
        s_minus = sqrt((tr - sqrt_delta) / two);
    }
    
    ResultType denominator = s_plus + s_minus;
    
    Matrix<ResultType> result(2, 2);
    
    if (!is_zero(s_plus - s_minus)) {
        result(0, 0) = (a + s_plus * s_minus) / denominator;
        result(0, 1) = b / denominator;
        result(1, 0) = c / denominator;
        result(1, 1) = (d + s_plus * s_minus) / denominator;
    } else {
        ResultType s = s_plus;
        result(0, 0) = (a + s * s) / (two * s);
        result(0, 1) = b / (two * s);
        result(1, 0) = c / (two * s);
        result(1, 1) = (d + s * s) / (two * s);
    }
    
    return result;
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::sqrt_newton_impl(int max_iter, ResultType tolerance) const {
    int n = rows_;
    Matrix<ResultType> X = this->template cast_to<ResultType>();
    Matrix<ResultType> A = X;
    
    if constexpr (detail::is_matrix_v<ResultType>) {
        int block_rows = X(0, 0).get_rows();
        int block_cols = X(0, 0).get_cols();
        X = Matrix<ResultType>::BlockIdentity(n, n, block_rows, block_cols);
    } else {
        X = Matrix<ResultType>::Identity(n);
    }
    
    // Получаем вещественный тип для сравнений
    using NormType = typename Matrix<ResultType>::norm_return_type;
    using RealType = real_type_from_norm_t<NormType>;
    
    RealType tol_norm;
    if constexpr (detail::is_matrix_v<ResultType>) {
        tol_norm = static_cast<RealType>(tolerance.frobenius_norm());
    } else if constexpr (detail::is_complex_v<ResultType>) {
        using std::abs;
        tol_norm = static_cast<RealType>(abs(tolerance));
    } else {
        tol_norm = static_cast<RealType>(tolerance);
    }
    
    Matrix<ResultType> best_X = X;
    RealType best_residual = std::numeric_limits<RealType>::max();
    
    // Инициализируем best_residual текущим остатком
    try {
        NormType norm_residual = (X * X - A).frobenius_norm();
        if constexpr (detail::is_complex_v<NormType>) {
            using std::abs;
            best_residual = abs(norm_residual);
        } else {
            best_residual = norm_residual;
        }
    } catch (...) {
        // Игнорируем ошибки при вычислении начального остатка
    }
    
    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix<ResultType> X_prev = X;
        bool iteration_failed = false;
        std::string error_msg;
        
        try {
            Matrix<ResultType> X_inv;
            
            try {
                X_inv = X.inverse();
            }
            catch (const std::exception& e) {
                // Сохраняем сообщение об ошибке
                error_msg = e.what();
                
                // Если матрица вырожденная, добавляем небольшой сдвиг
                if constexpr (detail::is_matrix_v<ResultType>) {
                    using ElemType = typename ResultType::value_type;
                    auto shift_scalar = create_scalar(ElemType{}, 1e-6);
                    ResultType shift = ResultType::Identity(
                        X(0, 0).get_rows(), 
                        X(0, 0).get_cols()) * shift_scalar;
                    
                    for (int i = 0; i < n; ++i) {
                        X(i, i) = X(i, i) + shift;
                    }
                    
                    try {
                        X_inv = X.inverse();
                    }
                    catch (const std::exception& e2) {
                        iteration_failed = true;
                        error_msg = std::string("Failed even with regularization: ") + e2.what();
                    }
                    catch (...) {
                        iteration_failed = true;
                        error_msg = "Failed even with regularization: unknown error";
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        X(i, i) = X(i, i) + static_cast<ResultType>(1e-6);
                    }
                    
                    try {
                        X_inv = X.inverse();
                    }
                    catch (const std::exception& e2) {
                        iteration_failed = true;
                        error_msg = std::string("Failed even with regularization: ") + e2.what();
                    }
                    catch (...) {
                        iteration_failed = true;
                        error_msg = "Failed even with regularization: unknown error";
                    }
                }
            }
            
            if (!iteration_failed) {
                // Создаем скаляр 0.5
                if constexpr (detail::is_matrix_v<ResultType>) {
                    using ElemType = typename ResultType::value_type;
                    auto half_scalar = create_scalar(ElemType{}, 0.5);
                    
                    // Упрощенный подход для блочных матриц
                    Matrix<ResultType> temp = X_inv * A;
                    Matrix<ResultType> X_new(n, n);
                    
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < n; ++j) {
                            X_new(i, j) = (X(i, j) + temp(i, j)) * half_scalar;
                        }
                    }
                    X = X_new;
                } else {
                    auto half = create_scalar(ResultType{}, 0.5);
                    X = (X + X_inv * A) * half;
                }
            }
        }
        catch (const std::exception& e) {
            iteration_failed = true;
            if (error_msg.empty()) {
                error_msg = e.what();
            }
        }
        catch (...) {
            iteration_failed = true;
            if (error_msg.empty()) {
                error_msg = "Unknown error in Newton iteration";
            }
        }
        
        if (iteration_failed) {
            // Вычисляем остаток для предыдущего приближения
            RealType prev_residual;
            try {
                NormType norm_residual = (X_prev * X_prev - A).frobenius_norm();
                if constexpr (detail::is_complex_v<NormType>) {
                    using std::abs;
                    prev_residual = abs(norm_residual);
                } else {
                    prev_residual = norm_residual;
                }
            } catch (...) {
                prev_residual = std::numeric_limits<RealType>::max();
            }
            
            // Если у нас есть какое-то приближение, возвращаем лучшее
            if (iter > 0 || best_residual < std::numeric_limits<RealType>::max()) {
                // Логируем ошибку, но возвращаем приближение
                std::cerr << "Warning: Newton iteration " << iter + 1 
                          << " failed: " << error_msg 
                          << ", returning best approximation with residual " 
                          << best_residual << std::endl;
                return best_X;
            } else {
                // Первая итерация не удалась
                throw std::runtime_error("Newton method failed on first iteration: " + error_msg);
            }
        }
        
        // Вычисляем норму ошибки и остаток
        RealType error = 0;
        RealType residual = 0;
        
        try {
            NormType norm_error = (X - X_prev).frobenius_norm();
            if constexpr (detail::is_complex_v<NormType>) {
                using std::abs;
                error = abs(norm_error);
            } else {
                error = norm_error;
            }
            
            NormType norm_residual = (X * X - A).frobenius_norm();
            if constexpr (detail::is_complex_v<NormType>) {
                using std::abs;
                residual = abs(norm_residual);
            } else {
                residual = norm_residual;
            }
        } catch (...) {
            // Ошибка при вычислении норм
            iteration_failed = true;
            error_msg = "Failed to compute norms";
        }
        
        if (!iteration_failed) {
            // Сохраняем лучшее приближение
            if (residual < best_residual) {
                best_X = X;
                best_residual = residual;
            }
            
            // Проверка сходимости
            if (error < tol_norm && residual < tol_norm * 10) {
                return X;
            }
            
            // Проверка на медленную сходимость или расходимость
            if (iter > 20 && residual > best_residual * 5) {
                // Метод, вероятно, расходится
                std::cerr << "Warning: Newton method appears to be diverging at iteration " 
                          << iter + 1 << ", returning best approximation" << std::endl;
                return best_X;
            }
        }
    }
    
    // Достигнут максимум итераций
    if (best_residual < tol_norm * 100) {
        std::cerr << "Warning: Newton method reached maximum iterations (" 
                  << max_iter << "), returning best approximation with residual " 
                  << best_residual << std::endl;
        return best_X;
    }
    
    throw std::runtime_error("Newton method did not converge in " + 
                            std::to_string(max_iter) + " iterations, best residual: " +
                            std::to_string(best_residual));
}

template<typename T>
bool Matrix<T>::has_square_root() const {
    if (rows_ != cols_) return false;
    
    if constexpr (detail::is_builtin_integral_v<T>) {
        using ResultType = sqrt_return_type<T>;
        Matrix<ResultType> A = this->template cast_to<ResultType>();
        return A.template has_square_root_impl<ResultType>();
    }
    
    return this->template has_square_root_impl<T>();
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::has_square_root_impl() const {
    if (rows_ == 1) {
        if constexpr (std::is_arithmetic_v<ResultType> && !detail::is_complex_v<ResultType>) {
            ResultType val = static_cast<ResultType>((*this)(0, 0));
            if constexpr (detail::is_matrix_v<ResultType>) {
                // Для блочных матриц не проверяем неотрицательность
                return true;
            } else {
                return val >= ResultType(0);
            }
        }
        return true;
    }
    
    if (rows_ == 2) {
        // Для 2x2 матриц можно проверить условие существования корня
        if constexpr (std::is_arithmetic_v<ResultType> && !detail::is_complex_v<ResultType>) {
            ResultType a = static_cast<ResultType>((*this)(0, 0));
            ResultType b = static_cast<ResultType>((*this)(0, 1));
            ResultType c = static_cast<ResultType>((*this)(1, 0));
            ResultType d = static_cast<ResultType>((*this)(1, 1));
            
            // Проверка условия для существования вещественного квадратного корня
            ResultType tr = a + d;
            ResultType det = a * d - b * c;
            
            // Дискриминант характеристического уравнения
            ResultType discr = tr * tr - 4 * det;
            if (discr < ResultType(0)) {
                return false;  // Комплексные собственные значения
            }
            
            ResultType sqrt_discr = std::sqrt(discr);
            ResultType lambda1 = (tr + sqrt_discr) / ResultType(2);
            ResultType lambda2 = (tr - sqrt_discr) / ResultType(2);
            
            // Оба собственных значения должны быть неотрицательными
            return lambda1 >= ResultType(0) && lambda2 >= ResultType(0);
        }
        // Для комплексных и блочных матриц предполагаем, что корень есть
        return true;
    }
    
    // Определяем, нужно ли делать прямую проверку
    constexpr bool should_check_directly = 
        std::is_arithmetic_v<ResultType> && 
        !detail::is_complex_v<ResultType> && 
        !detail::is_matrix_v<ResultType>;
    
    if constexpr (should_check_directly) {
        // Только для вещественных скалярных типов
        
        // Проверяем симметричность и положительную определенность
        bool symmetric = true;
        for (int i = 0; i < rows_ && symmetric; ++i) {
            for (int j = i + 1; j < cols_ && symmetric; ++j) {
                if (std::abs(static_cast<ResultType>((*this)(i, j)) - 
                             static_cast<ResultType>((*this)(j, i))) > 
                    ResultType(1e-6)) {
                    symmetric = false;
                }
            }
        }
        
        // Проверяем положительность диагональных элементов
        bool positive_diag = true;
        for (int i = 0; i < rows_; ++i) {
            if (static_cast<ResultType>((*this)(i, i)) <= ResultType(0)) {
                positive_diag = false;
                break;
            }
        }
        
        // Для симметричных положительно определенных матриц корень гарантированно существует
        if (symmetric && positive_diag) {
            return true;
        }
        
        // В остальных случаях пытаемся вычислить напрямую
        try {
            return this->template has_square_root_direct_impl<ResultType>();
        }
        catch (...) {
            return false;
        }
    } else {
        // Для комплексных чисел и блочных матриц всегда предполагаем, что корень есть
        // (хотя на самом деле могут быть вырожденные случаи)
        return true;
    }
}

template<typename T>
template<typename ResultType>
bool Matrix<T>::has_square_root_direct_impl() const {
    // Эта функция вызывается только для вещественных скалярных типов
    
    // Проверка диагональных элементов на неотрицательность
    for (int i = 0; i < rows_; ++i) {
        ResultType diag = static_cast<ResultType>((*this)(i, i));
        if (diag < ResultType(0)) {
            return false;
        }
    }
    
    // Пробуем вычислить квадратный корень
    try {
        Matrix<ResultType> test_sqrt = this->template sqrt_impl<ResultType>();
        Matrix<ResultType> check = test_sqrt * test_sqrt;
        Matrix<ResultType> A = this->template cast_to<ResultType>();
        
        // Вычисляем ошибку
        using NormType = typename decltype(check - A)::norm_return_type;
        NormType norm_error = (check - A).frobenius_norm();
        
        // Для вещественных типов нормой будет вещественное число
        return norm_error < static_cast<NormType>(1e-6);
    }
    catch (...) {
        return false;
    }
}

template<typename T>
template<typename ResultType>
Matrix<ResultType> Matrix<T>::sqrt_impl() const {
    if (rows_ == 1) {
        Matrix<ResultType> result(1, 1);
        if constexpr (detail::is_builtin_integral_v<ResultType>) {
            result(0, 0) = static_cast<ResultType>(
                std::sqrt(static_cast<double>(static_cast<ResultType>((*this)(0, 0)))));
        } else if constexpr (detail::is_matrix_v<ResultType>) {
            result(0, 0) = static_cast<ResultType>((*this)(0, 0)).sqrt();
        } else {
            using std::sqrt;
            result(0, 0) = sqrt(static_cast<ResultType>((*this)(0, 0)));
        }
        return result;
    }
    
    if (rows_ == 2) {
        return this->template sqrt_2x2_impl<ResultType>();
    }
    
    // Для больших матриц используем метод Ньютона
    if constexpr (detail::is_matrix_v<ResultType>) {
        using ElemType = typename ResultType::value_type;
        // Создаем tolerance как скалярную матрицу
        auto tolerance_scalar = create_scalar(ElemType{}, 1e-10);
        ResultType tolerance = ResultType::Identity(
            matrix_[0][0].get_rows(), 
            matrix_[0][0].get_cols()) * tolerance_scalar;
        
        return this->template sqrt_newton_impl<ResultType>(100, tolerance);
    } else {
        auto tolerance_value = create_scalar(ResultType{}, 1e-10);
        return this->template sqrt_newton_impl<ResultType>(100, tolerance_value);
    }
}
