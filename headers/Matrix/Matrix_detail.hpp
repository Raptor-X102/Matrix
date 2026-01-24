#pragma once

#include <type_traits>
#include <complex>

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

    template<typename T> struct is_complex : std::false_type {};
    template<typename T> struct is_complex<std::complex<T>> : std::true_type {};
    template<typename T> constexpr bool is_complex_v = is_complex<T>::value;

    template<typename T> struct is_matrix : std::false_type {};
    template<typename T> struct is_matrix<Matrix<T>> : std::true_type {};
    template<typename T> constexpr bool is_matrix_v = is_matrix<T>::value;

    template<typename T>
    struct has_division<Matrix<T>> : has_division<typename Matrix<T>::value_type> {};

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
    template<typename T, typename U> struct matrix_common_type<Matrix<T>, U> {
        using type = Matrix<typename matrix_common_type<T, U>::type>;
    };
    template<typename T, typename U> struct matrix_common_type<T, Matrix<U>> {
        using type = Matrix<typename matrix_common_type<T, U>::type>;
    };

    template<typename...> struct dependent_false : std::false_type {};

    template<typename T, typename = void> struct inverse_return_type_impl {
        using type = T;
    };

    template<typename T>
    struct inverse_return_type_impl<T, std::enable_if_t<detail::is_builtin_integral_v<T>>> {
        static constexpr bool is_small = 
            std::is_same_v<T, short> || std::is_same_v<T, unsigned short>;
        using type = std::conditional_t<is_small, float, double>;
    };

    template<typename T> 
    struct inverse_return_type_impl<Matrix<T>> {
        using type = Matrix<typename inverse_return_type_impl<T>::type>;
    };

    template<typename T, typename = void> struct sqrt_return_type_impl {
        using type = T;
    };

    template<typename T>
    struct sqrt_return_type_impl<T, std::enable_if_t<detail::is_builtin_integral_v<T>>> {
        static constexpr bool is_small = 
            std::is_same_v<T, short> || std::is_same_v<T, unsigned short>;
        using type = std::conditional_t<is_small, float, double>;
    };

    template<typename T> 
    struct sqrt_return_type_impl<Matrix<T>> {
        using type = Matrix<typename sqrt_return_type_impl<T>::type>;
    };

    template<typename T, typename = void> struct has_sqrt : std::false_type {};
    template<typename T>
    struct has_sqrt<T, std::void_t<decltype(sqrt(std::declval<T>()))>> : std::true_type {};
    template<typename T> constexpr bool has_sqrt_v = has_sqrt<T>::value;

    template<typename T, typename = void> struct has_acos : std::false_type {};
    template<typename T>
    struct has_acos<T, std::void_t<decltype(acos(std::declval<T>()))>> : std::true_type {};
    template<typename T> constexpr bool has_acos_v = has_acos<T>::value;

    template<typename T>
    auto sqrt_impl(const T& x) -> decltype(std::sqrt(x)) {
        return std::sqrt(x);
    }
    
    template<typename T>
    auto sqrt_impl_for_norm(const T& x) -> T {
        if constexpr (is_builtin_integral_v<T>) {
            return static_cast<T>(std::sqrt(static_cast<double>(x)));
        } else if constexpr (is_complex_v<T>) {
            using std::sqrt;
            return sqrt(x);
        } else {
            using std::sqrt;
            return sqrt(x);
        }
    }
    
    template<typename T>
    auto sqrt_impl(const Matrix<T>& mat) -> 
        Matrix<typename Matrix<T>::template sqrt_return_type<T>> {
        return mat.sqrt();
    }

    template<typename T, typename = void> 
    struct norm_return_type_impl {
        using type = T;
    };

    // Специализация для целых чисел
    template<typename T>
    struct norm_return_type_impl<T, std::enable_if_t<is_builtin_integral_v<T>>> {
        using type = double;
    };

    // Специализация для матриц
    template<typename T>
    struct norm_return_type_impl<Matrix<T>> {
        using type = typename norm_return_type_impl<T>::type;
    };
} // namespace detail
