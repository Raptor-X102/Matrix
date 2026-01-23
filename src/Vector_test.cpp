#include "Vector.hpp"
#include "Matrix.hpp"
#include <iostream>
#include <complex>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <sstream>
#include <typeinfo>
#include <limits>
#include <random>

// Конфигурация тестов
struct TestConfig {
    struct TypeConfig {
        bool enabled = true;
        int vector_size = 5;
        int block_rows = 2;    // для Matrix типов
        int block_cols = 2;    // для Matrix типов
        std::string inner_type = "double"; // для Matrix типов: "int", "double", "complex"
    };
    
    TypeConfig int_config;
    TypeConfig double_config;
    TypeConfig complex_config;
    TypeConfig matrix_config;
    
    bool test_mixed = true;
    bool test_avx = true;
    bool test_edge = true;
    bool test_interactive = true;
};

// Вспомогательные функции
template<typename T> std::string type_name() { return typeid(T).name(); }
template<> std::string type_name<int>() { return "int"; }
template<> std::string type_name<float>() { return "float"; }
template<> std::string type_name<double>() { return "double"; }
template<> std::string type_name<std::complex<double>>() { return "complex<double>"; }

// Прототипы функций
template<typename T> void test_constructors(int size);
template<typename T> void test_access_operators(int size);
template<typename T> void test_arithmetic_operators(int size);
template<typename T> void test_vector_operations(int size);
template<typename T> void test_static_methods(int size);
template<typename T> void test_iterators(int size);
template<typename T> void test_block_vector(int size, int block_rows = 2, int block_cols = 2);

template<typename T1, typename T2> 
void test_mixed_types_with_T(int size1, int size2);

void test_mixed_types(const TestConfig& config);
void test_avx_optimization();
void test_edge_cases(const TestConfig& config);
void interactive_test();

// ========== РЕАЛИЗАЦИИ ТЕСТОВЫХ ФУНКЦИЙ ==========

template<typename T>
void test_constructors(int size) {
    std::cout << "=== Testing Vector Constructors (size=" << size << ") ===" << std::endl;
    
    // 1. Default constructor
    Vector<T> v1;
    std::cout << "1. Default constructor: size = " << v1.size() << std::endl;
    
    // 2. Constructor with size
    Vector<T> v2(size);
    std::cout << "2. Constructor with size(" << size << "): size = " << v2.size() << std::endl;
    
    // 3. Constructor with size and initial value
    Vector<T> v3(size, static_cast<T>(7));
    std::cout << "3. Constructor with size(" << size << ") and value(7): ";
    if (size <= 10) v3.print();
    else std::cout << "[vector too large to display]" << std::endl;
    
    // 4. Constructor from std::vector
    std::vector<T> data(size);
    for (int i = 0; i < size; ++i) data[i] = static_cast<T>(i + 1);
    Vector<T> v4(data);
    std::cout << "4. Constructor from std::vector: ";
    if (size <= 10) v4.print();
    else std::cout << "[vector too large to display]" << std::endl;
    
    // 5. Constructor from Matrix
    Matrix<T> m(size, 1);
    for (int i = 0; i < size; ++i) m(i, 0) = static_cast<T>(i + 1);
    Vector<T> v5(m);
    std::cout << "5. Constructor from Matrix: ";
    if (size <= 10) v5.print();
    else std::cout << "[vector too large to display]" << std::endl;
    
    std::cout << "===================================\n" << std::endl;
}

template<typename T>
void test_access_operators(int size) {
    std::cout << "=== Testing Access Operators (size=" << size << ") ===" << std::endl;
    
    Vector<T> v(size);
    for (int i = 0; i < size; ++i) {
        v(i) = static_cast<T>(i + 1);
    }
    
    std::cout << "Original vector: ";
    if (size <= 10) v.print();
    else std::cout << "size = " << size << std::endl;
    
    // Test operator()
    std::cout << "1. Using operator(): ";
    for (int i = 0; i < std::min(size, 5); ++i) {
        std::cout << v(i) << " ";
    }
    if (size > 5) std::cout << "...";
    std::cout << std::endl;
    
    // Test operator[]
    std::cout << "2. Using operator[]: ";
    for (int i = 0; i < std::min(size, 5); ++i) {
        std::cout << v[i] << " ";
    }
    if (size > 5) std::cout << "...";
    std::cout << std::endl;
    
    // Test bounds
    if (size > 0) {
        try {
            T val = v(size); // out of bounds
            std::cout << "3. Bounds check FAILED - should have thrown!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "3. Bounds check passed: " << e.what() << std::endl;
        }
    }
    
    std::cout << "===================================\n" << std::endl;
}

template<typename T>
void test_arithmetic_operators(int size) {
    std::cout << "=== Testing Arithmetic Operators (size=" << size << ") ===" << std::endl;
    
    Vector<T> v1(size);
    Vector<T> v2(size);
    
    for (int i = 0; i < size; ++i) {
        v1[i] = static_cast<T>(i + 1);
        v2[i] = static_cast<T>(i + 2);
    }
    
    std::cout << "v1 = ";
    if (size <= 5) v1.print();
    else std::cout << "[size=" << size << "]" << std::endl;
    
    std::cout << "v2 = ";
    if (size <= 5) v2.print();
    else std::cout << "[size=" << size << "]" << std::endl;
    
    // 1. Addition
    {
        auto sum = v1 + v2;
        std::cout << "1. v1 + v2 = ";
        if (size <= 5) sum.print();
        else std::cout << "[calculated]" << std::endl;
    }
    
    // 2. Subtraction
    {
        auto diff = v1 - v2;
        std::cout << "2. v1 - v2 = ";
        if (size <= 5) diff.print();
        else std::cout << "[calculated]" << std::endl;
    }
    
    // 3. Scalar multiplication
    {
        T scalar = static_cast<T>(2);
        auto scaled = v1 * scalar;
        std::cout << "3. v1 * " << scalar << " = ";
        if (size <= 5) scaled.print();
        else std::cout << "[calculated]" << std::endl;
    }
    
    // 4. Compound operators
    {
        Vector<T> v = v1;
        v += v2;
        std::cout << "4. v1 += v2 = ";
        if (size <= 5) v.print();
        else std::cout << "[calculated]" << std::endl;
        
        v = v1;
        v -= v2;
        std::cout << "   v1 -= v2 = ";
        if (size <= 5) v.print();
        else std::cout << "[calculated]" << std::endl;
    }
    
    std::cout << "===================================\n" << std::endl;
}

template<typename T>
void test_vector_operations(int size) {
    std::cout << "=== Testing Vector Operations (size=" << size << ") ===" << std::endl;
    
    if (size < 2) {
        std::cout << "Skipping - need size >= 2 for vector operations" << std::endl;
        std::cout << "===================================\n" << std::endl;
        return;
    }
    
    Vector<T> v1(size);
    Vector<T> v2(size);
    
    for (int i = 0; i < size; ++i) {
        v1[i] = static_cast<T>(i + 1);
        v2[i] = static_cast<T>(i + 2);
    }
    
    std::cout << "v1 = ";
    if (size <= 5) v1.print();
    else std::cout << "[size=" << size << "]" << std::endl;
    
    std::cout << "v2 = ";
    if (size <= 5) v2.print();
    else std::cout << "[size=" << size << "]" << std::endl;
    
    // 1. Dot product
    try {
        T dot_product = v1.dot(v2);
        std::cout << "1. Dot product v1 · v2 = " << dot_product << std::endl;
    } catch (const std::exception& e) {
        std::cout << "1. Dot product error: " << e.what() << std::endl;
    }
    
    // 2. Norm
    try {
        T norm_v1 = v1.norm();
        T norm_sq_v1 = v1.norm_squared();
        std::cout << "2. Norm of v1 = " << norm_v1 << std::endl;
        std::cout << "   Norm squared of v1 = " << norm_sq_v1 << std::endl;
    } catch (const std::exception& e) {
        std::cout << "2. Norm computation error: " << e.what() << std::endl;
    }
    
    // 3. Normalization
    if (size >= 2) {
        try {
            Vector<T> normalized = v1.normalized();
            std::cout << "3. Normalized v1 = ";
            if (size <= 5) normalized.print();
            else std::cout << "[calculated]" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "3. Normalization error: " << e.what() << std::endl;
        }
    }
    
    std::cout << "===================================\n" << std::endl;
}

template<typename T>
void test_static_methods(int size) {
    std::cout << "=== Testing Static Methods (size=" << size << ") ===" << std::endl;
    
    // 1. Zero vector
    {
        Vector<T> zero_vec = Vector<T>::zero(size);
        std::cout << "1. Zero vector of size " << size << ": ";
        if (size <= 5) {
            std::cout << "[";
            for (int i = 0; i < std::min(size, 5); ++i) {
                std::cout << zero_vec[i];
                if (i < std::min(size, 5) - 1) std::cout << ", ";
            }
            if (size > 5) std::cout << ", ...";
            std::cout << "]" << std::endl;
        } else {
            std::cout << "[created]" << std::endl;
        }
    }
    
    // 2. Ones vector
    {
        Vector<T> ones_vec = Vector<T>::ones(size);
        std::cout << "2. Ones vector of size " << size << ": ";
        if (size <= 5) {
            std::cout << "[";
            for (int i = 0; i < std::min(size, 5); ++i) {
                std::cout << ones_vec[i];
                if (i < std::min(size, 5) - 1) std::cout << ", ";
            }
            if (size > 5) std::cout << ", ...";
            std::cout << "]" << std::endl;
        } else {
            std::cout << "[created]" << std::endl;
        }
    }
    
    // 3. Basis vector
    if (size > 0) {
        int k = std::min(size - 1, 2);
        Vector<T> basis_vec = Vector<T>::basis(size, k);
        std::cout << "3. Basis vector of size " << size << " (k=" << k << "): ";
        if (size <= 10) {
            std::cout << "[";
            for (int i = 0; i < size; ++i) {
                std::cout << basis_vec[i];
                if (i < size - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "[created]" << std::endl;
        }
    }
    
    // 4. Random vector
    if (size > 0) {
        try {
            Vector<T> random_vec = Vector<T>::random(size);
            std::cout << "4. Random vector of size " << size << ": ";
            if (size <= 5) random_vec.print();
            else std::cout << "[created]" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "4. Random vector generation: " << e.what() << std::endl;
        }
    }
    
    std::cout << "===================================\n" << std::endl;
}

template<typename T>
void test_iterators(int size) {
    std::cout << "=== Testing Iterators (size=" << size << ") ===" << std::endl;
    
    Vector<T> v(size);
    for (int i = 0; i < size; ++i) {
        v[i] = static_cast<T>(i + 1);
    }
    
    std::cout << "Vector: ";
    if (size <= 5) v.print();
    else std::cout << "[size=" << size << "]" << std::endl;
    
    // Using range-based for loop
    std::cout << "Range-based for loop (first 5 elements): ";
    int count = 0;
    for (const auto& elem : v) {
        std::cout << elem << " ";
        if (++count >= 5) break;
    }
    if (size > 5) std::cout << "...";
    std::cout << std::endl;
    
    // Using STL algorithms
    if (size > 0) {
        T sum = std::accumulate(v.begin(), v.end(), T{});
        std::cout << "Sum using std::accumulate: " << sum << std::endl;
    }
    
    std::cout << "===================================\n" << std::endl;
}

template<typename T>
void test_block_vector(int size, int block_rows, int block_cols) {
    std::cout << "=== Testing Block Vector (Vector<Matrix<" << type_name<T>() << ">>) ===" << std::endl;
    std::cout << "Vector size: " << size << ", Block size: " << block_rows << "x" << block_cols << std::endl;
    
    using BlockType = Matrix<T>;
    using BlockVector = Vector<BlockType>;
    
    // Create a vector of matrices
    BlockVector v(size);
    
    // Fill with identity matrices scaled by index
    for (int i = 0; i < v.size(); ++i) {
        v[i] = BlockType::Identity(block_rows, block_cols) * static_cast<T>(i + 1);
    }
    
    std::cout << "Block vector created successfully." << std::endl;
    
    // Test basic operations
    try {
        // Scalar multiplication
        BlockVector scaled = v * static_cast<T>(2);
        std::cout << "1. Block vector * 2: successful" << std::endl;
        
        // Addition
        BlockVector sum = v + v;
        std::cout << "2. Block vector addition: successful" << std::endl;
        
        // Subtraction
        BlockVector diff = v - v;
        std::cout << "3. Block vector subtraction: successful" << std::endl;
        
        // Zero vector
        BlockVector zero = BlockVector::zero(size);
        std::cout << "4. Block zero vector: successful" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Operation failed: " << e.what() << std::endl;
    }
    
    std::cout << "===================================\n" << std::endl;
}

// ========== MIXED TYPE TESTS ==========

template<typename T1, typename T2>
void test_mixed_types_with_T(int size1, int size2) {
    std::cout << "=== Testing Mixed Types: " << type_name<T1>() << " + " << type_name<T2>() << " ===" << std::endl;
    std::cout << "Sizes: " << size1 << " and " << size2 << std::endl;
    
    if (size1 != size2) {
        std::cout << "Skipping - sizes must match for mixed type operations" << std::endl;
        std::cout << "===================================\n" << std::endl;
        return;
    }
    
    Vector<T1> v1(size1);
    Vector<T2> v2(size2);
    
    for (int i = 0; i < size1; ++i) {
        v1[i] = static_cast<T1>(i + 1);
        if constexpr (std::is_same_v<T2, std::complex<double>>) {
            v2[i] = T2(static_cast<double>(i + 2), 0.0);
        } else {
            v2[i] = static_cast<T2>(i + 2);
        }
    }
    
    std::cout << "v1 (" << type_name<T1>() << ") = ";
    if (size1 <= 5) v1.print();
    
    std::cout << "v2 (" << type_name<T2>() << ") = ";
    if (size2 <= 5) v2.print();
    
    // Addition
    try {
        auto sum = v1 + v2;
        std::cout << "1. v1 + v2: successful" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "1. v1 + v2 failed: " << e.what() << std::endl;
    }
    
    // Subtraction
    try {
        auto diff = v1 - v2;
        std::cout << "2. v1 - v2: successful" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "2. v1 - v2 failed: " << e.what() << std::endl;
    }
    
    // Scalar multiplication - используем подходящий скаляр для типа
    try {
        if constexpr (std::is_same_v<T1, int> && std::is_same_v<T2, std::complex<double>>) {
            // Для int * complex используем int скаляр
            auto scaled = v1 * 2;
            std::cout << "3. v1 * 2 (int): successful" << std::endl;
        } else {
            // Для других комбинаций используем подходящий скаляр
            T2 scalar;
            if constexpr (std::is_same_v<T2, std::complex<double>>) {
                scalar = T2(2.0, 0.0);
            } else {
                scalar = static_cast<T2>(2);
            }
            auto scaled = v1 * scalar;
            std::cout << "3. v1 * " << scalar << ": successful" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "3. Scalar multiplication failed: " << e.what() << std::endl;
    }
    
    std::cout << "===================================\n" << std::endl;
}

void test_mixed_types(const TestConfig& config) {
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "MIXED TYPE TESTS" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    // Test combinations based on enabled types
    int common_size = 3; // Use small size for mixed type tests
    
    if (config.int_config.enabled && config.double_config.enabled) {
        test_mixed_types_with_T<int, double>(common_size, common_size);
    }
    
    if (config.double_config.enabled && config.complex_config.enabled) {
        test_mixed_types_with_T<double, std::complex<double>>(common_size, common_size);
    }
    
    if (config.int_config.enabled && config.complex_config.enabled) {
        test_mixed_types_with_T<int, std::complex<double>>(common_size, common_size);
    }
}

// ========== AVX OPTIMIZATION TESTS ==========

void test_avx_optimization() {
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "AVX OPTIMIZATION TESTS" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    // Test float vectors (should use AVX)
    {
        std::cout << "=== Testing AVX with float (size=8 for 256-bit registers) ===" << std::endl;
        Vector<float> v1(8);
        Vector<float> v2(8);
        
        for (int i = 0; i < 8; ++i) {
            v1[i] = static_cast<float>(i + 1);
            v2[i] = static_cast<float>(i + 2);
        }
        
        std::cout << "Float vectors created" << std::endl;
        
        try {
            float dot = v1.dot(v2);
            std::cout << "Dot product: " << dot << std::endl;
            
            auto sum = v1 + v2;
            std::cout << "Addition: successful" << std::endl;
            
            std::cout << "AVX float operations completed" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
        std::cout << "===================================\n" << std::endl;
    }
    
    // Test double vectors
    {
        std::cout << "=== Testing AVX with double (size=4 for 256-bit registers) ===" << std::endl;
        Vector<double> v1(4);
        Vector<double> v2(4);
        
        for (int i = 0; i < 4; ++i) {
            v1[i] = static_cast<double>(i + 1);
            v2[i] = static_cast<double>(i + 2);
        }
        
        std::cout << "Double vectors created" << std::endl;
        
        try {
            double dot = v1.dot(v2);
            std::cout << "Dot product: " << dot << std::endl;
            
            auto sum = v1 + v2;
            std::cout << "Addition: successful" << std::endl;
            
            std::cout << "AVX double operations completed" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
        std::cout << "===================================\n" << std::endl;
    }
}

// ========== EDGE CASE TESTS ==========

void test_edge_cases(const TestConfig& config) {
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "EDGE CASE TESTS" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    // Test with size 0
    {
        std::cout << "=== Testing size 0 vector ===" << std::endl;
        Vector<double> empty;
        std::cout << "1. Empty vector size: " << empty.size() << std::endl;
        std::cout << "2. Empty vector begin == end: " << (empty.begin() == empty.end()) << std::endl;
        
        try {
            Vector<double> zero = Vector<double>::zero(0);
            std::cout << "3. Zero vector of size 0: successful" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "3. Error: " << e.what() << std::endl;
        }
        std::cout << "===================================\n" << std::endl;
    }
    
    // Test with size 1
    {
        std::cout << "=== Testing size 1 vector ===" << std::endl;
        Vector<double> singleton(1, 5.0);
        std::cout << "1. Singleton vector: ";
        singleton.print();
        
        try {
            double norm = singleton.norm();
            std::cout << "2. Norm of singleton: " << norm << std::endl;
            
            Vector<double> normalized = singleton.normalized();
            std::cout << "3. Normalized singleton: ";
            normalized.print();
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
        std::cout << "===================================\n" << std::endl;
    }
    
    // Test cross product with wrong dimensions
    {
        std::cout << "=== Testing cross product edge cases ===" << std::endl;
        Vector<double> v2d(2, 1.0);
        Vector<double> w2d(2, 2.0);
        
        try {
            auto cross = v2d.cross(w2d);
            std::cout << "1. Cross product of 2D vectors: SHOULD HAVE THROWN!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "1. Cross product of 2D vectors correctly threw: " << e.what() << std::endl;
        }
        
        Vector<double> v3(3, 1.0);
        Vector<double> w4(4, 1.0);
        
        try {
            auto dot = v3.dot(w4);
            std::cout << "2. Dot product of different sizes: SHOULD HAVE THROWN!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "2. Dot product of different sizes correctly threw: " << e.what() << std::endl;
        }
        std::cout << "===================================\n" << std::endl;
    }
}

// ========== INTERACTIVE TEST ==========

void interactive_test() {
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "INTERACTIVE TEST" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    std::cout << "Choose vector type:" << std::endl;
    std::cout << "1. double" << std::endl;
    std::cout << "2. int" << std::endl;
    std::cout << "3. complex<double>" << std::endl;
    std::cout << "Choice: ";
    
    int choice;
    std::cin >> choice;
    
    std::cout << "Enter vector size: ";
    int size;
    std::cin >> size;
    
    if (size <= 0) {
        std::cout << "Invalid size!" << std::endl;
        return;
    }
    
    if (choice == 1) {
        Vector<double> v(size);
        std::cout << "Enter " << size << " double values: ";
        for (int i = 0; i < size; ++i) {
            double val;
            std::cin >> val;
            v[i] = val;
        }
        
        std::cout << "\nYour vector: ";
        v.print();
        
        std::cout << "Norm: " << v.norm() << std::endl;
        
        if (size >= 2) {
            Vector<double> w(size);
            std::cout << "\nEnter another vector of size " << size << ": ";
            for (int i = 0; i < size; ++i) {
                double val;
                std::cin >> val;
                w[i] = val;
            }
            
            std::cout << "Second vector: ";
            w.print();
            
            std::cout << "Dot product: " << v.dot(w) << std::endl;
            
            if (size == 3) {
                std::cout << "Cross product: ";
                v.cross(w).print();
            }
        }
        
    } else if (choice == 2) {
        Vector<int> v(size);
        std::cout << "Enter " << size << " int values: ";
        for (int i = 0; i < size; ++i) {
            int val;
            std::cin >> val;
            v[i] = val;
        }
        
        std::cout << "\nYour vector: ";
        v.print();
        
    } else if (choice == 3) {
        Vector<std::complex<double>> v(size);
        std::cout << "Enter " << size << " complex values (real imag): ";
        for (int i = 0; i < size; ++i) {
            double real, imag;
            std::cin >> real >> imag;
            v[i] = std::complex<double>(real, imag);
        }
        
        std::cout << "\nYour vector: ";
        v.print();
    }
}

// ========== КОНФИГУРАЦИЯ И ЗАПУСК ==========

TestConfig configure_tests() {
    TestConfig config;
    
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "VECTOR TEST CONFIGURATION" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    // Настройка для int
    std::cout << "1. Test Vector<int>:" << std::endl;
    std::cout << "   Enable? (y/n): ";
    char response;
    std::cin >> response;
    config.int_config.enabled = (response == 'y' || response == 'Y');
    
    if (config.int_config.enabled) {
        std::cout << "   Enter vector size (default 5): ";
        std::string input;
        std::cin >> input;
        if (!input.empty()) {
            config.int_config.vector_size = std::stoi(input);
            if (config.int_config.vector_size <= 0) config.int_config.vector_size = 5;
        }
    }
    
    // Настройка для double
    std::cout << "\n2. Test Vector<double>:" << std::endl;
    std::cout << "   Enable? (y/n): ";
    std::cin >> response;
    config.double_config.enabled = (response == 'y' || response == 'Y');
    
    if (config.double_config.enabled) {
        std::cout << "   Enter vector size (default 5): ";
        std::string input;
        std::cin >> input;
        if (!input.empty()) {
            config.double_config.vector_size = std::stoi(input);
            if (config.double_config.vector_size <= 0) config.double_config.vector_size = 5;
        }
    }
    
    // Настройка для complex
    std::cout << "\n3. Test Vector<complex<double>>:" << std::endl;
    std::cout << "   Enable? (y/n): ";
    std::cin >> response;
    config.complex_config.enabled = (response == 'y' || response == 'Y');
    
    if (config.complex_config.enabled) {
        std::cout << "   Enter vector size (default 5): ";
        std::string input;
        std::cin >> input;
        if (!input.empty()) {
            config.complex_config.vector_size = std::stoi(input);
            if (config.complex_config.vector_size <= 0) config.complex_config.vector_size = 5;
        }
    }
    
    // Настройка для Matrix
    std::cout << "\n4. Test Vector<Matrix<?>>:" << std::endl;
    std::cout << "   Enable? (y/n): ";
    std::cin >> response;
    config.matrix_config.enabled = (response == 'y' || response == 'Y');
    
    if (config.matrix_config.enabled) {
        std::cout << "   Enter vector size (default 3): ";
        std::string input;
        std::cin >> input;
        if (!input.empty()) {
            config.matrix_config.vector_size = std::stoi(input);
            if (config.matrix_config.vector_size <= 0) config.matrix_config.vector_size = 3;
        }
        
        std::cout << "   Enter block rows (default 2): ";
        std::cin >> input;
        if (!input.empty()) {
            config.matrix_config.block_rows = std::stoi(input);
            if (config.matrix_config.block_rows <= 0) config.matrix_config.block_rows = 2;
        }
        
        std::cout << "   Enter block cols (default 2): ";
        std::cin >> input;
        if (!input.empty()) {
            config.matrix_config.block_cols = std::stoi(input);
            if (config.matrix_config.block_cols <= 0) config.matrix_config.block_cols = 2;
        }
        
        std::cout << "   Select inner matrix type:" << std::endl;
        std::cout << "   1. int" << std::endl;
        std::cout << "   2. double" << std::endl;
        std::cout << "   3. complex<double>" << std::endl;
        std::cout << "   Choice (1-3): ";
        int choice;
        std::cin >> choice;
        
        switch (choice) {
            case 1: config.matrix_config.inner_type = "int"; break;
            case 2: config.matrix_config.inner_type = "double"; break;
            case 3: config.matrix_config.inner_type = "complex"; break;
            default: config.matrix_config.inner_type = "double";
        }
    }
    
    // Дополнительные тесты
    std::cout << "\n5. Additional tests:" << std::endl;
    std::cout << "   Test mixed type operations? (y/n): ";
    std::cin >> response;
    config.test_mixed = (response == 'y' || response == 'Y');
    
    std::cout << "   Test AVX optimization? (y/n): ";
    std::cin >> response;
    config.test_avx = (response == 'y' || response == 'Y');
    
    std::cout << "   Test edge cases? (y/n): ";
    std::cin >> response;
    config.test_edge = (response == 'y' || response == 'Y');
    
    std::cout << "   Run interactive test? (y/n): ";
    std::cin >> response;
    config.test_interactive = (response == 'y' || response == 'Y');
    
    return config;
}

// Функция для запуска всех тестов для одного типа
template<typename T>
void run_all_tests_for_type(const std::string& type_name, 
                           const TestConfig::TypeConfig& config) {
    if (!config.enabled) return;
    
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "TESTING VECTOR<" << type_name << ">" << std::endl;
    std::cout << "Vector size: " << config.vector_size << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    test_constructors<T>(config.vector_size);
    test_access_operators<T>(config.vector_size);
    test_arithmetic_operators<T>(config.vector_size);
    test_vector_operations<T>(config.vector_size);
    test_static_methods<T>(config.vector_size);
    test_iterators<T>(config.vector_size);
}

// Функция для тестирования Matrix типов
void run_matrix_tests(const TestConfig::TypeConfig& config) {
    if (!config.enabled) return;
    
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "TESTING VECTOR<Matrix<" << config.inner_type << ">>" << std::endl;
    std::cout << "Vector size: " << config.vector_size << std::endl;
    std::cout << "Block size: " << config.block_rows << "x" << config.block_cols << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    if (config.inner_type == "int") {
        test_block_vector<int>(config.vector_size, config.block_rows, config.block_cols);
    } else if (config.inner_type == "double") {
        test_block_vector<double>(config.vector_size, config.block_rows, config.block_cols);
    } else if (config.inner_type == "complex") {
        test_block_vector<std::complex<double>>(config.vector_size, config.block_rows, config.block_cols);
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "VECTOR CLASS COMPREHENSIVE TEST SUITE" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    // Конфигурация тестов
    TestConfig config = configure_tests();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "STARTING TESTS" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    // Запуск тестов для каждого типа
    if (config.int_config.enabled) {
        run_all_tests_for_type<int>("int", config.int_config);
    }
    
    if (config.double_config.enabled) {
        run_all_tests_for_type<double>("double", config.double_config);
    }
    
    if (config.complex_config.enabled) {
        run_all_tests_for_type<std::complex<double>>("complex<double>", config.complex_config);
    }
    
    if (config.matrix_config.enabled) {
        run_matrix_tests(config.matrix_config);
    }
    
    // Дополнительные тесты
    if (config.test_mixed) {
        test_mixed_types(config);
    }
    
    if (config.test_avx) {
        test_avx_optimization();
    }
    
    if (config.test_edge) {
        test_edge_cases(config);
    }
    
    if (config.test_interactive) {
        interactive_test();
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "ALL TESTS COMPLETED" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}
