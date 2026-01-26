#include <iostream>
#include <complex>
#include <chrono>
#include <iomanip>
#include <type_traits>
#include <vector>
#include <numeric>
#include <random>

#include "Matrix.hpp"
#include "Vector.hpp"

template<typename T> std::string type_name() {
    if constexpr (std::is_same_v<T, int>)
        return "int";
    else if constexpr (std::is_same_v<T, float>)
        return "float";
    else if constexpr (std::is_same_v<T, double>)
        return "double";
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        return "complex<float>";
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        return "complex<double>";
    else if constexpr (detail::is_matrix_v<T>)
        return "Matrix<" + type_name<typename T::value_type>() + ">";
    else
        return "unknown";
}

template<typename T>
Matrix<T> generate_random_test_matrix(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if constexpr (detail::is_matrix_v<T>) {
        using ElemType = typename T::value_type;
        int block_size;
        std::cout << "    Enter block size (e.g., 2): ";
        std::cin >> block_size;
        
        Matrix<T> A = Matrix<T>::BlockMatrix(size, size, block_size, block_size);
        
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if constexpr (std::is_same_v<ElemType, int>) {
                    std::uniform_int_distribution<> dist(-10, 10);
                    A(i, j) = T::Generate_matrix(block_size, block_size, -5, 5);
                } else if constexpr (std::is_same_v<ElemType, float>) {
                    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
                    T block(block_size, block_size);
                    for (int bi = 0; bi < block_size; ++bi) {
                        for (int bj = 0; bj < block_size; ++bj) {
                            block(bi, bj) = dist(gen);
                        }
                    }
                    A(i, j) = block;
                } else if constexpr (std::is_same_v<ElemType, double>) {
                    std::uniform_real_distribution<double> dist(-5.0, 5.0);
                    T block(block_size, block_size);
                    for (int bi = 0; bi < block_size; ++bi) {
                        for (int bj = 0; bj < block_size; ++bj) {
                            block(bi, bj) = dist(gen);
                        }
                    }
                    A(i, j) = block;
                } else if constexpr (detail::is_complex_v<ElemType>) {
                    using RealType = typename ElemType::value_type;
                    std::uniform_real_distribution<RealType> dist(-2.0, 2.0);
                    T block(block_size, block_size);
                    for (int bi = 0; bi < block_size; ++bi) {
                        for (int bj = 0; bj < block_size; ++bj) {
                            block(bi, bj) = ElemType(dist(gen), dist(gen));
                        }
                    }
                    A(i, j) = block;
                }
            }
        }
        return A;
    } else if constexpr (std::is_same_v<T, int>) {
        std::uniform_int_distribution<> dist(-10, 10);
        Matrix<int> A(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                A(i, j) = dist(gen);
            }
            A(i, i) += 5;
        }
        return A;
    } else if constexpr (std::is_same_v<T, float>) {
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        Matrix<float> A(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                A(i, j) = dist(gen);
            }
            A(i, i) += 2.0f;
        }
        return A;
    } else if constexpr (std::is_same_v<T, double>) {
        std::uniform_real_distribution<double> dist(-5.0, 5.0);
        Matrix<double> A(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                A(i, j) = dist(gen);
            }
            A(i, i) += 2.0;
        }
        return A;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        Matrix<std::complex<float>> A(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                A(i, j) = std::complex<float>(dist(gen), dist(gen));
            }
            A(i, i) += std::complex<float>(2.0f, 0.0f);
        }
        return A;
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        std::uniform_real_distribution<double> dist(-2.0, 2.0);
        Matrix<std::complex<double>> A(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                A(i, j) = std::complex<double>(dist(gen), dist(gen));
            }
            A(i, i) += std::complex<double>(2.0, 0.0);
        }
        return A;
    } else {
        Matrix<T> A = Matrix<T>::Generate_matrix(size, size, T(-2), T(2), 100, T(1));
        for (int i = 0; i < size; ++i) {
            A(i, i) = A(i, i) + T(2);
        }
        return A;
    }
}

template<typename T>
void print_eigenvalue(const T& value, int idx) {
    std::cout << "    λ" << idx << " = ";
    
    if constexpr (detail::is_matrix_v<T>) {
        std::cout << type_name<T>() << " " << value.get_rows() << "x" << value.get_cols();
        if (value.get_rows() <= 3 && value.get_cols() <= 3) {
            std::cout << ":\n";
            value.detailed_print();
        } else {
            std::cout << " matrix\n";
        }
    } else if constexpr (detail::is_complex_v<T>) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "(" << value.real() << " + " << value.imag() << "i)\n";
    } else {
        std::cout << std::fixed << std::setprecision(6) << value << "\n";
    }
}

template<typename T>
void print_eigenvector(const Vector<T>& vec, int idx, int max_display = 5) {
    std::cout << "    v" << idx << " (size " << vec.size() << "): ";
    
    if (vec.size() <= max_display) {
        std::cout << "[";
        for (int i = 0; i < vec.size(); ++i) {
            if (i > 0) std::cout << ", ";
            
            if constexpr (detail::is_matrix_v<T>) {
                if (vec[i].get_rows() == 1 && vec[i].get_cols() == 1) {
                    std::cout << vec[i](0, 0);
                } else {
                    std::cout << type_name<T>() << " " 
                             << vec[i].get_rows() << "x" << vec[i].get_cols();
                }
            } else if constexpr (detail::is_complex_v<T>) {
                std::cout << "(" << std::fixed << std::setprecision(4) 
                         << vec[i].real() << "+" << vec[i].imag() << "i)";
            } else {
                std::cout << std::fixed << std::setprecision(4) << vec[i];
            }
        }
        std::cout << "]\n";
    } else {
        std::cout << "first " << max_display << " elements: [";
        for (int i = 0; i < max_display; ++i) {
            if (i > 0) std::cout << ", ";
            
            if constexpr (detail::is_matrix_v<T>) {
                if (vec[i].get_rows() == 1 && vec[i].get_cols() == 1) {
                    std::cout << vec[i](0, 0);
                } else {
                    std::cout << type_name<T>() << " " 
                             << vec[i].get_rows() << "x" << vec[i].get_cols();
                }
            } else if constexpr (detail::is_complex_v<T>) {
                std::cout << "(" << std::fixed << std::setprecision(4) 
                         << vec[i].real() << "+" << vec[i].imag() << "i)";
            } else {
                std::cout << std::fixed << std::setprecision(4) << vec[i];
            }
        }
        std::cout << ", ...]\n";
    }
}

template<typename T>
void test_eigen_for_type(int size) {
    using ComputeType = typename Matrix<T>::template eigen_return_type<T>;
    
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "TESTING EIGENVALUES AND EIGENVECTORS\n";
    std::cout << "Type: " << type_name<T>() << ", Size: " << size << "x" << size << "\n";
    std::cout << "Computing type: " << type_name<ComputeType>() << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    try {
        std::cout << "\n1. GENERATING RANDOM MATRIX...\n";
        auto matrix_gen_start = std::chrono::high_resolution_clock::now();
        
        Matrix<T> A = generate_random_test_matrix<T>(size);
        
        auto matrix_gen_end = std::chrono::high_resolution_clock::now();
        double matrix_gen_time = std::chrono::duration<double>(matrix_gen_end - matrix_gen_start).count();
        std::cout << "   Done in " << std::fixed << std::setprecision(4) << matrix_gen_time << "s\n";
        
        std::cout << "\n2. ORIGINAL MATRIX:\n";
        if (size <= 6) {
            if constexpr (detail::is_matrix_v<T>) {
                A.detailed_print();
            } else {
                A.precise_print(6);
            }
        } else {
            std::cout << "   [Matrix " << size << "x" << size;
            if constexpr (detail::is_matrix_v<T>) {
                if (size > 0 && A(0, 0).get_rows() > 0) {
                    std::cout << " with blocks " << A(0, 0).get_rows() 
                             << "x" << A(0, 0).get_cols();
                }
            }
            std::cout << "]\n";
            std::cout << "   Displaying first 6x6 elements:\n";
            A.print(6);
        }
        
        std::cout << "\n3. COMPUTING EIGENVALUES AND EIGENVECTORS...\n";
        auto eigen_start = std::chrono::high_resolution_clock::now();
        
        auto [eigenvalues, eigenvectors] = A.eigen();
        
        auto eigen_end = std::chrono::high_resolution_clock::now();
        double eigen_time = std::chrono::duration<double>(eigen_end - eigen_start).count();
        std::cout << "   Done in " << std::fixed << std::setprecision(4) << eigen_time << "s\n";
        
        std::cout << "\n4. RESULTS:\n";
        std::cout << "   Found " << eigenvalues.size() << " eigenvalues\n";
        
        std::cout << "\n   EIGENVALUES:\n";
        for (size_t i = 0; i < eigenvalues.size(); ++i) {
            print_eigenvalue(eigenvalues[i], i);
        }
        
        std::cout << "\n   EIGENVECTORS (as columns of matrix):\n";
        if (size <= 6) {
            eigenvectors.print(6);
        } else {
            std::cout << "   Matrix " << eigenvectors.get_rows() << "x" 
                     << eigenvectors.get_cols() << "\n";
            std::cout << "   Displaying first 6 rows and columns:\n";
            eigenvectors.print(6);
        }
        
        std::cout << "\n5. VERIFICATION (A*v ≈ λ*v):\n";
        
        int checks_to_show = std::min(3, static_cast<int>(eigenvalues.size()));
        double max_error = 0.0;
        
        for (int i = 0; i < checks_to_show; ++i) {
            if (i >= eigenvectors.get_cols()) break;
            
            Vector<ComputeType> v(eigenvectors.get_rows());
            for (int j = 0; j < eigenvectors.get_rows(); ++j) {
                v[j] = eigenvectors(j, i);
            }
            
            Matrix<ComputeType> A_computed = A.template cast_to<ComputeType>();
            Vector<ComputeType> Av = A_computed * v;
            Vector<ComputeType> lambda_v = v * eigenvalues[i];
            
            Vector<ComputeType> diff = Av - lambda_v;
            auto diff_norm = diff.norm();
            double error = 0.0;

            if constexpr (detail::is_matrix_v<decltype(diff_norm)>) {
                auto frob_norm = diff_norm.frobenius_norm();
                if constexpr (detail::is_complex_v<decltype(frob_norm)>) {
                    using std::abs;
                    error = abs(frob_norm);
                } else {
                    error = static_cast<double>(frob_norm);
                }
            } else if constexpr (detail::is_complex_v<decltype(diff_norm)>) {
                using std::abs;
                error = abs(diff_norm);
            } else {
                error = static_cast<double>(diff_norm);
            }
            
            if (error > max_error) max_error = error;
            
            std::cout << "   λ" << i << ": error = " << std::scientific 
                     << std::setprecision(2) << error;
            
            if constexpr (detail::is_matrix_v<ComputeType>) {
                std::cout << " (Frobenius norm)";
            }
            std::cout << "\n";
        }
        
        if (eigenvalues.size() > checks_to_show) {
            std::cout << "   ... and " << (eigenvalues.size() - checks_to_show) 
                     << " more eigenvalues\n";
        }
        
        std::cout << "\n   Max error: " << std::scientific << std::setprecision(2) 
                 << max_error << "\n";
        
        if (max_error < 1e-4) {
            std::cout << "   ✓ GOOD: Small error\n";
        } else if (max_error < 1e-2) {
            std::cout << "   ~ ACCEPTABLE: Moderate error\n";
        } else {
            std::cout << "   ⚠ WARNING: Large error\n";
        }
        
        std::cout << "\n6. ADDITIONAL PROPERTIES:\n";
        
        if (A.is_symmetric()) {
            std::cout << "   ✓ Matrix is symmetric\n";
        } else {
            std::cout << "   Matrix is not symmetric\n";
        }
        
        try {
            auto trace = A.trace();
            std::cout << "   Trace(A) = ";
            if constexpr (detail::is_matrix_v<decltype(trace)>) {
                std::cout << type_name<decltype(trace)>() << " " 
                         << trace.get_rows() << "x" << trace.get_cols() << "\n";
            } else if constexpr (detail::is_complex_v<decltype(trace)>) {
                std::cout << "(" << trace.real() << " + " << trace.imag() << "i)\n";
            } else {
                std::cout << trace << "\n";
            }
        } catch (...) {
            std::cout << "   Trace computation failed\n";
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(total_end - total_start).count();
        
        std::cout << "\n═══════════════════════════════════════════════════════════════\n";
        std::cout << "TOTAL TIME: " << std::fixed << std::setprecision(3) 
                 << total_time << " seconds\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        
    } catch (const std::exception& e) {
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(total_end - total_start).count();
        
        std::cout << "\n❌ ERROR: " << e.what() << "\n";
        std::cout << "\nTime elapsed: " << std::fixed << std::setprecision(3) 
                 << total_time << " seconds\n";
        
        if (std::string(e.what()).find("square") != std::string::npos) {
            std::cout << "\nNote: Eigenvalue computation requires square matrix.\n";
        }
    } catch (...) {
        std::cout << "\n❌ UNKNOWN ERROR occurred during computation\n";
    }
}

int main() {
    std::cout << "===============================================\n";
    std::cout << "MATRIX EIGENVALUES AND EIGENVECTORS TESTER\n";
    std::cout << "===============================================\n";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    while (true) {
        int size;
        std::cout << "\nEnter matrix size (n for n×n matrix, 0 to exit): ";
        std::cin >> size;
        
        if (size <= 0) {
            std::cout << "Goodbye!\n";
            break;
        }
        
        std::cout << "\nSelect matrix element type:\n";
        std::cout << "1. int (will be computed as complex<double>)\n";
        std::cout << "2. float (will be computed as complex<double>)\n";
        std::cout << "3. double (will be computed as complex<double>)\n";
        std::cout << "4. complex<float>\n";
        std::cout << "5. complex<double>\n";
        std::cout << "6. Matrix<int> (block matrix)\n";
        std::cout << "7. Matrix<double> (block matrix)\n";
        std::cout << "8. Matrix<complex<double>> (block matrix)\n";
        std::cout << "Choice [1-8]: ";
        
        int choice;
        std::cin >> choice;
        
        switch (choice) {
            case 1:
                test_eigen_for_type<int>(size);
                break;
            case 2:
                test_eigen_for_type<float>(size);
                break;
            case 3:
                test_eigen_for_type<double>(size);
                break;
            case 4:
                test_eigen_for_type<std::complex<float>>(size);
                break;
            case 5:
                test_eigen_for_type<std::complex<double>>(size);
                break;
            case 6:
                test_eigen_for_type<Matrix<int>>(size);
                break;
            case 7:
                test_eigen_for_type<Matrix<double>>(size);
                break;
            case 8:
                test_eigen_for_type<Matrix<std::complex<double>>>(size);
                break;
            default:
                std::cout << "Invalid choice!\n";
                break;
        }
        
        std::cout << "\nPress Enter to continue...";
        std::cin.ignore();
        std::cin.get();
    }
    
    return 0;
}
