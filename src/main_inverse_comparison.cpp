#include <iostream>
#include <chrono>
#include <iomanip>
#include <complex>

#include "Matrix.hpp"

template<typename T>
void test_simple_matrix(const std::string& type_name, int size) {
    std::cout << "\n=== Testing " << type_name << " matrix " << size << "x" << size << " ===" << std::endl;
    
    try {
        Matrix<T> A = Matrix<T>::Generate_matrix(size, size);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if constexpr (std::is_same_v<T, int>) {
            auto A_inv = A.template inverse<double>();
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            
            std::cout << "Inverse computed in: " << std::fixed << std::setprecision(6) 
                      << elapsed << " seconds" << std::endl;
            
            if (size <= 3) {
                std::cout << "\nOriginal matrix:" << std::endl;
                A.print();
                
                std::cout << "\nInverse matrix:" << std::endl;
                A_inv.print();
                
                std::cout << "\nVerification (A * A⁻¹ ≈ I):" << std::endl;
                Matrix<double> A_double(size, size);
                for (int i = 0; i < size; ++i)
                    for (int j = 0; j < size; ++j)
                        A_double(i, j) = static_cast<double>(A(i, j));
                        
                auto identity_check = A_double * A_inv;
                identity_check.print();
            }
        } else {
            auto A_inv = A.inverse();
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            
            std::cout << "Inverse computed in: " << std::fixed << std::setprecision(6) 
                      << elapsed << " seconds" << std::endl;
            
            if (size <= 3) {
                std::cout << "\nOriginal matrix:" << std::endl;
                A.print();
                
                std::cout << "\nInverse matrix:" << std::endl;
                A_inv.print();
                
                std::cout << "\nVerification (A * A⁻¹ ≈ I):" << std::endl;
                auto identity_check = A * A_inv;
                identity_check.print();
            }
        }
        
        std::cout << "✓ Success" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
    }
}

void test_block_matrix(int block_size, int n_blocks) {
    std::cout << "\n=== Testing Matrix<Matrix<double>> ===" << std::endl;
    std::cout << "Blocks: " << n_blocks << "x" << n_blocks << " of size " << block_size << "x" << block_size << std::endl;
    
    try {
        // Тест 1: Создание блочной матрицы
        std::cout << "1. Creating block matrix..." << std::endl;
        Matrix<Matrix<double>> A(n_blocks, n_blocks);
        std::cout << "   Created successfully" << std::endl;
        
        // Тест 2: Инициализация блоков
        std::cout << "2. Initializing blocks..." << std::endl;
        for (int i = 0; i < n_blocks; ++i) {
            for (int j = 0; j < n_blocks; ++j) {
                if (i == j) {
                    A(i, j) = Matrix<double>::Identity(block_size, block_size);
                } else {
                    A(i, j) = Matrix<double>::Zero(block_size, block_size);
                }
            }
        }
        std::cout << "   Initialized successfully" << std::endl;
        
        // Тест 3: Вычисление обратной матрицы
        std::cout << "3. Computing inverse..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto A_inv = A.inverse();
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "   Inverse computed in: " << elapsed << " seconds" << std::endl;
        
        std::cout << "✓ All tests passed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
        std::cout << "Stack trace (if available):" << std::endl;
    }
}

int main() {
    std::cout << "Matrix Inverse Test" << std::endl;
    std::cout << "===================" << std::endl;
    
    int choice;
    std::cout << "\nChoose test type:" << std::endl;
    std::cout << "1. Simple matrix (int)" << std::endl;
    std::cout << "2. Simple matrix (double)" << std::endl;
    std::cout << "3. Simple matrix (complex)" << std::endl;
    std::cout << "4. Block matrix (Matrix<Matrix<double>>)" << std::endl;
    std::cout << "Choice: ";
    std::cin >> choice;
    
    int size;
    
    if (choice >= 1 && choice <= 3) {
        std::cout << "Enter matrix size: ";
        std::cin >> size;
        
        switch (choice) {
            case 1:
                test_simple_matrix<int>("int", size);
                break;
            case 2:
                test_simple_matrix<double>("double", size);
                break;
            case 3:
                test_simple_matrix<std::complex<double>>("complex", size);
                break;
        }
    } else if (choice == 4) {
        int block_size, n_blocks;
        std::cout << "Enter block size: ";
        std::cin >> block_size;
        std::cout << "Enter number of blocks: ";
        std::cin >> n_blocks;
        test_block_matrix(block_size, n_blocks);
    } else {
        std::cout << "Invalid choice!" << std::endl;
        return 1;
    }
    
    return 0;
}
