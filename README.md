# Matrix Library Project

## Описание

Проект представляет собой реализацию шаблонного класса матриц на C++ с поддержкой различных числовых типов и блочных операций. Библиотека включает в себя полный набор линейно-алгебраических операций, оптимизации для работы с блоками и SIMD-инструкциями.

## Особенности

- **Шаблонный дизайн**: Поддержка различных числовых типов (int, float, double, complex)
- **Блочные матрицы**: Частичная поддержка блочных операций с оптимизированными алгоритмами
- **Полный набор операций**:
  - Арифметические операции (+, -, *, /)
  - Транспонирование с оптимизациями (SIMPLE, BLOCKED, SIMD_BLOCKED)
  - Вычисление определителя
  - Обращение матриц
  - Вычисление квадратного корня матрицы
  - Нахождение собственных значений и векторов
  - QR-разложение
- **Оптимизации**: AVX/SIMD инструкции, блочные алгоритмы, кэш-оптимизация
- **Автоматическое управление памятью**: Использование unique_ptr для безопасности

## Структура проекта

```
Matrix/
├── headers/
│   ├── Matrix/
│   │   ├── Matrix.hpp                    # Основной заголовочный файл
│   │   ├── Matrix_constructors.ipp       # Конструкторы
│   │   ├── Matrix_operators.ipp          # Операторы
│   │   ├── Matrix_determinant.ipp        # Определитель
│   │   ├── Matrix_inverse.ipp            # Обращение матриц
│   │   ├── Matrix_sqrt.ipp               # Квадратный корень
│   │   ├── Matrix_eigen.ipp              # Собственные значения
│   │   ├── Matrix_transpose.ipp          # Транспонирование
│   │   ├── Matrix_properties.ipp         # Свойства матриц
│   │   ├── Matrix_helpers.ipp            # Вспомогательные функции
│   │   ├── Matrix_generators.ipp         # Генерация матриц
│   │   └── Matrix_detail.hpp             # Детали реализации
│   ├── Vector/
│   │   ├── Vector.hpp                    # Класс векторов
│   │   ├── Vector_constructors.ipp       # Конструкторы векторов
│   │   ├── Vector_operators.ipp          # Операторы для векторов
│   │   ├── Vector_geometry.ipp           # Геометрические операции
│   │   └── Vector_helpers.ipp            # Вспомогательные функции
│   ├── Debug_printf.h                    # Отладочный вывод
│   └── Timer.hpp                         # Измерение времени
├── src/
│   ├── main_determinant.cpp              # Основная программа (определитель)
│   └── tests/                            # Тесты
│       ├── main.cpp                      # Тест арифметических операций
│       ├── main_double.cpp               # Тест с double
│       ├── main_int.cpp                  # Тест с int
│       ├── main_complex.cpp              # Тест с complex
│       ├── main_inverse_comparison.cpp   # Сравнение методов обращения
│       ├── main_universal_det.cpp        # Универсальный тест определителя
│       ├── main_test_multiplication.cpp  # Тест умножения
│       ├── main_transpose_timing.cpp     # Тест времени транспонирования
│       ├── Vector_test.cpp               # Тест векторов
│       ├── test_sqrt.cpp                 # Тест квадратного корня
│       └── test_eigen.cpp                # Тест собственных значений
├── CMakeLists.txt                        # Конфигурация CMake
└── build/                                (Создается при сборке)
```

## Сборка

### Основная программа (определитель матрицы)

```bash
# Создание директории для сборки
mkdir build
cd build

# Конфигурация проекта
cmake ..

cmake --build . --target determinant_calculator
```

### Запуск программы с определителем

Программа читает размер матрицы `n` и затем `n×n` элементов построчно:

```bash
# Сборка
make determinant_calculator

# Запуск (пример входных данных: размер 2, затем элементы 1 0 0 1)
./determinant_calculator
2
1 0
0 1

# Результат: 1
```

### Сборка тестов

```bash
# Сборка всех тестов
make tests

# Или по отдельности
make main_double        # Тест с double
make main_int           # Тест с int
make main_complex       # Тест с complex
make test_eigen         # Тест собственных значений
make test_sqrt          # Тест квадратного корня

# Запуск теста
./main_double
```

### Опции сборки

```bash
# Отключение AVX оптимизаций
cmake -DENABLE_AVX=OFF ..

# Изменение типа сборки
cmake -DCMAKE_BUILD_TYPE=Debug ..      # Отладочная сборка
cmake -DCMAKE_BUILD_TYPE=Release ..    # Оптимизированная сборка
```

### Очистка

```bash
# Удаление собранных файлов
rm -rf build

# Или только объектов
cd build
make clean
```

## Использование библиотеки

### Базовое создание матриц

```cpp
#include "Matrix.hpp"

// Создание матрицы 3x3
Matrix<double> A(3, 3);

// Заполнение значений
A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;

// Вычисление определителя
double det = A.det();

// Обращение матрицы
auto A_inv = A.inverse();

// Транспонирование
auto A_t = A.transpose();
```

### Конструкторы и фабричные методы

```cpp
// Создание квадратной матрицы
Matrix<double> M1 = Matrix<double>::Square(4);  // 4x4

// Создание прямоугольной матрицы
Matrix<double> M2 = Matrix<double>::Rectangular(3, 5);  // 3x5

// Единичная матрица
Matrix<double> I1 = Matrix<double>::Identity(3, 3);  // 3x3
Matrix<double> I2 = Matrix<double>::Identity(4);     // 4x4

// Нулевая матрица
Matrix<double> Z1 = Matrix<double>::Zero(3, 3);
Matrix<double> Z2 = Matrix<double>::Zero(5);

// Диагональная матрица
std::vector<double> diag = {1.0, 2.0, 3.0};
Matrix<double> D1 = Matrix<double>::Diagonal(3, 3, diag);
Matrix<double> D2 = Matrix<double>::Diagonal(diag);
Matrix<double> D3 = Matrix<double>::Diagonal(4, 4, 5.0);
Matrix<double> D4 = Matrix<double>::Diagonal(3, 7.0);

// Создание из вектора векторов
std::vector<std::vector<double>> data = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};
Matrix<double> M3 = Matrix<double>::From_vector(data);

// Подматрица
Matrix<double> sub = Matrix<double>::Submatrix(M3, 0, 0, 2, 2);
Matrix<double> sub_sq = Matrix<double>::Submatrix(M3, 0, 0, 2);
```

### Блочные матрицы

```cpp
// Создание блочной матрицы 6x6 с блоками 2x3
Matrix<double> B1 = Matrix<double>::BlockMatrix(6, 6, 2, 3);

// Блочно-единичная матрица 8x8 с блоками 4x4
Matrix<double> B2 = Matrix<double>::BlockIdentity(8, 8, 4, 4);

// Блочно-нулевая матрица 9x9 с блоками 3x3
Matrix<double> B3 = Matrix<double>::BlockZero(9, 9, 3, 3);

// Работа с подматрицами
Matrix<double> main_matrix(10, 10);
Matrix<double> sub_block = main_matrix.get_submatrix(2, 2, 4, 4);
```

### Генерация случайных матриц

```cpp
// Генерация случайной матрицы 5x5 с элементами от -10 до 10
Matrix<double> R1 = Matrix<double>::Generate_matrix(5, 5, -10.0, 10.0);

// Генерация с контролем определителя
Matrix<double> R2 = Matrix<double>::Generate_matrix(4, 4, 0.0, 1.0, 
                                                    100,  // iterations
                                                    2.5); // target determinant

// Генерация с контролем числа обусловленности
Matrix<double> R3 = Matrix<double>::Generate_matrix(4, 4, 0.0, 1.0,
                                                    100,    // iterations
                                                    1.0,    // target determinant
                                                    1000.0); // max condition number

// Бинарные матрицы (элементы 0 или 1)
Matrix<int> B = Matrix<int>::Generate_binary_matrix(3, 3, 1);

// Генерация случайного числа для матрицы
double rand_val = Matrix<double>::generate_random(-5.0, 5.0);
double rand_nonzero = Matrix<double>::generate_nonzero_random(-5.0, 5.0);
```

### Арифметические операции

```cpp
Matrix<double> A(2, 2), B(2, 2), C(2, 2);

// Базовые операции
C = A + B;
C = A - B;
C = A * B;
C = A / B;

// Скалярные операции
C = A * 2.5;
C = 3.0 * A;
C = A / 2.0;
C = A + 1.0;
C = A - 1.0;

// Составные операции
A += B;
A -= B;
A *= B;
A *= 2.0;
A /= 2.0;

// Унарные операции
C = -A;

// Сравнение
bool equal = (A == B);
bool not_equal = (A != B);
```

### Свойства и операции

```cpp
Matrix<double> M(3, 3);

// Определитель
double determinant = M.det();
std::optional<double> opt_det = M.try_det();

// След матрицы
std::optional<double> trace = M.trace();

// Нормы
double frob_norm = M.frobenius_norm();
double frob_norm_sq = M.frobenius_norm_squared();

// Проверка свойств
bool symmetric = M.is_symmetric();
bool has_sqrt = M.has_square_root();

// Квадратный корень матрицы
auto sqrt_result = M.safe_sqrt();  // pair<Matrix, bool>
if (sqrt_result.second) {
    Matrix<double> sqrt_M = sqrt_result.first;
}

// Транспонирование
Matrix<double> M_t = M.transpose();
M.transpose_in_place();  // на месте

// Обращение матрицы
Matrix<double> inv = M.inverse();

// Собственные значения и векторы
auto eigenvalues = M.eigenvalues();
auto eigenvectors = M.eigenvectors();
auto eigen_pair = M.eigen();  // pair<values, vectors>

// QR разложение
auto qr_pair = M.qr_decomposition<double>();  // pair<Q, R>
```

### Вспомогательные методы

```cpp
Matrix<double> M(4, 4);

// Получение размеров
int rows = M.get_rows();
int cols = M.get_cols();
int min_dim = M.get_min_dim();

// Получение определителя (если уже вычислен)
std::optional<double> det_cache = M.get_determinant();

// Поиск опорного элемента
std::optional<int> pivot_row = M.find_pivot_in_subcol<double>(1, 1);

// Элементарные преобразования строк
M.swap_rows(0, 1);
M.multiply_row(2, 2.5);
M.add_row_scaled(3, 0, -1.5);

// Проверка элементов
bool is_zero_elem = M.is_zero(0, 0);
bool is_equal = Matrix<double>::is_equal(1.0, 1.0000001);
bool is_zero = Matrix<double>::is_zero(0.0000001);

// Вывод матрицы
M.print();
M.print(5);  // ограничение по размеру
M.precise_print(10);  // высокая точность
M.detailed_print();

// Приведение типов
Matrix<int> int_mat(2, 2);
Matrix<double> double_mat = int_mat.cast_to<double>();
```

## Тестирование

Все тесты находятся в директории `src/tests/`. Для проверки функциональности можно запустить:

```bash
# Пройти все тесты
./main_double
./main_int
./main_complex
./test_eigen
./test_sqrt
```

## Требования

- Компилятор с поддержкой C++17
- CMake 3.10 или выше
- Процессор с поддержкой AVX (опционально, для оптимизаций)
