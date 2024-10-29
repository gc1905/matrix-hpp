# Matrix HPP - C++11 library for matrix class container and linear algebra computations

This library provides a self-contained and easy to use implementation of matrix container class. The main features include:
- Full template parameterization with support for both real and complex datatypes.
- Lightweight and self-contained - single header, no dependencies outside of C++ standard library.
- C++11 based.
- Operator overloading for matrix operations like multiplication and addition.
- Support the basic linear algebra operations, including matrix inversion, factorization and linear equation solving.

## Installation

Copy the [matrix.hpp](matrix.hpp) file into include directory of your project.

## Hello world example

A simple hello world example is provided below. The program creates two matrices with two rows and three columns, and initializes their content with constants. Then, the matrices are added and the resulting matrix is printed to `stdout`.

Note that the `Matrix` class is a template class defined within the `Mtx` namespace. The template parameter specifies the numeric type to represent elements of the matrix container.
 
```cpp
#include <iostream>
#include "matrix.hpp"

void main() {
  Mtx::Matrix<double> A({ 1, 2, 3, 
                          4, 5, 6}, 2, 3);

  Mtx::Matrix<double> B({ 7, 8, 9, 
                         10,11,12}, 2, 3);

  auto C = A + B;

  std::cout << "A + B = [" << C << "];" << std::endl;
}
```

For more examples, refer to [examples.cpp](examples.cpp) file. Remark that not all features of the library are used in the provided examples.

## License

MIT license is used for this project.