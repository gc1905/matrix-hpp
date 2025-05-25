# Matrix HPP - C++11 library for matrix class container and linear algebra computations

This library provides a self-contained and easy to use implementation of matrix container class. The main features include:
- Full template parameterization with support for both real and complex data-types.
- Lightweight and self-contained - single header, no dependencies outside of C++ standard library.
- C++11 based.
- Operator overloading for matrix operations like multiplication and addition.
- Support the basic linear algebra operations, including matrix inversion, factorization and linear equation solving.

## Installation

Copy the [matrix.hpp](matrix.hpp) file into the include directory of your project.

## Functionality

This library provides the following functionality (but is not limited to):
- Elementary operations: transposition, addition, subtraction, multiplication and element-wise product.
- Matrix determinant.
- Matrix inverse.
- Frobenius norm.
- LU decomposition.
- Cholesky decomposition.
- LDL decomposition.
- Eigenvalue decomposition.
- Hessenberg decomposition.
- QR decomposition.
- Linear equation solving.

For further details please refer to the documentation: [docs/matrix_hpp.pdf](docs/matrix_hpp.pdf). The documentation is auto generated directly from the source code by Doxygen.

## Hello world example

A simple hello world example is provided below. The program defines two matrices with two rows and three columns each, and initializes their content with constant values. Then, the matrices are added together and the resulting matrix is printed to `stdout`.

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

For more examples, refer to [examples/examples.cpp](examples/examples.cpp) file. Remark that not all features of the library are used in the provided examples.

## Debugging

The `MATRIX_STRICT_BOUNDS_CHECK` preprocessor macro controls whether runtime bounds checking is performed for element access operations, e.g., using `operator()`, within the `Matrix` class. When enabled, out-of-bounds access attempts will throw a `std::out_of_range` exception. Please refer to documentation of a specific function for information if it is affected by the `MATRIX_STRICT_BOUNDS_CHECK` preprocessor macro or not.

Disabling bounds checking improves performance but removes protection against errors. It should be disabled only for optimized release builds where peak performance is required.

## Tests

Unit tests are compiled with `make tests`. 

## License

MIT license is used for this project. Please refer to [LICENSE](LICENSE) for details.
