/** \file examples.cpp
 * 
 *  Provides various examples of matrix.hpp library usage.
 */

#include <iostream>
#include <complex>
#include <cassert>
#include <random>
#include <vector>

#include "matrix.hpp"

using namespace std;
using namespace Mtx;

void example_1() {
  cout << "-----------------------------" << endl;
  cout << "Example 1" << endl;

  Matrix<double> A({ 1, 2, 3, 
                     4, 5, 6}, 2, 3);

  Matrix<double> B({ 7, 8, 9, 
                    10,11,12}, 2, 3);

  auto C = A + B;

  cout << "A + B = [" << C << "];" << endl;
}

void example_2() {
  cout << "-----------------------------" << endl;
  cout << "Example 2" << endl;

  // Create empty 3x3 matrix byt providing number of rows and columns as contructor arguments.
  // The type of the matrix is provided as template argument.
  Matrix<double> A(3, 3);

  // Fill matrix A with consecutive numbers using for loop and element access. 
  // Note that single index access is using row-major format.
  for (unsigned i = 0; i < A.numel(); i++)
    A(i) = i+1;

  // Add column index to each element.
  // Obviously, double index access is used here for row and column indexing.
  for (unsigned r = 0; r < A.rows(); r++)
    for (unsigned c = 0; c < A.cols(); c++)
      A(r,c) += c;

  // There are operators defined by matrix-matrix and matrix-scalar addition (+), multiplication (*), subtraction (-),
  // elementwise multiplication (^, scalar only) and division (/, scalar only).
  // Here, we multiply all elements by 2.
  A = A * 2.0;

  // Make A lower triangular by trunating elements above diagonal. There is a function for this.
  A = tril(A);

  // Calculate positive definite matrix C as a product of A and its transpose.
  auto C = A * A.transpose();

  // Note that this operation can be done more efficiently due to zero-copy using mult<> method by specifying 
  // argument matrix transposition as template parameters. 
  C = mult<double,false,true>(A,A); 

  // Calculate Cholesky decomposition of C.
  auto C_sqrt = chol(C);

  // Print matrices
  cout << "A = [" << A << "];" << endl;
  cout << "C_sqrt = [" << C_sqrt << "];" << endl;

  // At this point, C_sqrt should be equal to A.
  // Assert equality of C_sqrt and A using Frobenius norm.
  assert(norm_fro(C_sqrt - A) / norm_fro(A) < 1e-12);
}

void example_3() {
  cout << "-----------------------------" << endl;
  cout << "Example 3" << endl;

  // Initialize matrix. Unlike in element indexing, column major format is used here for readability.
  Matrix<complex<double>> A({{1.1,1.0}, 2.0, 3.4, 
                             4.0,       3.5, 5.2, 
                             1.0,     -10.0, 1.0}, 3, 3);

  // QR decomposition. Note that the function returns QR_result structure containing Q and R matrices.
  auto qrr = qr(A);

  // Defines aliases for Q and R matrices.
  auto& Q = qrr.Q;
  auto& R = qrr.R;

  // Reconstructed A matrix from Q and R.
  auto A_rec = Q * R;

  // Test othogonality of matrix Q.
  auto orth = eye<complex<double>>(A.rows()) - mult<complex<double>,false,true>(Q,Q);

  // Print matrices
  cout << "A = [" << A << "];" << endl;
  cout << "A_rec = [" << A_rec << "];" << endl;

  // A_rec and A should be equal.
  // Assert equality using isequal method with selected tolerance.
  assert(A.isequal(A_rec, {1e-12,0}));

  // Matrix orth should be close to zero matrix
  assert(orth.isequal(zeros<complex<double>>(orth.rows(),orth.cols()), 1e-12));
  // The same can be checked using Frobenius norm
  assert(norm_fro(orth) < 1e-12);
}

int main() {
  example_1();
  example_2();
  example_3();

  return 0;
}