#include <iostream>
#include <complex>
#include <cassert>
#include <random>
#include <vector>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>

#include "matrix.hpp"

using namespace std;
using namespace Mtx;

const string CONSOLE_COLOR_RESET = "\033[0m";
const string CONSOLE_COLOR_RED   = "\033[31m";
const string CONSOLE_COLOR_GREEN = "\033[32m";

/*
 * Utility class for generation of random matrices.
 */

class Matrix_rng {
  public:
    Matrix_rng(unsigned seed) : gen(seed) { }

    virtual ~Matrix_rng() { }

    void reset(unsigned seed) {
      gen.seed(seed);
    }

    // real
    template<typename T, typename std::enable_if<!is_complex<T>::value,int>::type = 0>
    Matrix<T> gen_matrix(unsigned rows, unsigned cols) {
      Matrix<T> A(rows, cols);
      for (unsigned i = 0; i < A.numel(); i++)
        A(i) = dist(gen);
      return A;
    }

    // complex
    template<typename T, typename std::enable_if<is_complex<T>::value,int>::type = 0>
    Matrix<T> gen_matrix(unsigned rows, unsigned cols) {
      Matrix<T> A(rows, cols);
      for (unsigned i = 0; i < A.numel(); i++) {
        A(i).real(dist(gen));
        A(i).imag(dist(gen));
      }
      return A;
    }

    template<typename T>
    Matrix<T> gen_psd_matrix(unsigned rows) {
      auto P = gen_matrix<T>(rows, rows);
      return P * P.ctranspose();
    }

  protected:
    std::mt19937 gen;
    std::normal_distribution<double> dist;
};

/*
 * Utilities for printing human readable typenames. 
 */

// default compiler dependent - may be not human readable 
template<typename T> inline const string type_str() { return typeid(T).name(); }

template<> inline const string type_str<double>() { return "double"; }
template<> inline const string type_str<float>() { return "float"; }
template<> inline const string type_str<complex<double>>() { return "complex double"; }
template<> inline const string type_str<complex<float>>() { return "complex float"; }

template<typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  for (unsigned i = 0; i < v.size(); i ++)
    os << v[i] << " ";
  return os;
}

/*
 * Abstract testcase class definition. 
 */

class Testcase_Abstract {
  public:
    void run() {
      num_errors = 0;

      try {
        test();
      } catch (const std::exception& exc) {
        num_errors ++;
        error_log << "Test execution ended with unexpected exception: '" << exc.what() << "'" << endl;
      }

      if (num_errors == 0)
        cout << CONSOLE_COLOR_GREEN << name() << ": PASSED" << CONSOLE_COLOR_RESET << endl;
      else 
        cout << CONSOLE_COLOR_RED << name() << ": FAILED (" << num_errors << " errors)" << CONSOLE_COLOR_RESET << endl << error_log.str() << endl;
    }

  protected:
  	virtual const char* name() = 0;
  	virtual void test() = 0;

    void error(const string& what) {
      num_errors ++;
      error_log << what << endl;
    }

    template<typename T>
    void assertEqual(const Matrix<T>& result, const Matrix<T>& expected, const string& what) {
      if (!result.isequal(expected)) {
        num_errors ++;

        error_log << "Assertion failed for '" << what << "'" << endl;
        error_log << "Result = [" << result << "]" << endl;
        error_log << "Expected = [" << expected << "]" << endl;
      }
    }

    template<typename T>
    void assertEqualTol(const Matrix<T>& result, const Matrix<T>& expected, T tol, const string& what) {
      if (!result.isequal(expected, tol)) {
        num_errors ++;

        error_log << "Assertion failed for '" << what << "'" << endl;
        error_log << "Result = [" << result << "]" << endl;
        error_log << "Expected = [" << expected << "]" << endl;
      }
    }

    template<typename T>
    void assertEqual(const vector<T>& result, const vector<T>& expected, const string& what) {
      if ((! equal(result.begin(), result.end(), expected.begin())) || (result.size() != expected.size())) {
        num_errors ++;

        error_log << "Assertion failed for '" << what << "'" << endl;
        error_log << "Result = [" << result << "]" << endl;
        error_log << "Expected = [" << expected << "]" << endl;
      }
    }

    void assertBoolean(bool result, bool expected, const string& what) {
      if (result != expected) {
        num_errors ++;

        error_log << "Assertion failed for '" << what << "'. Result = " << result << ". Expected = " << expected << endl;
      }
    }

    void assertDouble(double result, double expected, double tol, const string& what) {
      if (abs(result - expected) > tol) {
        num_errors ++;

        error_log << "Assertion failed for '" << what << "'. Result = " << result << ". Expected = " << expected << endl;
      }
    }

    // generic function to test throwing of exception on singular matrix input
    // works for single argument functions
    template<typename T, typename F>
    void test_singular_exception(F fun, const string& name) {
      try {
        volatile auto x = fun(zeros<T>(8,8));
      } catch(singular_matrix_exception& e) { 
        return; // expected exception
      }
      error(string("Singular doesn't throw ") + name + string(" ") + type_str<T>());
    }

    int num_errors;
    stringstream error_log;
};

/*
 * Testcase implementations.
 */

class TC_Arithmetic: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Basic Arithmetic";
    }


  virtual void test() {
    test_real();
    test_complex();
  }

  void test_real() {
    using T = double;

    auto A2x3 = Matrix<T>({8, 4, 5, 
                           2, 2, 3}, 2, 3);

    auto A3x2 = Matrix<T>({5, 1, 
                           1, 7, 
                           8, 2}, 3, 2);

    vector<T> v2 {3,5};

    assertEqual(add<T,false,false>(A2x3,A2x3), Matrix<T>({16, 8, 10, 4, 4, 6}, 2, 3), string("Addition ") + type_str<T>());
    assertEqual(add<T,true ,false>(A3x2,A2x3), Matrix<T>({13, 5, 13, 3, 9, 5}, 2, 3), string("Addition T1 ") + type_str<T>());
    assertEqual(add<T,false,true >(A2x3,A3x2), Matrix<T>({13, 5, 13, 3, 9, 5}, 2, 3), string("Addition T2 ") + type_str<T>());
    assertEqual(add<T,true ,true >(A3x2,A3x2), Matrix<T>({10, 2, 16, 2, 14, 4}, 2, 3), string("Addition T12 ") + type_str<T>());
    
    assertEqual(subtract<T,false,false>(A2x3,A2x3), Matrix<T>({0, 0, 0, 0, 0, 0}, 2, 3), string("Subtraction ") + type_str<T>());
    assertEqual(subtract<T,true ,false>(A3x2,A2x3), Matrix<T>({-3, -3, 3, -1, 5, -1}, 2, 3), string("Subtraction T1 ") + type_str<T>());
    assertEqual(subtract<T,false,true >(A2x3,A3x2), Matrix<T>({3, 3, -3, 1, -5, 1}, 2, 3), string("Subtraction T2 ") + type_str<T>());
    assertEqual(subtract<T,true ,true >(A3x2,A3x2), Matrix<T>({0, 0, 0, 0, 0, 0}, 2, 3), string("Subtraction T12 ") + type_str<T>());
    
    assertEqual(mult_hadamard<T,false,false>(A2x3,A2x3), Matrix<T>({64, 16, 25, 4, 4, 9}, 2, 3), string("Hadamard ") + type_str<T>());
    assertEqual(mult_hadamard<T,true ,false>(A3x2,A2x3), Matrix<T>({40, 4, 40, 2, 14, 6}, 2, 3), string("Hadamard T1 ") + type_str<T>());
    assertEqual(mult_hadamard<T,false,true >(A2x3,A3x2), Matrix<T>({40, 4, 40, 2, 14, 6}, 2, 3), string("Hadamard T2 ") + type_str<T>());
    assertEqual(mult_hadamard<T,true ,true >(A3x2,A3x2), Matrix<T>({25, 1, 64, 1, 49, 4}, 2, 3), string("Hadamard T12 ") + type_str<T>());
    
    assertEqual(mult<T,false,false>(A2x3,A3x2), Matrix<T>({84, 46, 36, 22}, 2, 2), string("Multiplication ") + type_str<T>());
    assertEqual(mult<T,true ,false>(A2x3,A2x3), Matrix<T>({68, 36, 46, 36, 20, 26, 46, 26, 34}, 3, 3), string("Multiplication T1 ") + type_str<T>());
    assertEqual(mult<T,false,true >(A2x3,A2x3), Matrix<T>({105, 39, 39, 17}, 2, 2), string("Multiplication T2 ") + type_str<T>());
    assertEqual(mult<T,true ,true >(A3x2,A2x3), Matrix<T>({84, 36, 46, 22}, 2, 2), string("Multiplication T12 ") + type_str<T>());

    assertEqual(mult<T,false>(v2,A2x3), vector<T> {34,22,30}, string("Multiplication vec-M ") + type_str<T>());
    assertEqual(mult<T,true >(v2,A3x2), vector<T> {20,38,34}, string("Multiplication vec-M^T ") + type_str<T>());
    assertEqual(mult<T,false>(A3x2,v2), vector<T> {20,38,34}, string("Multiplication M-vec ") + type_str<T>());
    assertEqual(mult<T,true >(A2x3,v2), vector<T> {34,22,30}, string("Multiplication M^T-vec ") + type_str<T>());
  }

  void test_complex() {
    using T = complex<double>;

    auto A2x3 = Matrix<T>({{2,9}, {4,9}, {7,1}, {4,9}, {8,10}, {10,3}}, 2, 3);
    auto A3x2 = Matrix<T>({{10,6}, {9,10}, {6,5}, {5,1}, {5,4}, {7,1}}, 3, 2);

    assertEqual(add<T,false,false>(A2x3,A2x3), Matrix<T>({{4,18}, {8,18}, {14,2}, {8,18}, {16,20}, {20,6}}, 2, 3), string("Addition ") + type_str<T>());
    assertEqual(add<T,true ,false>(A3x2,A2x3), Matrix<T>({{12,3}, {10,4}, {12,-3}, {13,-1}, {13,9}, {17,2}}, 2, 3), string("Addition T1 ") + type_str<T>());
    assertEqual(add<T,false,true >(A2x3,A3x2), Matrix<T>({{12,3}, {10,4}, {12,-3}, {13,-1}, {13,9}, {17,2}}, 2, 3), string("Addition T2 ") + type_str<T>());
    assertEqual(add<T,true ,true >(A3x2,A3x2), Matrix<T>({{20,-12}, {12,-10}, {10,-8}, {18,-20}, {10,-2}, {14,-2}}, 2, 3), string("Addition T12 ") + type_str<T>());

    assertEqual(subtract<T,false,false>(A2x3,A2x3), Matrix<T>({0, 0, 0, 0, 0, 0}, 2, 3), string("Subtraction ") + type_str<T>());
    assertEqual(subtract<T,true ,false>(A3x2,A2x3), Matrix<T>({{8,-15}, {2,-14}, {-2,-5}, {5,-19}, {-3,-11}, {-3,-4}}, 2, 3), string("Subtraction T1 ") + type_str<T>());
    assertEqual(subtract<T,false,true >(A2x3,A3x2), Matrix<T>({{-8,15}, {-2,14}, {2,5}, {-5,19}, {3,11}, {3,4}}, 2, 3), string("Subtraction T2 ") + type_str<T>());
    assertEqual(subtract<T,true ,true >(A3x2,A3x2), Matrix<T>({0, 0, 0, 0, 0, 0}, 2, 3), string("Subtraction T12 ") + type_str<T>());

    assertEqual(mult_hadamard<T,false,false>(A2x3,A2x3), Matrix<T>({{-77,36}, {-65,72}, {48,14}, {-65,72}, {-36,160}, {91,60}}, 2, 3), string("Hadamard ") + type_str<T>());
    assertEqual(mult_hadamard<T,true ,false>(A3x2,A2x3), Matrix<T>({{74,78}, {69,34}, {39,-23}, {126,41}, {50,42}, {73,11}}, 2, 3), string("Hadamard T1 ") + type_str<T>());
    assertEqual(mult_hadamard<T,false,true >(A2x3,A3x2), Matrix<T>({{74,78}, {69,34}, {39,-23}, {126,41}, {50,42}, {73,11}}, 2, 3), string("Hadamard T2 ") + type_str<T>());
    assertEqual(mult_hadamard<T,true ,true >(A3x2,A3x2), Matrix<T>({{64,-120}, {11,-60}, {9,-40}, {-19,-180}, {24,-10}, {48,-14}}, 2, 3), string("Hadamard T12 ") + type_str<T>());

    assertEqual(mult<T,false,false>(A2x3,A3x2), Matrix<T>({{-24,209}, {-13,164}, {22,269}, {43,210}}, 2, 2), string("Multiplication ") + type_str<T>());
    assertEqual(mult<T,true ,false>(A2x3,A2x3), Matrix<T>({{182,0}, {211,-50}, {90,-139}, {211,50}, {261,0}, {147,-135}, {90,139}, {147,135}, {159,0}}, 3, 3), string("Multiplication T1 ") + type_str<T>());
    assertEqual(mult<T,false,true >(A2x3,A2x3), Matrix<T>({{232,0}, {284,39}, {284,-39}, {370,0}}, 2, 2), string("Multiplication T2 ") + type_str<T>());
    assertEqual(mult<T,true ,true >(A3x2,A2x3), Matrix<T>({{-24,-209}, {22,-269}, {-13,-164}, {43,-210}}, 2, 2), string("Multiplication T12 ") + type_str<T>());
  }
};

class TC_Norms: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Matrix Norms";
    }

  virtual void test() {
    auto E = eye<double>(9);
    assertDouble(norm_fro(E), 3.0, 1e-12, "Frobenius eye(9)");
    assertDouble(cond(E), 9.0, 1e-12, "Condition eye(9)");

    Matrix<double> A({ 2.052113613487281e-01,-1.525304055468150e+00,-2.316040468802091e+00,-2.428753334959137e+00,
                      -9.867569768350846e-01, 1.350146369363881e-01,-2.874386478765925e-02,-8.111123486654980e-01,
                      -6.499990607857457e-01, 3.725026263812751e-01,-1.160188911359364e+00,-2.879539632448196e+00,
                       1.654519282512769e-01, 1.517864378728704e-01,-4.698341950083968e-01,-1.813775783139517e+00}, 4, 4);

    assertDouble(norm_fro(A), 5.389079896536130, 1e-12, "Frobenius 4x4");
    assertDouble(cond(A), 21.90459850374101, 1e-12, "Condition 4x4");
  }
};

class TC_Determinant: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Matrix Determinant";
    }

  virtual void test() {
    Matrix<double> A4({ 2.052113613487281e-01,-1.525304055468150e+00,-2.316040468802091e+00,-2.428753334959137e+00,
                       -9.867569768350846e-01, 1.350146369363881e-01,-2.874386478765925e-02,-8.111123486654980e-01,
                       -6.499990607857457e-01, 3.725026263812751e-01,-1.160188911359364e+00,-2.879539632448196e+00,
                        1.654519282512769e-01, 1.517864378728704e-01,-4.698341950083968e-01,-1.813775783139517e+00}, 4, 4);

    Matrix<double> A3 = A4.get_submatrix(0,2,0,2);
    Matrix<double> A2 = A4.get_submatrix(0,1,0,1);
    Matrix<double> A1 = A4.get_submatrix(0,0,0,0);

    double d4 = det(A4);
    double d3 = det(A3);
    double d2 = det(A2);
    double d1 = det(A1);

    assertDouble(d4, -2.137653742895919, 1e-12, "Determinant 4x4");
    assertDouble(d3,  2.335811572066522, 1e-12, "Determinant 3x3");
    assertDouble(d2, -1.477397881080325, 1e-12, "Determinant 2x2");
    assertDouble(d1,  0.205211361348728, 1e-12, "Determinant 1x1");

    assertDouble(det(ones<double>(4,4)), 0, 1e-12, "Determinant 4x4 ones");
    assertDouble(det(zeros<double>(7,7)), 0, 1e-12, "Determinant 7x7 zeros");
    assertDouble(det(eye<double>(9)), 1.0, 1e-12, "Determinant 9x9 eye");
  }
};

class TC_Elementwise: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Elementwise Functions";
    }

  virtual void test() {
    Matrix<double> A({ 2.450025742693661e+00, 5.658436100316567e-01,-2.450833326441732e+00,
                       5.658436100316567e-01, 3.399770114564154e+00, 3.008401682318424e-01,
                      -2.450833326441732e+00, 3.008401682318424e-01, 3.546190310517363e+00}, 3, 3);

    Matrix<double> sinA_ref({ 6.377448740959660e-01, 5.361281288208246e-01,-6.371226266961434e-01,
                              5.361281288208246e-01,-2.553188424525822e-01, 2.963227456345058e-01,
                             -6.371226266961434e-01, 2.963227456345058e-01,-3.936489340065361e-01}, 3, 3);

    Matrix<double> sinA = foreach_elem_copy<double>(A, [](double x) -> double {return sin(x);});

    assertEqualTol(sinA, sinA_ref, 1e-12, "Elementwise sin");
  }
};

class TC_Permutations: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Row and Column Permutations";
    }

  virtual void test() {
    Matrix<int> A({1, 2, 3, 4,
                   4, 5, 6, 7,
                   8, 9,10,11}, 3, 4);

    std::vector<unsigned> permute_row({1, 2, 0, 2, 1});
    Matrix<int> A_permute_row_ref({4, 5, 6, 7,
                                   8, 9,10,11,
                                   1, 2, 3, 4,
                                   8, 9,10,11,
                                   4, 5, 6, 7}, 5, 4);
    auto A_permute_row = permute_rows(A, permute_row);

    std::vector<unsigned> permute_col({2,0,1,3});
    Matrix<int> A_permute_col_ref({ 3, 1, 2, 4,
                                    6, 4, 5, 7,
                                   10, 8, 9,11}, 3, 4);
    auto A_permute_col = permute_cols(A, permute_col);

    assertEqual(A_permute_row, A_permute_row_ref, "Row permutation");
    assertEqual(A_permute_col, A_permute_col_ref, "Column permutation");
  }
};

class TC_Concatenation: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Matrix Concatenation";
    }

  virtual void test() {
    Matrix<int> A({1, 2, 3,
                   4, 5, 6}, 2, 3);
    Matrix<int> B({7, 8,
                   9,10}, 2, 2);
    Matrix<int> C({1, 2, 3, 7, 8,
                   4, 5, 6, 9,10}, 2, 5);
    
    assertEqual(concatenate_horizontal(A,B), C, "Horizontal concatenation");

    Matrix<int> D({ 1, 2, 3,
                    4, 5, 6}, 2, 3);
    Matrix<int> E({ 7, 8, 9,
                   10,11,12,
                   13,14,15}, 3, 3);
    Matrix<int> F({ 1, 2, 3,
                    4, 5, 6,
                    7, 8, 9,
                   10,11,12,
                   13,14,15}, 5, 3);

    assertEqual(concatenate_vertical(D,E), F, "Vertical concatenation");
  }
};

class TC_CholeskyDecomposition: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Cholesky Decomposition";
    }

  virtual void test() {
    test_singular<double>();
    test_singular<complex<double>>();
    test_standard<double>(1);
    test_standard<complex<double>>(2);
  }

  template<typename T>
  void test_singular() {
    test_singular_exception<T>(chol<T,false>, "Cholesky lower");
    test_singular_exception<T>(chol<T,true >, "Cholesky upper");
  }

  template<typename T>
  void test_standard(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A = rng.gen_psd_matrix<T>(4);
    auto L = chol<T,false>(A);
    auto U = chol<T,true >(A);
    auto Linv = cholinv(A);

    assertBoolean(istril(L), true, string("L is lower triangular ") + type_str<T>());
    assertBoolean(istriu(U), true, string("U is upper triangular ") + type_str<T>());
    assertEqualTol(L * L.ctranspose(), A, static_cast<T>(1e-12), string("Cholesky lower ") + type_str<T>());
    assertEqualTol(U.ctranspose() * U, A, static_cast<T>(1e-12), string("Cholesky upper ") + type_str<T>());
    assertEqualTol(L * Linv, eye<T>(4), static_cast<T>(1e-12), string("Cholesky inverse ") + type_str<T>());
  }
};

class TC_LdlDecomposition: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "LDL Decomposition";
    }

  virtual void test() {
    test_singular<double>();
    test_singular<complex<double>>();

    test_standard<double>(1);
    test_standard<complex<double>>(2);
  }

  template<typename T>
  void test_singular() {
    test_singular_exception<T>(ldl<T>, "LDL");
  }

  template<typename T>
  void test_standard(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A = rng.gen_psd_matrix<T>(4);

    auto LDL = ldl(A);
    auto A_rec = LDL.L * diag(LDL.d) * LDL.L.ctranspose();

    assertBoolean(istril(LDL.L), true, string("L is lower triangular ") + type_str<T>());
    assertEqualTol(A_rec, A, static_cast<T>(1e-12), string("LDL reconstructed ") + type_str<T>());
  }
};

class TC_MatrixInversion: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Matrix Inversion";
    }

  virtual void test() {
    test_small<double>();
    test_small<complex<double>>();

    test_posdef<double>();
    test_posdef<complex<double>>();

    test_tril<double>();
    test_tril<complex<double>>();

    test_triu<double>();
    test_triu<complex<double>>();

    test_square<double>();
    test_square<complex<double>>();

    test_pinv<double>();
    test_pinv<complex<double>>();

    test_singular<double>();
    test_singular<complex<double>>();
  }

  template<typename T>
  void test_small(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A3 = rng.gen_matrix<T>(3,3);
    auto A2 = rng.gen_matrix<T>(2,2);
    auto A1 = rng.gen_matrix<T>(1,1);

    assertEqualTol(A3 * inv(A3), eye<T>(3), static_cast<T>(1e-12), string("Matrix inverse 3x3 ") + type_str<T>());
    assertEqualTol(A2 * inv(A2), eye<T>(2), static_cast<T>(1e-12), string("Matrix inverse 2x2 ") + type_str<T>());
    assertEqualTol(A1 * inv(A1), eye<T>(1), static_cast<T>(1e-12), string("Matrix inverse 1x1 ") + type_str<T>());
  }

  template<typename T>
  void test_posdef(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A = rng.gen_psd_matrix<T>(4);
    assertEqualTol(A * inv_posdef(A), eye<T>(4), static_cast<T>(1e-12), string("Posdef inverse ") + type_str<T>());
  }

  template<typename T>
  void test_tril(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A = tril(rng.gen_matrix<T>(4,4));
    assertEqualTol(A * inv_tril(A), eye<T>(4), static_cast<T>(1e-12), string("Triangular lower inverse ") + type_str<T>());
  }
  
  template<typename T>
  void test_triu(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A = triu(rng.gen_matrix<T>(4,4));
    assertEqualTol(A * inv_triu(A), eye<T>(4), static_cast<T>(1e-12), string("Triangular upper inverse ") + type_str<T>());
  }

  template<typename T>
  void test_pinv(unsigned seed = 1) {
    Matrix_rng rng(seed);
    
    auto A = rng.gen_matrix<T>(4,3);
    assertEqualTol(pinv(A) * A, eye<T>(3), static_cast<T>(1e-12), string("Pinv left ") + type_str<T>());

    auto B = A.transpose();
    assertEqualTol(B * pinv(B), eye<T>(3), static_cast<T>(1e-12), string("Pinv right ") + type_str<T>());
  }

  template<typename T>
  void test_square(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A4 = rng.gen_matrix<T>(4,4);
    assertEqualTol(A4 * inv(A4), eye<T>(4), static_cast<T>(1e-12), string("Matrix inverse 4x4 ") + type_str<T>());
    assertEqualTol(A4 * inv_gauss_jordan(A4), eye<T>(4), static_cast<T>(1e-12), string("Matrix inverse 4x4 (Gauss-Jordan) ") + type_str<T>());
    assertEqualTol(A4 * pinv(A4), eye<T>(4), static_cast<T>(1e-12), string("Matrix inverse 4x4 (Pseudo inverse) ") + type_str<T>());
  }

  template<typename T>
  void test_singular() {
    test_singular_exception<T>(inv_posdef<T>, "inv_posdef");
    test_singular_exception<T>(inv_tril<T>, "inv_tril");
    test_singular_exception<T>(inv_triu<T>, "inv_triu");
    test_singular_exception<T>(inv<T>, "inv");
  }
};

class TC_LuDecomposition: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "LU Decomposition";
    }

  virtual void test() {
    test_lu<double>();
    test_lu<complex<double>>();

    test_lup<double>();
    test_lup<complex<double>>();
  }

  template<typename T>
  void test_lu(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A4x4 = rng.gen_matrix<T>(4,4);
    auto A4x3 = A4x4.get_submatrix(0, 3, 0, 2);
    auto A3x4 = A4x4.get_submatrix(0, 2, 0, 3);

    LU_result<T> res;
    vector<unsigned> P_identity;

    P_identity.resize(4);
    for (int i = 0; i < 4; i++)
      P_identity[i] = i;

    res = lu(A4x4);
    checkResults("LU 4x4", A4x4, res.L, res.U, P_identity);

    res = lu(A3x4);
    checkResults("LU 3x4", A3x4, res.L, res.U, P_identity);

    P_identity.resize(3);
    res = lu(A4x3);
    checkResults("LU 4x3", A4x3, res.L, res.U, P_identity);
  }

  template<typename T>
  void test_lup(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A4x4 = rng.gen_matrix<T>(4,4);
    auto A4x3 = A4x4.get_submatrix(0, 3, 0, 2);
    auto A3x4 = A4x4.get_submatrix(0, 2, 0, 3);

    LUP_result<T> res;

    res = lup(A4x4);
    checkResults("LU pivot 4x4", A4x4, res.L, res.U, res.P);

    res = lup(A3x4);
    checkResults("LU pivot 3x4", A3x4, res.L, res.U, res.P);

    res = lup(A4x3);
    checkResults("LU pivot 4x3", A4x3, res.L, res.U, res.P);
  }

  template<typename T>
  void checkResults(const string& name, const Matrix<T>& A, const Matrix<T>& L, const Matrix<T>& U, const vector<unsigned>& P) {
    auto A_rec = permute_cols(L * U, P);

    assertEqualTol(A_rec, A, static_cast<T>(1e-12), name + string(" reconstructed ") + type_str<T>());
    assertBoolean(istril(L), true, name + string(" L is lower triangular ") + type_str<T>());
    assertBoolean(istriu(U), true, name + string(" R is upper triangular ") + type_str<T>());
  }
};

class TC_QrDecomposition: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "QR Decomposition";
    }

  virtual void test() {
    test_standard<double>();
    test_standard<complex<double>>();
  }

  template<typename T>
  void test_standard(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A4x4 = rng.gen_matrix<T>(4,4);
    auto A4x3 = rng.gen_matrix<T>(4,3);
    auto A3x4 = rng.gen_matrix<T>(3,4);

    QR_result<T> res;

    res = qr_householder(A4x4);
    checkResults("QR House 4x4", A4x4, res);

    res = qr_householder(A4x3);
    checkResults("QR House 4x3", A4x3, res);

    res = qr_householder(A3x4);
    checkResults("QR House 3x4", A3x4, res);

    res = qr_red_gs(A4x4);
    checkResults("QR GS 4x4", A4x4, res);

    res = qr_red_gs(A4x3);
    checkResults("QR GS 4x3", A4x3, res, false);

    res = qr_red_gs(A3x4);
    checkResults("QR GS 3x4", A3x4, res, false);
  }

  template<typename T>
  void checkResults(const string& name, const Matrix<T>& A, const QR_result<T>& QR, bool check_orth = true) {
    assertEqualTol(QR.Q * QR.R, A, static_cast<T>(1e-12), name + string(" reconstructed ") + type_str<T>());
    assertBoolean(istriu(QR.R), true, name + string(" R is upper triangular ") + type_str<T>());
    if (check_orth)
      assertEqualTol(QR.Q * QR.Q.ctranspose(), eye<T>(QR.Q.rows()), static_cast<T>(1e-12), name + string(" Q is orthogonal ") + type_str<T>());
  }
};

class TC_HessenbergDecomposition: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Hessenberg Decomposition";
    }

  virtual void test() {
    test_standard<double>();
    test_standard<complex<double>>();
  }

  template<typename T>
  void test_standard(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A = rng.gen_matrix<T>(8,8);

    auto res = hessenberg(A);
    auto& H = res.H;
    auto& Q = res.Q;

    assertEqualTol(Q * H * Q.ctranspose(), A, static_cast<T>(1e-12), string("Hessenberg reconstructed ") + type_str<T>());
    assertEqualTol(Q * Q.ctranspose(), eye<T>(A.rows()), static_cast<T>(1e-12), string("Q is orthogonal ") + type_str<T>());
    assertBoolean(ishess(H), true, string("H is Hessenberg matrix ") + type_str<T>());
  }
};

class TC_Eigenvalues: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Eigenvalues";
    }

  virtual void test() {
    test_diagonal();
    test_full();
  }

  void test_diagonal() {
    auto A = diag(vector<double>({1.2, 3.1, 5, 0.1}));
    auto res = eigenvalues(A, 1e-16, 10);

    assertEqualTol(diag(res.eig), make_complex(A), {1e-12,0}, "Eigenvalues of diagonal");
  }

  void test_full(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto d0 = vector<complex<double>>({3, 10, 2, 0.5});
    auto V = make_complex(rng.gen_matrix<double>(4,4));

    auto res = eigenvalues(V * diag(d0) * inv(V), 1e-16, 200);
    auto& d = res.eig;
    
    // sort d0 and d by abs value before comparison
    auto f = [](const complex<double>& a, const complex<double>& b) { 
      return std::abs(a) < std::abs(b); 
    };

    std::sort(d0.begin(), d0.end(), f);
    std::sort(d.begin(), d.end(), f);

    assertEqualTol(diag(d), diag(d0), {1e-10,0}, "Eigenvalues of 4x4");
  }
};


class TC_Solvers: public Testcase_Abstract {
  protected:
    virtual const char* name() {
      return "Linear System Solvers";
    }

  virtual void test() {
    test_triu<double>();
    test_triu<complex<double>>();

    test_tril<double>();
    test_tril<complex<double>>();

    test_square<double>();
    test_square<complex<double>>();

    test_posdef<double>();
    test_posdef<complex<double>>();

    test_singular<double>();
    test_singular<complex<double>>();
  }

  template<typename T>
  void test_triu(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto U = triu(rng.gen_matrix<T>(4,4));
    auto X_ex = rng.gen_matrix<T>(4,3);
    auto B = U * X_ex;

    auto X = solve_triu(U, B);

    assertEqualTol(X, X_ex, static_cast<T>(1e-12), string("Upper Triangular Solver ") + type_str<T>());
  }

  template<typename T>
  void test_tril(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto L = tril(rng.gen_matrix<T>(4,4));
    auto X_ex = rng.gen_matrix<T>(4,3);
    auto B = L * X_ex;
    
    auto X = solve_tril(L, B);

    assertEqualTol(X, X_ex, static_cast<T>(1e-12), string("Lower Triangular Solver ") + type_str<T>());
  }

  template<typename T>
  void test_square(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A = rng.gen_matrix<T>(4,4);
    auto X_ex = rng.gen_matrix<T>(4,3);
    auto B = A * X_ex;
    
    auto X = solve_square(A, B);

    assertEqualTol(X, X_ex, static_cast<T>(1e-12), string("Square Solver ") + type_str<T>());
  }

  template<typename T>
  void test_posdef(unsigned seed = 1) {
    Matrix_rng rng(seed);

    auto A = rng.gen_psd_matrix<T>(4);
    auto X_ex = rng.gen_matrix<T>(4,3);
    auto B = A * X_ex;
    
    auto X = solve_posdef(A, B);

    assertEqualTol(X, X_ex, static_cast<T>(1e-12), string("Posdef Solver ") + type_str<T>());
  }

  template<typename T>
  void test_singular() {
    test_singular_exception_in_solve<T>(solve_posdef<T>, "solve_posdef");
    test_singular_exception_in_solve<T>(solve_tril<T>, "solve_tril");
    test_singular_exception_in_solve<T>(solve_triu<T>, "solve_triu");
    test_singular_exception_in_solve<T>(solve_square<T>, "solve_square");
  }

  template<typename T, typename F>
  void test_singular_exception_in_solve(F fun, const string& name) {
    try {
      volatile auto x = fun(zeros<T>(8,8), zeros<T>(8,8));
    } catch(singular_matrix_exception& e) { 
      return; // expected exception
    }
    error(string("Singular doesn't throw ") + name + string(" ") + type_str<T>());
  }
};


int main() {
  TC_Arithmetic().run();
  TC_Norms().run();
  TC_Elementwise().run();
  TC_Determinant().run();
  TC_Permutations().run();
  TC_Concatenation().run();
  TC_CholeskyDecomposition().run();
  TC_LdlDecomposition().run();
  TC_MatrixInversion().run();
  TC_LuDecomposition().run();
  TC_QrDecomposition().run();
  TC_HessenbergDecomposition().run();
  TC_Eigenvalues().run();
  TC_Solvers().run();

  return 0;
}