#include <iostream>
#include <complex>
#include <random>
#include <vector>
#include <sstream>

#include "matrix.hpp"

const std::string CONSOLE_COLOR_RESET = "\033[0m";
const std::string CONSOLE_COLOR_RED   = "\033[31m";
const std::string CONSOLE_COLOR_GREEN = "\033[32m";

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
    template<typename T, typename std::enable_if<!Mtx::Util::is_complex<T>::value,int>::type = 0>
    Mtx::Matrix<T> gen_matrix(unsigned rows, unsigned cols) {
      Mtx::Matrix<T> A(rows, cols);
      for (unsigned i = 0; i < A.numel(); i++)
        A(i) = dist(gen);
      return A;
    }

    // complex
    template<typename T, typename std::enable_if<Mtx::Util::is_complex<T>::value,int>::type = 0>
    Mtx::Matrix<T> gen_matrix(unsigned rows, unsigned cols) {
      Mtx::Matrix<T> A(rows, cols);
      for (unsigned i = 0; i < A.numel(); i++) {
        A(i).real(dist(gen));
        A(i).imag(dist(gen));
      }
      return A;
    }

    template<typename T>
    Mtx::Matrix<T> gen_psd_matrix(unsigned rows) {
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
template<typename T> inline const std::string type_str() { return typeid(T).name(); }

template<> inline const std::string type_str<double>() { return "double"; }
template<> inline const std::string type_str<float>() { return "float"; }
template<> inline const std::string type_str<std::complex<double>>() { return "complex double"; }
template<> inline const std::string type_str<std::complex<float>>() { return "complex float"; }

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
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
        error_log << "Test execution ended with unexpected exception: '" << exc.what() << "'" << std::endl;
      }

      if (num_errors == 0)
        std::cout << CONSOLE_COLOR_GREEN << name() << ": PASSED" << CONSOLE_COLOR_RESET << std::endl;
      else 
        std::cout << CONSOLE_COLOR_RED << name() << ": FAILED (" << num_errors << " errors)" << CONSOLE_COLOR_RESET << std::endl << error_log.str() << std::endl;
    }

  protected:
  	virtual const char* name() = 0;
  	virtual void test() = 0;

    void error(const std::string& what) {
      num_errors ++;
      error_log << what << std::endl;
    }

    template<typename T>
    void assertEqual(const Mtx::Matrix<T>& result, const Mtx::Matrix<T>& expected, const std::string& what) {
      if (!result.isequal(expected)) {
        num_errors ++;

        error_log << "Assertion failed for '" << what << "'" << std::endl;
        error_log << "Result = [" << result << "]" << std::endl;
        error_log << "Expected = [" << expected << "]" << std::endl;
      }
    }

    template<typename T>
    void assertEqualTol(const Mtx::Matrix<T>& result, const Mtx::Matrix<T>& expected, T tol, const std::string& what) {
      if (!result.isequal(expected, tol)) {
        num_errors ++;

        error_log << "Assertion failed for '" << what << "'" << std::endl;
        error_log << "Result = [" << result << "]" << std::endl;
        error_log << "Expected = [" << expected << "]" << std::endl;
      }
    }

    template<typename T>
    void assertEqual(const std::vector<T>& result, const std::vector<T>& expected, const std::string& what) {
      if ((! equal(result.begin(), result.end(), expected.begin())) || (result.size() != expected.size())) {
        num_errors ++;

        error_log << "Assertion failed for '" << what << "'" << std::endl;
        error_log << "Result = [" << result << "]" << std::endl;
        error_log << "Expected = [" << expected << "]" << std::endl;
      }
    }

    void assertBoolean(bool result, bool expected, const std::string& what) {
      if (result != expected) {
        num_errors ++;

        error_log << "Assertion failed for '" << what << "'. Result = " << result << ". Expected = " << expected << std::endl;
      }
    }

    void assertDouble(double result, double expected, double tol, const std::string& what) {
      if (abs(result - expected) > tol) {
        num_errors ++;

        error_log << "Assertion failed for '" << what << "'. Result = " << result << ". Expected = " << expected << std::endl;
      }
    }

    // generic function to test throwing of exception on singular matrix input
    // works for single argument functions
    template<typename T, typename F>
    void test_singular_exception(F fun, const std::string& name) {
      try {
        volatile auto x = fun(Mtx::zeros<T>(8,8));
      } catch(Mtx::singular_matrix_exception& e) { 
        return; // expected exception
      }
      error(std::string("Singular doesn't throw ") + name + std::string(" ") + type_str<T>());
    }

    int num_errors;
    std::stringstream error_log;
};