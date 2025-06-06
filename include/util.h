// util.h

#pragma once

#include <complex>
#include <string>
#include <Eigen/Dense>

// typedef  std::float64_t  Double;



typedef std::complex<Double> Complex;
typedef Eigen::Vector3<Double> Vec3;
typedef Eigen::Vector4<Double> Vec4;

/// @brief Same as sprintf but generates a std::string
/// @tparam ...Args Template for string arguments
/// @param format Format string (same as sprintf)
/// @param ...args String arguments
/// @return A formatted std::string
template <typename... Args>
std::string string_format(const char* format, Args... args) {
  int size = std::snprintf(nullptr, 0, format, args...) + 1;
  assert(size > 0);
  char buf[size];
  sprintf(buf, format, args...);
  return std::string(buf);
}

/// @brief Check if 2 numbers are almost equal
/// @param x1 First number
/// @param x2 Second number
/// @param eps Numbers are almost equal if their difference is within +/-
/// epsilon
/// @return true if almost equal, false otherwise
bool AlmostEq(const Double x1, const Double x2, Double eps = 1.0e-14) {
  return std::abs(x1 - x2) < eps;
}

/// @brief Check if 2 vectors are almost equal
/// @param v1 First vector
/// @param v2 Second vector
/// @param eps Vectors are almost equal if v1.v2 is within 1 +/- epsilon
/// @return true if almost equal, false otherwise
template<typename V>
bool AlmostEq(const V& v1, const V& v2,
              Double eps = 1.0e-14) {
  return std::abs(v1.dot(v2) - 1.0) < eps;
}
// bool AlmostEq(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2,
//               Double eps = 1.0e-14) {
//   return std::abs(v1.dot(v2) - 1.0) < eps;
// }

/// @brief Chop a floating point number that is close to zero
/// @param x Number to chop
/// @param eps Numbers with absolute values less than epsilon will be chopped
/// @return Chopped number
Double Chop(Double x, Double eps = 1.0e-14) {
  if (std::abs(x) < eps) return 0.0;
  return x;
}

/// @brief Convert a 3-vector to a hashable string
/// @param v 3-Vector
/// @param n_digits Number of digits to trim each coordinate value to
/// @return Hashable string that can be used to identify this vector
std::string Vec3ToString(const Vec3 v, int n_digits = 10) {
  Double eps = pow(10.0, Double(-n_digits));
  Double v_x = Chop(v.x(), eps);
  Double v_y = Chop(v.y(), eps);
  Double v_z = Chop(v.z(), eps);
  return string_format("(%+.*f,%+.*f,%+.*f)", n_digits, v_x, n_digits, v_y,
                       n_digits, v_z);
}

/// @brief Convert a 4-vector to a hashable string
/// @param v 4-Vector
/// @param n_digits Number of digits to trim each coordinate value to
/// @return Hashable string that can be used to identify this vector
std::string Vec4ToString(const Vec4 v, int n_digits = 10) {
  Double eps = pow(10.0, Double(-n_digits));
  Double v_x = Chop(v.x(), eps);
  Double v_y = Chop(v.y(), eps);
  Double v_z = Chop(v.z(), eps);
  Double v_w = Chop(v.w(), eps);
  return string_format("(%+.*f,%+.*f,%+.*f,%+.*f)", n_digits, v_x, n_digits,
                       v_y, n_digits, v_z, n_digits, v_w);
}
