// grp_o4.h

#pragma once

#include <Eigen/Dense>
#include <vector>

typedef Eigen::Quaternion<double> Quat;
typedef Eigen::Vector4<double> Vec4;

// vectors are stored (x,y,z,w) but quaternions are (w,x,y,z)
// it's annoying but this is how Eigen works
#define QuatToVec4(q) Vec4(q.x(), q.y(), q.z(), q.w())
#define Vec4ToQuat(v) Quat(v.w(), v.x(), v.y(), v.z())

class GrpElemO4 {
 public:
  GrpElemO4();
  GrpElemO4(Quat ql, Quat qr, int inv);
  GrpElemO4(double ql[], double qr[], int inv);
  Vec4 operator*(Vec4 const& v) const;
  GrpElemO4 operator*(GrpElemO4 const& g) const;
  void ReadGrpElem(FILE* file);

  Quat ql;  // left quaternion
  Quat qr;  // right quaternion
  int inv;  // -1 if this group element includes an inversion
};

GrpElemO4::GrpElemO4() {
  this->ql = Quat(0.0, 0.0, 0.0, 1.0);
  this->qr = Quat(0.0, 0.0, 0.0, 1.0);
  this->inv = 1;
}

/// @brief O(4) group element constructor
/// @param ql left quaternion
/// @param qr right quaternion
/// @param inv -1 if this group element includes an inversion
GrpElemO4::GrpElemO4(Quat ql, Quat qr, int inv) {
  this->ql = ql;
  this->qr = qr;
  this->inv = inv;
}

/// @brief O(3) group element constructor
/// @param ql left quaternion
/// @param qr right quaternion
/// @param inv -1 if this group element includes an inversion
GrpElemO4::GrpElemO4(double ql[], double qr[], int inv) {
  this->ql = Quat(ql[0], ql[1], ql[2], ql[3]);
  this->qr = Quat(qr[0], qr[1], qr[2], qr[3]);
  this->inv = inv;
}

/// @brief Rotate a vector by this group element
/// @param v Vector to rotate
/// @return Rotated vector
Vec4 GrpElemO4::operator*(Vec4 const& v) const {
  Quat q = Vec4ToQuat(v);
  if (inv == -1) {
    q = q.conjugate();
  }
  Quat gv = ql * q * qr;
  return QuatToVec4(gv);
}

/// @brief Multiply by another group element
/// @param g Group element to multiply
/// @return Product of this element on the left times @p g on the right
GrpElemO4 GrpElemO4::operator*(GrpElemO4 const& g) const {
  if (inv == -1) {
    return GrpElemO4(ql * g.qr.conjugate(), g.ql.conjugate() * qr, inv * g.inv);
  } else {
    return GrpElemO4(ql * g.ql, g.qr * qr, inv * g.inv);
  }
}

void GrpElemO4::ReadGrpElem(FILE* file) {
  int index;
  double ql_w, ql_x, ql_y, ql_z;
  double qr_w, qr_x, qr_y, qr_z;
  fscanf(file, "%d %d %lf %lf %lf %lf %lf %lf %lf %lf\n", &index, &inv, &ql_w,
         &ql_x, &ql_y, &ql_z, &qr_w, &qr_x, &qr_y, &qr_z);
  ql = Quat(ql_w, ql_x, ql_y, ql_z);
  qr = Quat(qr_w, qr_x, qr_y, qr_z);
}