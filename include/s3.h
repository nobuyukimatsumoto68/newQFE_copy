// s3.h

#pragma once

#include <Eigen/Dense>
#include <boost/math/special_functions/gegenbauer.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>

#include "grp_o4.h"
#include "lattice.h"
#include "util.h"

// symmetry group data directory must be defined
#ifndef GRP_DIR
#error Error: GRP_DIR is not defined (it should be in the Makefile)
#endif

/// @brief Lattice discretization of a 3-sphere.
class QfeLatticeS3 : public QfeLattice {
 public:
  QfeLatticeS3(int q, int k = 1);
  void ReadBaseLattice(int q);
  void ResizeSites(int n_sites);
  void WriteSite(FILE* file, int s);
  void ReadSite(FILE* file, int s);
  int CreateOrbit(double xi1, double xi2, double xi3);
  Vec4 CalcOrbitPos(int id);
  void ReadOrbits(FILE* file);
  void WriteOrbits(FILE* file);
  void UpdateOrbits();
  void ReadSymmetryData(int q, int k);
  void Inflate();
  void UpdateAntipodes();
  void CalcFEMWeights();
  double CalcLatticeSpacing();
  Complex GetYjlm(int s, int j, int l, int m);
  double CosTheta(int s1, int s2);
  Vec4 EdgeCenter(int l);
  Vec4 FaceCircumcenter(int f);
  Vec4 CellCircumcenter(int c);
  double EdgeLength(int l);
  double CellVolume(int c);
  void PrintCoordinates();

  int q;                        // base polytope
  std::vector<Vec4> r;          // site coordinates
  std::vector<int> antipode;    // antipode of each site (0 by default)
  std::vector<int> site_orbit;  // orbit id for each site
  std::vector<int> cell_orbit;  // orbit id for each cell
  std::vector<Vec4> orbit_xi;   // barycentric coordinates for each orbit
  Vec4 first_cell_r[4];         // coordinates of first cell vertices
  std::vector<GrpElemO4> G;     // symmetry group elements
  std::vector<int> site_g;      // group element for each site
};

/// @brief Create a simplicial discretization of a 3-sphere.
/// @param q Number of links meeting at each site. Valid values for @p q are 3,
/// 4, and 5 for a tetrahedron, octahedron, and icosahedron, respectively.
/// @param k Refinement level.
QfeLatticeS3::QfeLatticeS3(int q, int k) {
  // refinement level must be positive
  assert(k >= 1);

  // 5-cell, 16-cell, and 600-cell are the only valid base polytopes
  assert(q >= 3 && q <= 5);

  this->q = q;

  // return base lattice if unrefined
  if (k == 1) {
    ReadBaseLattice(q);
    UpdateDistinct();
    ReadSymmetryData(q, k);
    return;
  }

  if (q == 3) {
    // k-refined 5-cell has 5 * k * (k^2 + 2) / 3 vertices
    // and 5 * k * (5 k^2 - 2) / 3 cells
    ResizeSites((5 * k * (k * k + 2)) / 3);
  } else if (q == 4) {
    // k-refined 16-cell has 8 * k * (2 * k^2 + 1) / 3 vertices
    // and 16 * k * (5 k^2 - 2) / 3 cells
    ResizeSites((8 * k * (2 * k * k + 1)) / 3);
  } else if (q == 5) {
    // k-refined 600-cell has 40 * k * (5 * k^2 - 2) vertices
    // and 200 * k * (5 k^2 - 2) cells
    ResizeSites(40 * k * (5 * k * k - 2));
  }

  // set the lattice volume
  vol = double(n_sites);

  // create an unrefined base lattice
  QfeLatticeS3 base_lattice(q);

  std::unordered_map<std::string, int> coord_map;
  std::unordered_map<std::string, int> orbit_map;
  std::unordered_map<std::string, int> cell_map;
  int s_next = 0;
  int c_next = 0;

  // xyz coordinates of each site
  std::vector<int> site_x(n_sites);
  std::vector<int> site_y(n_sites);
  std::vector<int> site_z(n_sites);

  // loop over cells of base polytope
  for (int c = 0; c < base_lattice.n_cells; c++) {
    // get the vertices of the base polytope cell
    Vec4 cell_r[4];
    for (int i = 0; i < 4; i++) {
      int b = base_lattice.cells[c].sites[i];
      cell_r[i] = base_lattice.r[b];
      if (c == 0) {
        // set the coordinates of the first cell's vertices
        first_cell_r[i] = cell_r[i];
      }
    }

    // calculate unit vectors for this cell in the cubic xyz basis
    Vec4 n_x = -(cell_r[0] + cell_r[1] - cell_r[2] - cell_r[3]) / double(2 * k);
    Vec4 n_y = -(cell_r[0] - cell_r[1] + cell_r[2] - cell_r[3]) / double(2 * k);
    Vec4 n_z = -(cell_r[0] - cell_r[1] - cell_r[2] + cell_r[3]) / double(2 * k);

    // list of sites in this cell labeled by xyz positions in the cube
    int k1 = k + 1;
    int xyz_max = k1 * k1 * k1;
    int xyz_list[k1][k1][k1];

    // loop over xyz to find all sites and set their positions
    for (int xyz = 0; xyz <= xyz_max; xyz++) {
      int x = xyz % k1;
      int y = (xyz / k1) % k1;
      int z = xyz / (k1 * k1);

      // skip xyz values outside the tetrahedral cell
      if ((x + y + z) > (2 * k)) continue;
      if (x + y < z) continue;
      if (y + z < x) continue;
      if (z + x < y) continue;

      // calculate the coordinates of this vertex
      Vec4 v = cell_r[0] + x * n_x + y * n_y + z * n_z;

      // deal with negative zero
      std::string vec_name = Vec4ToString(v.normalized());

      // check if the site already exists
      if (coord_map.find(vec_name) == coord_map.end()) {
        // barycentric coordinates
        int xi[4];
        xi[0] = -x + y + z;
        xi[1] = +x - y + z;
        xi[2] = +x + y - z;
        xi[3] = 2 * k - x - y - z;
        std::sort(xi, xi + 4, std::greater<int>());
        std::string orbit_name =
            string_format("%d_%d_%d_%d", xi[0], xi[1], xi[2], xi[3]);

        // check if the orbit already exists
        if (orbit_map.find(orbit_name) == orbit_map.end()) {
          // create a new orbit
          double xi1 = double(xi[0]) / double(2 * k);
          double xi2 = double(xi[1]) / double(2 * k);
          double xi3 = double(xi[2]) / double(2 * k);
          orbit_map[orbit_name] = CreateOrbit(xi1, xi2, xi3);
        }

        // create a new site
        int orbit_id = orbit_map[orbit_name];
        coord_map[vec_name] = s_next;
        r[s_next] = v;
        sites[s_next].nn = 0;
        sites[s_next].wt = 1.0;
        sites[s_next].id = orbit_id;
        site_orbit[s_next] = orbit_id;
        s_next++;
        assert(s_next <= n_sites);
      }

      // get the site index
      int s = coord_map[vec_name];

      // save this site in the xyz list
      xyz_list[x][y][z] = s;
      site_x[s] = x;
      site_y[s] = y;
      site_z[s] = z;
    }

    // add cells, faces, and links
    for (int xyz = 0; xyz <= xyz_max; xyz++) {
      int x = xyz % k1;
      int y = (xyz / k1) % k1;
      int z = xyz / (k1 * k1);
      if ((x + y + z) > (2 * k)) continue;
      if (x + y < z) continue;
      if (y + z < x) continue;
      if (z + x < y) continue;

      if ((x + y + z) & 1) {
        // odd site -> octahedron center
        int s0 = xyz_list[x][y][z];
        int s1 = xyz_list[x - 1][y][z];
        int s2 = xyz_list[x][y - 1][z];
        int s3 = xyz_list[x][y][z - 1];
        int s4 = xyz_list[x + 1][y][z];
        int s5 = xyz_list[x][y + 1][z];
        int s6 = xyz_list[x][y][z + 1];
        AddCell(s0, s1, s2, s3);
        AddCell(s0, s1, s2, s6);
        AddCell(s0, s1, s5, s3);
        AddCell(s0, s1, s5, s6);
        AddCell(s0, s4, s2, s3);
        AddCell(s0, s4, s2, s6);
        AddCell(s0, s4, s5, s3);
        AddCell(s0, s4, s5, s6);
        continue;
      }

      if ((x + y + z) != (2 * k)) {
        // even site -> "forward" tetrahedron
        int s1 = xyz_list[x][y][z];
        int s2 = xyz_list[x][y + 1][z + 1];
        int s3 = xyz_list[x + 1][y][z + 1];
        int s4 = xyz_list[x + 1][y + 1][z];
        AddCell(s1, s2, s3, s4);
      }

      if ((x + y >= z + 2) && (y + z >= x + 2) && (z + x >= y + 2)) {
        // even site -> "backward" tetrahedron
        int s1 = xyz_list[x][y][z];
        int s2 = xyz_list[x][y - 1][z - 1];
        int s3 = xyz_list[x - 1][y][z - 1];
        int s4 = xyz_list[x - 1][y - 1][z];
        AddCell(s1, s2, s3, s4);
      }
    }

    // set the distinct id for each cell
    int n_distinct_cells = 0;
    cell_orbit.resize(n_cells);
    while (c_next != n_cells) {
      int xi[4] = {0, 0, 0, 0};
      for (int i = 0; i < 4; i++) {
        int s = cells[c_next].sites[i];
        int x = site_x[s];
        int y = site_y[s];
        int z = site_z[s];

        xi[0] += -x + y + z;
        xi[1] += +x - y + z;
        xi[2] += +x + y - z;
        xi[3] += 2 * k - x - y - z;
      }

      std::sort(xi, xi + 4, std::greater<int>());
      std::string cell_name =
          string_format("%d_%d_%d_%d", xi[0], xi[1], xi[2], xi[3]);

      // check if the cell already exists
      if (cell_map.find(cell_name) == cell_map.end()) {
        cell_map[cell_name] = n_distinct_cells++;
        // printf("%04d %s\n", cell_map[cell_name], cell_name.c_str());
      }
      cell_orbit[c_next] = cell_map[cell_name];
      c_next++;
    }
  }

  // check that all of the sites have been created
  assert(s_next == n_sites);

  UpdateDistinct();
  ReadSymmetryData(q, k);
}

/// @brief Read base polyhedron data from grp directory
/// @param q polytope parameter
void QfeLatticeS3::ReadBaseLattice(int q) {
  // 5-cell, 16-cell, and 600-cell are the only valid base polytopes
  assert(q >= 3 && q <= 5);
  this->q = q;

  // read the base lattice file in the symmetry group directory
  std::string lattice_path = string_format("%s/lattice/o4q%d.dat", GRP_DIR, q);
  FILE* file = fopen(lattice_path.c_str(), "r");
  assert(file != nullptr);
  ReadLattice(file);
  fclose(file);

  // set the lattice volume
  vol = double(n_sites);

  // create a single orbit
  CreateOrbit(0.0, 0.0, 0.0);

  // initialize the first cell vertex coordinates
  for (int i = 0; i < 4; i++) {
    int s = cells[0].sites[i];
    first_cell_r[i] = r[s];
  }

  cell_orbit.resize(n_faces);
  for (int c = 0; c < n_cells; c++) cell_orbit[c] = 0;
}

/// @brief Change the number of sites.
/// @param n_sites New number of sites
void QfeLatticeS3::ResizeSites(int n_sites) {
  QfeLattice::ResizeSites(n_sites);
  r.resize(n_sites);
  antipode.resize(n_sites, 0);
  site_orbit.resize(n_sites);
}

/// @brief Write a site to a lattice file
/// @param file Lattice file
/// @param s Site index
void QfeLatticeS3::WriteSite(FILE* file, int s) {
  QfeLattice::WriteSite(file, s);
  double xi = acos(r[s].w());
  double theta = 0.0;
  if (xi != 0.0) {
    double cos_theta = r[s].z() / sin(xi);
    if (cos_theta < -1.0) cos_theta = -1.0;
    if (cos_theta > 1.0) cos_theta = 1.0;
    theta = acos(cos_theta);
  }
  double phi = atan2(r[s].y(), r[s].x());
  fprintf(file, " %+.20f %+.20f %+.20f", xi, theta, phi);
}

/// @brief Read a site from a lattice file
/// @param file Lattice file
/// @param s Site index
void QfeLatticeS3::ReadSite(FILE* file, int s) {
  QfeLattice::ReadSite(file, s);
  double xi, theta, phi;
  fscanf(file, " %lf %lf %lf", &xi, &theta, &phi);

  r[s][0] = sin(xi) * sin(theta) * cos(phi);
  r[s][1] = sin(xi) * sin(theta) * sin(phi);
  r[s][2] = sin(xi) * cos(theta);
  r[s][3] = cos(xi);
  r[s].normalize();

  Vec4 north_pole(0.0, 0.0, 0.0, 1.0);
  Vec4 south_pole(0.0, 0.0, 0.0, -1.0);
  if (AlmostEq(r[s], north_pole)) r[s] = north_pole;
  if (AlmostEq(r[s], south_pole)) r[s] = south_pole;
}

/// @brief Create an orbit
/// @param xi1 1st barycentric coordinate
/// @param xi2 2nd barycentric coordinate
/// @param xi3 3rd barycentric coordinate
int QfeLatticeS3::CreateOrbit(double xi1, double xi2, double xi3) {
  // barycentric coordinates, sorted to account for degeneracies
  double xi[4] = {xi1, xi2, xi3, 1.0 - xi1 - xi2 - xi3};
  std::sort(xi, xi + 4, std::greater<double>());
  int o = orbit_xi.size();
  orbit_xi.push_back(Vec4(xi));
  return o;
}

/// @brief Calculate the coordinates of the first site in an orbit
/// @param o Orbit index
/// @return Normalized orbit coordinates
Vec4 QfeLatticeS3::CalcOrbitPos(int o) {
  Vec4 r = Vec4::Zero();
  Vec4 xi = orbit_xi[o];
  for (int i = 0; i < 4; i++) {
    r += xi(i) * first_cell_r[i];
  }
  return r.normalized();
}

/// @brief Read an orbit file and convert to site coordinates
/// @param file Orbit file
void QfeLatticeS3::ReadOrbits(FILE* file) {
  // read orbit data
  orbit_xi.resize(n_distinct);
  for (int o = 0; o < n_distinct; o++) {
    // read barycentric coordinates
    int o_check;
    fscanf(file, "%d", &o_check);
    assert(o_check == o);

    // read dof values
    double xi_sum = 0.0;
    for (int i = 0; i < 3; i++) {
      double temp;
      fscanf(file, "%lf", &temp);
      xi_sum += temp;
      orbit_xi[o][i] = temp;
    }
    orbit_xi[o][3] = 1.0 - xi_sum;
    fscanf(file, "\n");
  }

  UpdateOrbits();
}

/// @brief Write orbit barycentric coordinates to a file that can be read
/// via ReadOrbits
/// @param file Orbit file
void QfeLatticeS3::WriteOrbits(FILE* file) {
  // read orbit data
  for (int o = 0; o < n_distinct; o++) {
    // read barycentric coordinates
    fprintf(file, "%d", o);

    // read dof values
    for (int i = 0; i < 3; i++) {
      fprintf(file, " %.16f", orbit_xi[o][i]);
    }
    fprintf(file, "\n");
  }
}

/// @brief Update positions of all sites using orbits and symmetry group data
void QfeLatticeS3::UpdateOrbits() {
  // calculate the orbit positions
  std::vector<Vec4> orbit_r(n_distinct);
  for (int id = 0; id < n_distinct; id++) {
    orbit_r[id] = CalcOrbitPos(id);
  }

  // use the symmetry group data to calculate site coordinates
  Vec4 north_pole(0.0, 0.0, 0.0, 1.0);
  Vec4 south_pole(0.0, 0.0, 0.0, -1.0);
  for (int s = 0; s < n_sites; s++) {
    int o = site_orbit[s];
    int g = site_g[s];
    r[s] = G[g] * orbit_r[o];
    r[s].normalize();
    if (AlmostEq(r[s], north_pole)) r[s] = north_pole;
    if (AlmostEq(r[s], south_pole)) r[s] = south_pole;
  }
}

/// @brief Read symmetry group data from pre-generated files in the grp
/// directory
/// @param q polytope parameter
/// @param k refinement level
void QfeLatticeS3::ReadSymmetryData(int q, int k) {
  // open the symmetry group data file
  std::string grp_path = string_format("%s/elem/o4q%d.dat", GRP_DIR, q);
  FILE* grp_file = fopen(grp_path.c_str(), "r");
  assert(grp_file != nullptr);

  // read group elements
  G.clear();
  while (!feof(grp_file)) {
    GrpElemO4 g;
    g.ReadGrpElem(grp_file);
    G.push_back(g);
  }
  fclose(grp_file);

  // calculate all of the orbit positions
  std::vector<Vec4> orbit_r(n_distinct);
  for (int id = 0; id < n_distinct; id++) {
    orbit_r[id] = CalcOrbitPos(id);
  }

  // open the site group element file
  std::string g_path = string_format("%s/site_g/o4q%dk%d.dat", GRP_DIR, q, k);
  FILE* g_file = fopen(g_path.c_str(), "r");
  bool site_g_success = true;
  if (g_file != nullptr) {
    // load pre-existing symmetry data
    site_g.resize(n_sites);
    std::vector<int>::iterator it = site_g.begin();
    while (!feof(g_file)) {
      assert(it != site_g.end());
      int g;
      fscanf(g_file, "%d\n", &g);
      *it++ = g;
    }
    fclose(g_file);

    // recalculate if the file was not long enough
    if (it != site_g.end()) {
      fprintf(stderr, "Rebuilding invalid data file: %s\n", g_path.c_str());
      site_g_success = false;
    } else {
      for (int s = 0; s < n_sites; s++) {
        int o = site_orbit[s];
        int g = site_g[s];
        Vec4 r_norm = r[s].normalized();
        Vec4 gr = G[g] * orbit_r[o];
        if (!AlmostEq(r_norm, gr, 1.0e-15)) {
          site_g_success = false;
          break;
        }
      }

      if (!site_g_success) {
        fprintf(stderr, "Rebuilding invalid data file: %s\n", g_path.c_str());
      }
    }

  } else {
    site_g_success = false;
  }

  if (!site_g_success) {
    // find the group element for each site
    site_g.resize(n_sites);
    for (int s = 0; s < n_sites; s++) {
      int o = site_orbit[s];
      Vec4 r_norm = r[s].normalized();

      // find the appropriate group element
      bool found_g = false;
      for (int g = 0; g < G.size(); g++) {
        Vec4 gr = G[g] * orbit_r[o];
        if (!AlmostEq(r_norm, gr, 1.0e-15)) continue;
        site_g[s] = g;
        found_g = true;
        break;
      }
      assert(found_g);
    }

    // write the site group elements to a file
    g_file = fopen(g_path.c_str(), "w");
    for (int s = 0; s < n_sites; s++) {
      fprintf(g_file, "%d\n", site_g[s]);
    }
    fclose(g_file);
  }

  UpdateOrbits();
}

/// @brief Project all site coordinates onto a unit sphere.
void QfeLatticeS3::Inflate() {
  for (int s = 0; s < n_sites; s++) {
    r[s].normalize();
  }
}

/// @brief Identify each site's antipode, i.e. for a site with position r, find
/// the site which has position -r. A lattice with a 5-cell base (q = 3) does
/// not have an antipode for every site.
void QfeLatticeS3::UpdateAntipodes() {
  std::unordered_map<std::string, int> antipode_map;
  for (int s = 0; s < n_sites; s++) {
    // find antipode
    int x_int = int(round(r[s].x() * 1.0e6));
    int y_int = int(round(r[s].y() * 1.0e6));
    int z_int = int(round(r[s].z() * 1.0e6));
    int w_int = int(round(r[s].w() * 1.0e6));
    std::string key =
        string_format("%+d,%+d,%+d,%+d", x_int, y_int, z_int, w_int);
    std::string anti_key =
        string_format("%+d,%+d,%+d,%+d", -x_int, -y_int, -z_int, -w_int);

    if (antipode_map.find(anti_key) != antipode_map.end()) {
      // antipode found in map
      int a = antipode_map[anti_key];
      antipode[s] = a;
      antipode[a] = s;
      antipode_map.erase(anti_key);
    } else {
      // antipode not found yet
      antipode_map[key] = s;
    }
  }

  if (antipode_map.size()) {
    // print error message if there are any unpaired sites
    fprintf(stderr, "no antipode found for %lu/%d sites\n", antipode_map.size(),
            n_sites);
  }
}

/// @brief Calculate finite element weights for each site, link, and cell via
/// the discrete exterior calculus formulation
void QfeLatticeS3::CalcFEMWeights() {
  // set site and link weights to zero
  for (int s = 0; s < n_sites; s++) sites[s].wt = 0.0;
  for (int l = 0; l < n_links; l++) links[l].wt = 0.0;

  // compute the DEC laplacian
  double cell_vol = 0.0;
  for (int c = 0; c < n_cells; c++) {
    cells[c].wt = CellVolume(c);
    cell_vol += cells[c].wt;

    // coordinates of vertices
    Vec4 cell_r[5];
    cell_r[0] = Vec4::Zero();  // distance from origin to each vertex is 1
    cell_r[1] = r[cells[c].sites[0]];
    cell_r[2] = r[cells[c].sites[1]];
    cell_r[3] = r[cells[c].sites[2]];
    cell_r[4] = r[cells[c].sites[3]];

    // generate the Cayley-Menger matrix
    Eigen::Matrix<double, 5, 5> CM;
    for (int i = 0; i < 5; i++) {
      CM(i, i) = 0.0;
      for (int j = i + 1; j < 5; j++) {
        CM(i, j) = (cell_r[i] - cell_r[j]).squaredNorm();
        CM(j, i) = CM(i, j);
      }
    }

    Eigen::Vector<double, 5> cell_lhs(1.0, 0.0, 0.0, 0.0, 0.0);
    Eigen::Vector<double, 5> cell_xi = CM.inverse() * cell_lhs;
    double cell_cr_sq = -cell_xi(0) / 2.0;

    // i and j are 1-indexed in the Cayley-Menger matrix
    for (int i = 1; i <= 4; i++) {
      for (int j = i + 1; j <= 4; j++) {
        // find the other two corners
        int k = 1;
        while (k == i || k == j) k++;
        int l = k + 1;
        while (l == i || l == j) l++;

        double x_ijk =
            2.0 * (CM(i, j) * CM(i, k) + CM(i, j) * CM(j, k) +
                   CM(i, k) * CM(j, k)) -
            (CM(i, j) * CM(i, j) + CM(i, k) * CM(i, k) + CM(j, k) * CM(j, k));
        double x_ijl =
            2.0 * (CM(i, j) * CM(i, l) + CM(i, j) * CM(j, l) +
                   CM(i, l) * CM(j, l)) -
            (CM(i, j) * CM(i, j) + CM(i, l) * CM(i, l) + CM(j, l) * CM(j, l));

        double A_tri_ijk = 0.25 * sqrt(x_ijk);
        double A_tri_ijl = 0.25 * sqrt(x_ijl);
        double dual_ijk = CM(i, k) + CM(j, k) - CM(i, j);
        double dual_ijl = CM(i, l) + CM(j, l) - CM(i, j);

        double h_ijk =
            sqrt(cell_cr_sq - CM(i, j) * CM(i, k) * CM(j, k) / x_ijk);
        double h_ijl =
            sqrt(cell_cr_sq - CM(i, j) * CM(i, l) * CM(j, l) / x_ijl);

        // sign factor
        if (cell_xi(l) < 0.0) h_ijk *= -1.0;
        if (cell_xi(k) < 0.0) h_ijl *= -1.0;

        double wt_ijk = dual_ijk * h_ijk / A_tri_ijk;
        double wt_ijl = dual_ijl * h_ijl / A_tri_ijl;
        double wt = (wt_ijk + wt_ijl) / 16.0;
        int s_i = cells[c].sites[i - 1];
        int s_j = cells[c].sites[j - 1];
        int l_ij = FindLink(s_i, s_j);

        // set FEM weights
        links[l_ij].wt += wt;
        sites[s_i].wt += wt * CM(i, j) / 6.0;
        sites[s_j].wt += wt * CM(i, j) / 6.0;
      }
    }
  }

  double site_vol = 0.0;
  for (int s = 0; s < n_sites; s++) site_vol += sites[s].wt;

  // normalize site volume
  double site_norm = site_vol / vol;
  for (int s = 0; s < n_sites; s++) sites[s].wt /= site_norm;

  // normalize link weights
  double link_norm = cbrt(site_norm);
  for (int l = 0; l < n_links; l++) links[l].wt /= link_norm;

  // normalize cell volume
  double cell_norm = cell_vol / double(n_cells);
  for (int c = 0; c < n_cells; c++) cells[c].wt /= cell_norm;
}

/// @brief Calculate the global effective lattice spacing
/// @return Lattice spacing a/r
double QfeLatticeS3::CalcLatticeSpacing() {
  double vol_sum = 0.0;
  for (int c = 0; c < n_cells; c++) {
    vol_sum += CellVolume(c);
  }
  double vol_mean = vol_sum / double(n_sites);
  return cbrt(vol_mean);
}

/// @brief Calculate a hyperspherical harmonic value (not pre-calculated)
/// @param s Site index
/// @param j Hyperspherical harmonic eigenvalue
/// @param l Hyperspherical harmonic eigenvalue
/// @param m Hyperspherical harmonic eigenvalue
/// @return Hyperspherical harmonic evaluated at site @p s
Complex QfeLatticeS3::GetYjlm(int s, int j, int l, int m) {
  double cos_xi = r[s].w();
  double xi = acos(cos_xi);
  double rho = sin(xi);
  double theta = 0.0;
  if (xi != 0.0) {
    double cos_theta = r[s].z() / rho;
    if (cos_theta < -1.0) cos_theta = -1.0;
    if (cos_theta > 1.0) cos_theta = 1.0;
    theta = acos(cos_theta);
  }
  double phi = atan2(r[s].y(), r[s].x());

  double l_real = double(l);
  double j_real = double(j);
  double c0 = pow(2.0, l_real);
  double c1 = tgamma(l_real + 1.0);
  double c2 = tgamma(j_real - l_real + 1.0);
  double c3 = tgamma(j_real + l_real + 2.0);
  double c4 = 0.79788456080286535588L;  // sqrt(2/pi)
  double c5 = j_real + 1.0;
  double C = c0 * c4 * c1 * sqrt(c5 * c2 / c3);
  double S = pow(rho, l_real);
  double G = boost::math::gegenbauer(j - l, l_real + 1.0, cos_xi);
  Complex Y = boost::math::spherical_harmonic(l, m, theta, phi);
  return C * S * G * Y;
}

/// @brief Calculate the cosine of the angle between two sites. This function
/// assumes that the coordinates have been projected onto the unit sphere.
/// @param s1 1st site index
/// @param s2 2nd site index
/// @return cosine of the angle between @p s1 and @p s2
double QfeLatticeS3::CosTheta(int s1, int s2) {
  if (s1 == s2) return 1.0;
  // if (antipode[s1] == s2) return -1.0;
  return r[s1].dot(r[s2]);
}

/// @brief Compute the midpoint of an edge
/// @param l Link (edge) index
/// @return Midpoint of link @p l
Vec4 QfeLatticeS3::EdgeCenter(int l) {
  Vec4 r_a = r[links[l].sites[0]];
  Vec4 r_b = r[links[l].sites[1]];

  return 0.5 * (r_a + r_b);
}

/// @brief Compute the circumcenter of a triangular face
/// @param f Face index
/// @return Circumcenter of face @p f
Vec4 QfeLatticeS3::FaceCircumcenter(int f) {
  Vec4 r_a = r[faces[f].sites[0]];
  Vec4 r_b = r[faces[f].sites[1]];
  Vec4 r_c = r[faces[f].sites[2]];

  Vec4 v_ac = r_c - r_a;
  Vec4 v_bc = r_c - r_b;

  double A00 = v_ac.dot(v_ac);
  double A01 = v_ac.dot(v_bc);
  double A11 = v_bc.dot(v_bc);

  Eigen::Matrix2d A;
  A(0, 0) = A00;
  A(0, 1) = A01;
  A(1, 0) = A01;
  A(1, 1) = A11;

  Eigen::Vector2d b;
  b(0) = A00;
  b(1) = A11;

  Eigen::Vector2d x = 0.5 * A.inverse() * b;

  return x(0) * r_a + x(1) * r_b + (1.0 - x.sum()) * r_c;
}

/// @brief Compute the circumcenter of a tetrahedral cell
/// @param c Cell index
/// @return Circumcenter of cell @p c
Vec4 QfeLatticeS3::CellCircumcenter(int c) {
  Vec4 r_a = r[cells[c].sites[0]];
  Vec4 r_b = r[cells[c].sites[1]];
  Vec4 r_c = r[cells[c].sites[2]];
  Vec4 r_d = r[cells[c].sites[3]];

  Vec4 v_ad = r_d - r_a;
  Vec4 v_bd = r_d - r_b;
  Vec4 v_cd = r_d - r_c;

  double A00 = v_ad.dot(v_ad);
  double A01 = v_ad.dot(v_bd);
  double A02 = v_ad.dot(v_cd);
  double A11 = v_bd.dot(v_bd);
  double A12 = v_bd.dot(v_cd);
  double A22 = v_cd.dot(v_cd);

  Eigen::Matrix3d A;
  A(0, 0) = A00;
  A(0, 1) = A01;
  A(0, 2) = A02;
  A(1, 0) = A01;
  A(1, 1) = A11;
  A(1, 2) = A12;
  A(2, 0) = A02;
  A(2, 1) = A12;
  A(2, 2) = A22;

  Eigen::Vector3d b;
  b(0) = A00;
  b(1) = A11;
  b(2) = A22;

  Eigen::Vector3d x = 0.5 * A.inverse() * b;

  return x(0) * r_a + x(1) * r_b + x(2) * r_c + (1.0 - x.sum()) * r_d;
}

/// @brief Compute the length of a link
/// @param l Link index
/// @return Length of link @p l
double QfeLatticeS3::EdgeLength(int l) {
  int s_a = links[l].sites[0];
  int s_b = links[l].sites[1];
  Vec4 dr = r[s_a] - r[s_b];
  return dr.norm();
}

/// @brief Compute the volume of a tetrahedral cell
/// @param c Cell index
/// @return Volume of cell @p c
double QfeLatticeS3::CellVolume(int c) {
  Vec4 r_a = r[cells[c].sites[0]];
  Vec4 r_b = r[cells[c].sites[1]];
  Vec4 r_c = r[cells[c].sites[2]];
  Vec4 r_d = r[cells[c].sites[3]];

  Vec4 v_ad = r_d - r_a;
  Vec4 v_bd = r_d - r_b;
  Vec4 v_cd = r_d - r_c;

  double A00 = v_ad.dot(v_ad);
  double A01 = v_ad.dot(v_bd);
  double A02 = v_ad.dot(v_cd);
  double A11 = v_bd.dot(v_bd);
  double A12 = v_bd.dot(v_cd);
  double A22 = v_cd.dot(v_cd);

  Eigen::Matrix3d A;
  A(0, 0) = A00;
  A(0, 1) = A01;
  A(0, 2) = A02;
  A(1, 0) = A01;
  A(1, 1) = A11;
  A(1, 2) = A12;
  A(2, 0) = A02;
  A(2, 1) = A12;
  A(2, 2) = A22;
  double det = A.determinant();
  assert(det >= 0);

  return sqrt(det) / 6.0;
}

/// @brief Print the cartesian coordinates of the sites. This is helpful for
/// making plots in e.g. Mathematica.
void QfeLatticeS3::PrintCoordinates() {
  printf("{");
  for (int s = 0; s < n_sites; s++) {
    printf("{%.12f,%.12f,%.12f,%.12f}", r[s].w(), r[s].x(), r[s].y(), r[s].z());
    printf("%c\n", s == (n_sites - 1) ? '}' : ',');
  }
}
