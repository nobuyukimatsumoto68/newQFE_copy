// ads_strip.h

#pragma once

#include <vector>
#include "lattice.h"

// This class defines an AdS2 "strip" lattice. The strip has periodic
// boundary conditions in the temporal direction and non-periodic boundary
// conditions in the spatial direction (i.e. links do not wrap around).
// This means that the strip has a central axis which is defined to be halfway
// between the leftmost and rightmost lattice sites. Therefore if the spatial
// lattice size is odd, there will be a row of lattice sites down the center,
// but if the spatial lattice size is even there will be rows of lattice
// sites straddling the center.

class QfeLatticeAdSStrip : public QfeLattice {

public:
  QfeLatticeAdSStrip(int n_rho, int n_t, double t_scale = 1.0);
  virtual void ResizeSites(int n_sites);

  double t_scale = 1.0;

  // site coordinates
  std::vector<double> rho;  // geodesic distance from center axis
};

/**
 * @brief Create and AdS2 strip lattice with @p n_rho sites in the
 * spatial direction (non-periodic boundary) and @p n_t sites in
 * the temporal direction (periodic boundary). Site and link weights
 * are assigned according to the metric ds^2 = cosh^2(rho) dt^2 + drho^2.
 */

QfeLatticeAdSStrip::QfeLatticeAdSStrip(int n_rho, int n_t, double t_scale) {

  this->t_scale = t_scale;

  ResizeSites(n_rho * n_t);

  // integer distance from center to edge of strip (number of hops)
  double strip_radius = 0.5 * double(n_rho - 1);

  vol = 0.0;
  for (int s = 0; s < n_sites; s++) {

    // lattice coordinates of this site
    int x = s % n_rho;
    int y = s / n_rho;

    rho[s] = double(x) - strip_radius;
    sites[s].wt = cosh(rho[s]) / t_scale;
    vol += sites[s].wt;

    // lattice coordinates of adjacent sites
    int xm1 = (x - 1 + n_rho) % n_rho;
    int yp1 = (y + 1) % n_t;

    if (x != 0) {
      // connect to site in backward rho direction (non-periodic)
      double wt_rho = 0.5 * (cosh(rho[s - 1]) + cosh(rho[s])) / t_scale;
      AddLink(xm1 + y * n_rho, s, wt_rho);
    }

    // connect to site in forward time direction (periodic)
    double wt_t = t_scale / cosh(rho[s]);
    AddLink(s, x + yp1 * n_rho, wt_t);
  }
}

/**
 * @brief Change the number of sites to @p n_sites.
 */

void QfeLatticeAdSStrip::ResizeSites(int n_sites) {
  QfeLattice::ResizeSites(n_sites);
  rho.resize(n_sites);
}
