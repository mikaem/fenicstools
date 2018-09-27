#ifndef __STATISTICSPROBE_H
#define __STATISTICSPROBE_H

#include "Probe.h"

namespace dolfin
{

  class StatisticsProbe : public Probe
  {
  // Create a probe that computes mean and variance of the Function in a point
  public:

    StatisticsProbe(const Array<double>& x, const FunctionSpace& V);

    StatisticsProbe(const Array<double>& x, const FunctionSpace& V, bool segregated=false);

    StatisticsProbe(const Probe& p);

    // For segregated velocity components
    void eval(const Function& u);
    void eval(const Function& u, const Function& v); // 2D
    void eval(const Function& u, const Function& v, const Function& w); // 3D

    void erase_snapshot(std::size_t i) {cout << "Cannot erase snapshot for StatisticsProbe" << endl;};

    // Reset probe by deleting all previous evaluations
    void clear();

    // Return mean of Function probed for
    std::vector<double> mean();

    // Return the covariance of all components in Function
    std::vector<double> variance();

    // Reset probe values for entire tensor
    void restart_probe(const Array<double>& u, std::size_t num_evals);

  protected:

    // value_size of the Function being probed
    std::size_t value_size_loc_function;

    // True for segregated solvers
    std::size_t segregated;

  };
}

#endif
