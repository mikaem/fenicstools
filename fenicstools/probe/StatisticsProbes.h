#ifndef __STATISTICSPROBES_H
#define __STATISTICSPROBES_H

#include "Probes.h"
#include "StatisticsProbe.h"

namespace dolfin
{

  class StatisticsProbes : public Probes
  {

  public:

    StatisticsProbes(const Array<double>& x, const FunctionSpace& V, bool segregated=false);

    StatisticsProbes(const StatisticsProbes& p);

    ~StatisticsProbes();

    // Return an instance of probe i
    std::shared_ptr<StatisticsProbe> get_statisticsprobe(std::size_t i);

    // For regular and segregated velocity components
    void eval(const Function& u);
    void eval(const Function& u, const Function& v); // 2D segregated
    void eval(const Function& u, const Function& v, const Function& w); // 3D segregated

    // Add new probe positions
    void add_positions(const Array<double>& x, const FunctionSpace& V, bool segregated);

    // No snapshots for statistics, just averages.
    void erase_snapshot(std::size_t i) {cout << "Cannot erase snapshot for StatisticsProbes" << endl;};

    // Reset probe by deleting all previous evaluations
    void clear();

    // Reset probe values for entire tensor
    void restart_probes(const Array<double>& u, std::size_t num_evals);
    void set_probes_from_ids(const Array<double>& u, std::size_t num_evals);

  };
}

#endif
