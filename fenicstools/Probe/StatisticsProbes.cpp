#include "StatisticsProbes.h"

using namespace dolfin;

StatisticsProbes::StatisticsProbes(const Array<double>& x, const FunctionSpace& V, bool segregated)
{
  const std::size_t Nd = V.mesh()->geometry().dim();
  const std::size_t N = x.size() / Nd;
  Array<double> _x(Nd);
  total_number_probes = N;
  _num_evals = 0;
  _value_size = 1;
  for (std::size_t i = 0; i < V.element()->value_rank(); i++)
    _value_size *= V.element()->value_dimension(i);
  
  if (segregated)
  {
    assert(V.element()->value_rank() == 0);
    _value_size *= V.element()->geometric_dimension();
  }
    
  // Symmetric statistics. Velocity: u, v, w, uu, vv, ww, uv, uw, vw
  _value_size = _value_size*(_value_size+3)/2.;

  for (std::size_t i=0; i<N; i++)
  {
    for (std::size_t j=0; j<Nd; j++)
      _x[j] = x[i*Nd + j];
    try
    {
      StatisticsProbe* probe = new StatisticsProbe(_x, V, segregated);
      std::pair<std::size_t, StatisticsProbe*> newprobe = std::make_pair(i, probe);
      _allprobes.push_back(newprobe);
    } 
    catch (std::exception &e)
    { // do-nothing
    }
  }
  //cout << local_size() << " of " << N  << " probes found on processor " << MPI::process_number() << endl;
}
//
StatisticsProbes::~StatisticsProbes()
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    delete _allprobes[i].second;   
  }
  _allprobes.clear();      
}

//
StatisticsProbes::StatisticsProbes(const StatisticsProbes& p)
{
  _allprobes = p._allprobes;
  total_number_probes = p.total_number_probes;
  _value_size = p._value_size;
  _num_evals = p._num_evals;
  for (std::size_t i = 0; i < local_size(); i++)
  {    
    _allprobes[i].second = new StatisticsProbe(*(p._allprobes[i].second));
  }
}
//
void StatisticsProbes::add_positions(const Array<double>& x, const FunctionSpace& V, bool segregated)
{
  const std::size_t gdim = V.mesh()->geometry().dim();
  const std::size_t N = x.size() / gdim;
  Array<double> _x(gdim);
  const std::size_t old_N = total_number_probes;
  const std::size_t old_local_size = local_size();
  total_number_probes += N;

  for (std::size_t i=0; i<N; i++)
  {
    for (std::size_t j=0; j<gdim; j++)
      _x[j] = x[i*gdim + j];
    try
    {
      StatisticsProbe* probe = new StatisticsProbe(_x, V, segregated);
      std::pair<std::size_t, StatisticsProbe*> newprobe = std::make_pair(old_N+i, &(*probe));
      _allprobes.push_back(newprobe);
    } 
    catch (std::exception &e)
    { // do-nothing
    }
  }
  //cout << local_size() - old_local_size << " of " << N  << " probes found on processor " << MPI::process_number() << endl;
}
//
std::shared_ptr<StatisticsProbe> StatisticsProbes::get_statisticsprobe(std::size_t i)
{
  if (i >= local_size() || i < 0) 
  {
    dolfin_error("StatisticsProbes.cpp", "get probe", "Wrong index!");
  }
  return std::make_shared<StatisticsProbe>(*_allprobes[i].second);
}
//
void StatisticsProbes::eval(const Function& u)
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
    probe->eval(u);
  }
  _num_evals++;
}
void StatisticsProbes::eval(const Function& u, const Function& v)
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
    probe->eval(u, v);
  }
  _num_evals++;    
}
void StatisticsProbes::eval(const Function& u, const Function& v, const Function& w)
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
    probe->eval(u, v, w);
  }
  _num_evals++;
}
void StatisticsProbes::clear()
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
    probe->clear();
  }
  _num_evals = 0;
}
// Reset probe values for entire tensor
void StatisticsProbes::restart_probes(const Array<double>& u, std::size_t num_evals)
{
  Array<double> _u(value_size());  
  for (std::size_t i = 0; i < local_size(); i++)
  {
    std::size_t probe_id = _allprobes[i].first;
    StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
    for (std::size_t j=0; j<value_size(); j++)
      _u[j] = u[probe_id*value_size() + j];
    probe->restart_probe(_u, num_evals);
  }

  _num_evals = num_evals;
}
//
void StatisticsProbes::set_probes_from_ids(const Array<double>& u, std::size_t num_evals)
{
  Array<double> _u(value_size());  
  for (std::size_t i = 0; i < local_size(); i++)
  {
    StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
    for (std::size_t j=0; j<value_size(); j++)
      _u[j] = u[i*value_size() + j];
    probe->restart_probe(_u, num_evals);
  }

  _num_evals = num_evals;
}
