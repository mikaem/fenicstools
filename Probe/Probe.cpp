#include "Probe.h"

using namespace dolfin;

Probe::~Probe() 
{
  clear(); 
  delete dolfin_cell; 
  delete ufc_cell; 
}

Probe::Probe(const Array<double>& x, const FunctionSpace& V) :
   _element(V.element()), _num_evals(0)
{

  const Mesh& mesh = *V.mesh();
  std::size_t gdim = mesh.geometry().dim();
    
  // Find the cell that contains probe
  const Point point(gdim, x.data());
  int id = mesh.intersected_cell(point);
  
  // If the cell is on this process, then create an instance 
  // of the Probe class. Otherwise raise a dolfin_error.
  if (id != -1)
  {
    // Store position of probe
    for (std::size_t i = 0; i < 3; i++) 
      _x[i] = (i < gdim ? x[i] : 0.0);
    
    // Compute in tensor (one for scalar function, . . .)
    _value_size_loc = 1;
    for (std::size_t i = 0; i < _element->value_rank(); i++)
       _value_size_loc *= _element->value_dimension(i);

    _probes.resize(_value_size_loc);

    // Create cell that contains point
    dolfin_cell = new Cell(mesh, id);
    ufc_cell = new UFCCell(*dolfin_cell);

    // Create work vector for basis
    std::vector<double> basis(_value_size_loc);
    
    coefficients.resize(_element->space_dimension());
        
    // Create work vector for basis
    basis_matrix.resize(_value_size_loc);
    for (std::size_t i = 0; i < _value_size_loc; ++i)
      basis_matrix[i].resize(_element->space_dimension());
        
    for (std::size_t i = 0; i < _element->space_dimension(); ++i)
    {
      _element->evaluate_basis(i, &basis[0], &x[0], *ufc_cell);
      for (std::size_t j = 0; j < _value_size_loc; ++j)
        basis_matrix[j][i] = basis[j];
    }
  }
  else
  {
    dolfin_error("Probe.cpp","set probe","Probe is not found on processor");
  }
}
//
void Probe::eval(const Function& u)
{
  // Restrict function to cell
  u.restrict(&coefficients[0], *_element, *dolfin_cell, *ufc_cell);

  // Make room for one more evaluation
  for (std::size_t j = 0; j < _value_size_loc; j++)
    _probes[j].push_back(0.);
  
  // Compute linear combination
  for (std::size_t i = 0; i < _element->space_dimension(); i++)
  {
    for (std::size_t j = 0; j < _value_size_loc; j++)
      _probes[j][_num_evals] += coefficients[i]*basis_matrix[j][i];
  }
  
  _num_evals++;
}
// Remove one instance of the probe
void Probe::erase_snapshot(std::size_t i)
{
  for (std::size_t j = 0; j < _value_size_loc; j++)
    _probes[j].erase(_probes[j].begin()+i);  
  
  _num_evals--;
}
// Reset probe by removing all values
void Probe::clear()
{
  for (std::size_t j = 0; j < _value_size_loc; j++)
    _probes[j].clear();  
  
  _num_evals = 0;
}
// Return probe values for chosen component of tensor
std::vector<double> Probe::get_probe_sub(std::size_t i)
{
  return _probes[i];
}
// Return probe values for entire tensor at snapshot
std::vector<double> Probe::get_probe_at_snapshot(std::size_t i)
{
  std::vector<double> x(_value_size_loc);
  for (std::size_t j = 0; j < _value_size_loc; j++)
    x[j] = _probes[j][i];
  return x;
}
// Return coordinates of probe
std::vector<double> Probe::coordinates()
{
  std::vector<double> x(3);
  x.assign(_x, _x+3);
  return x;
}
//
void Probe::dump(std::size_t i, std::string filename, std::size_t id)
{
  std::ofstream fp(filename.c_str(), std::ios_base::app);
  std::ostringstream ss;
  ss << std::scientific;
  ss << "Probe id = " << id << std::endl;
  ss << "Number of evaluations = " << number_of_evaluations() << std::endl;
  ss << "Coordinates:" << std::endl;
  ss << _x[0] << " " << _x[1] << " " << _x[2] << std::endl; 
  ss << "Values for component " << i << std::endl;
  for (std::size_t j = 0; j < _probes[i].size(); j++)
      ss << _probes[i][j] << std::endl;
  ss << std::endl;
  cout << ss.str() << endl;
  fp << ss.str();
}
void Probe::dump(std::string filename, std::size_t id)
{
  std::ofstream fp(filename.c_str(), std::ios_base::app);
  std::ostringstream ss;
  ss << std::scientific;
  ss << "Probe id = " << id << std::endl;
  ss << "Number of evaluations = " << number_of_evaluations() << std::endl;
  ss << "Coordinates:" << std::endl;
  ss << _x[0] << " " << _x[1] << " " << _x[2] << std::endl; 
  ss << "Values for all components:" << std::endl;    
  std::size_t N = _probes[0].size();
  for (std::size_t j = 0; j < N; j++)
  {
    for (std::size_t i = 0; i < _value_size_loc; i++)
      ss << _probes[i][j] << " " ;
    ss << std::endl;
  }
  cout << ss.str() << endl;
  fp << ss.str();
}
//
void Probe::restart_probe(const Array<double>& u)
{
  // Make room for one more evaluation
  for (std::size_t j = 0; j < _value_size_loc; j++)
    _probes[j].push_back(u[j]);
}

