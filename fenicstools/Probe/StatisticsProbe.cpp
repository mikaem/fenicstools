#include "StatisticsProbe.h"

using namespace dolfin;

StatisticsProbe::StatisticsProbe(const Array<double>& x, const FunctionSpace& V) 
  : Probe(x, V), segregated(false)
{
  value_size_loc_function = _value_size_loc;
    
  // Symmetric statistics. Velocity: u, v, w, uu, vv, ww, uv, uw, vw
  _value_size_loc = _value_size_loc*(_value_size_loc+3)/2.;
                 
  _probes.resize(_value_size_loc);
    
  // Make room for exactly two values. The mean and the latest snapshot  
  for (std::size_t i = 0; i < 2; i++)
  {
    for (std::size_t j = 0; j < _value_size_loc; j++)
      _probes[j].push_back(0.);
  }
}


StatisticsProbe::StatisticsProbe(const Array<double>& x, const FunctionSpace& V, bool segregated) 
  : Probe(x, V), segregated(segregated)
{
  value_size_loc_function = _value_size_loc;
  
  if (segregated)
  {
    assert(_element->value_rank() == 0);
    _value_size_loc *= _element->geometric_dimension();
  }
    
  // Symmetric statistics. Velocity: u, v, w, uu, vv, ww, uv, uw, vw
  _value_size_loc = _value_size_loc*(_value_size_loc+3)/2.;
                 
  _probes.resize(_value_size_loc);
    
  // Make room for exactly two values. The mean and the latest snapshot  
  for (std::size_t i = 0; i < 2; i++)
  {
    for (std::size_t j = 0; j < _value_size_loc; j++)
      _probes[j].push_back(0.);
  }
}

StatisticsProbe::StatisticsProbe(const Probe& p) : Probe(p)
{
  StatisticsProbe *ps = (StatisticsProbe *) &p;
  segregated = ps->segregated;
  value_size_loc_function = ps->value_size_loc_function;
}

void StatisticsProbe::eval(const Function& u)
{
  assert(u.value_size() == value_size_loc_function);
  
  // Restrict function to cell
  u.restrict(&coefficients[0], *_element, *dolfin_cell, 
             vertex_coordinates.data(), ufc_cell);
  
  std::vector<double> tmp(value_size_loc_function);
  // Compute linear combination
  for (std::size_t i = 0; i < _element->space_dimension(); i++)
  {
    for (std::size_t j = 0; j < value_size_loc_function; j++)
      tmp[j] += coefficients[i]*basis_matrix[j][i];    
  }  
  for (std::size_t j = 0; j < value_size_loc_function; j++)
  {
     _probes[j][0] += tmp[j];           // u, v, w
     _probes[j][1]  = tmp[j];
  }
  for (std::size_t j = 0; j < value_size_loc_function; j++)
  {
    for (std::size_t k = j; k < value_size_loc_function; k++)
    {
       if (j == k)
         _probes[value_size_loc_function+j][0] += tmp[j]*tmp[k];
       else // covariance 
         _probes[2*value_size_loc_function+k+j-1][0] += tmp[j]*tmp[k];
    }
  }  
  _num_evals++; 
}
// Special for segregated solvers in 3D
void StatisticsProbe::eval(const Function& u, const Function& v, const Function& w)
{
  assert(u.value_size() == 1);
  assert(v.value_size() == 1);
  assert(w.value_size() == 1);
    
  std::vector<double> tmp(3);
  // Restrict function to cell
  u.restrict(&coefficients[0], *_element, *dolfin_cell, 
             vertex_coordinates.data(), ufc_cell);
  // Compute linear combination
  for (std::size_t i = 0; i < _element->space_dimension(); i++)
    tmp[0] += coefficients[i]*basis_matrix[0][i];    
  v.restrict(&coefficients[0], *_element, *dolfin_cell,
             vertex_coordinates.data(), ufc_cell);
  for (std::size_t i = 0; i < _element->space_dimension(); i++)
    tmp[1] += coefficients[i]*basis_matrix[0][i];    
  w.restrict(&coefficients[0], *_element, *dolfin_cell, 
             vertex_coordinates.data(), ufc_cell);
  for (std::size_t i = 0; i < _element->space_dimension(); i++)
    tmp[2] += coefficients[i]*basis_matrix[0][i];    

  for (std::size_t j = 0; j < 3; j++)
  {
     _probes[j][0] += tmp[j];           // u, v, w
     _probes[j][1]  = tmp[j];           
     _probes[j+3][0] += tmp[j]*tmp[j];  // uu, vv, ww
  }
  _probes[6][0] += tmp[0]*tmp[1];      // uv
  _probes[7][0] += tmp[0]*tmp[2];      // uw
  _probes[8][0] += tmp[1]*tmp[2];      // vw
  
  _num_evals++; 
}

// Special for segregated velocity in 2D
void StatisticsProbe::eval(const Function& u, const Function& v)
{
  assert(u.value_size() == 1);
  assert(v.value_size() == 1);
    
  std::vector<double> tmp(2);
  // Restrict function to cell
  u.restrict(&coefficients[0], *_element, *dolfin_cell, 
             vertex_coordinates.data(), ufc_cell);
  for (std::size_t i = 0; i < _element->space_dimension(); i++)
    tmp[0] += coefficients[i]*basis_matrix[0][i];    
  v.restrict(&coefficients[0], *_element, *dolfin_cell,
             vertex_coordinates.data(), ufc_cell);
  for (std::size_t i = 0; i < _element->space_dimension(); i++)
    tmp[1] += coefficients[i]*basis_matrix[0][i];    

  for (std::size_t j = 0; j < 2; j++)
  {
     _probes[j][0] += tmp[j];           // u, v
     _probes[j][1]  = tmp[j];
     _probes[j+2][0] += tmp[j]*tmp[j];  // uu, vv
  }
  _probes[4][0] += tmp[0]*tmp[1];       // uv
  
  _num_evals++; 
}
// Return mean of the function being probed
std::vector<double> StatisticsProbe::mean()
{
  std::size_t N = value_size_loc_function;
  if (segregated)
    N *= _element->geometric_dimension();
  std::vector<double> x(N);
  for (std::size_t j = 0; j < N; j++)
    x[j] = _probes[j][0] / _num_evals;
  return x;
}
// Return the covariance of the functions components
std::vector<double> StatisticsProbe::variance()
{
  std::size_t N = value_size_loc_function;
  if (segregated)
    N *= _element->geometric_dimension();
  std::vector<double> x(_value_size_loc-N);
  for (std::size_t j = N; j < _value_size_loc; j++)
    x[j-N] = _probes[j][0] / _num_evals;  
  return x;
}
void StatisticsProbe::clear()
{
  for (std::size_t j = 0; j < _value_size_loc; j++)
    _probes[j][0] = 0.;
  _num_evals = 0;
}
// Reset probe values for entire tensor
void StatisticsProbe::restart_probe(const Array<double>& u, std::size_t num_evals)
{
  for (std::size_t j = 0; j < _value_size_loc; j++)
    _probes[j][0] = u[j]*num_evals;
  
  _num_evals = num_evals;
}
