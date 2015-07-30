#include <dolfin/la/GenericMatrix.h>
#include <algorithm>

namespace dolfin
{

  void compute_cg1_cr_interpolation_matrix(GenericMatrix& A, std::size_t d)
  {
    std::vector<std::size_t> columns;
    std::vector<double> values;
    std::vector<std::vector<std::size_t> > allcolumns;
    std::vector<std::vector<double> > allvalues;
    
    const std::pair<std::size_t, std::size_t> row_range = A.local_range(0);
    std::size_t n_local_rows = row_range.second - row_range.first;

    // Value goes to non-zero entries of each row. The entries represent
    // CG1 dofs that are located by facet where row-th CR dof is located.
    const double value = 1./d;
    for(std::size_t row = 0; row < n_local_rows; row++)
    {
      // Get global row number
      const std::size_t global_row = row + row_range.first;
      A.getrow(global_row, columns, values);

      // Substitute
      std::replace_if(values.begin(), values.end(),
                      std::bind2nd(std::greater<double>(), DOLFIN_EPS), value);
     
      std::replace_if(values.begin(), values.end(),
                      std::bind2nd(std::less<double>(), value), 0.);

      allcolumns.push_back(columns);
      allvalues.push_back(values);
    }
  
    // Set the matrix
    for (std::size_t row = 0; row < n_local_rows; row++)
    {       
      // Get global row number
      const std::size_t global_row = row + row_range.first;
      
      A.setrow(global_row, allcolumns[row], allvalues[row]);
    }
    A.apply("insert");
  }

}
