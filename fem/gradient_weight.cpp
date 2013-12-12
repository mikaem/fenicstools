
#include "dolfin.h"

namespace dolfin
{
  void set_constant(GenericMatrix& A, double c)
  {
    std::vector<std::size_t> columns;
    std::vector<double> values;
    std::size_t M = A.size(0);
    const std::pair<std::size_t, std::size_t> row_range = A.local_range(0);
    const std::size_t m = row_range.second - row_range.first;
    for (std::size_t row=0; row<m; row++)
    {   
      // Get global row number
      const std::size_t global_row = row + row_range.first;
      
      A.getrow(global_row, columns, values);
      values.assign(values.size(), c);
      A.setrow(global_row, columns, values);
      A.apply("insert");
    }
  }
  void compute_weight(Function& DG)
  {
    // Compute weights for averaging with neighboring cells
      
    // Get the mesh, element and dofmap
    boost::shared_ptr<const FunctionSpace> V = DG.function_space(); 
    boost::shared_ptr<const Mesh> mesh = V->mesh();
    boost::shared_ptr<const FiniteElement> element = V->element();
    boost::shared_ptr<const GenericDofMap> dofmap_u = V->dofmap();
    
    // Allocate storage for weights on one cell
    std::vector<double> ws(element->space_dimension()); 
    
    // Compute weights
    GenericVector& dg_vector = *DG.vector();  
    dg_vector.zero();
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      const std::vector<dolfin::la_index>& dofs
        = dofmap_u->cell_dofs(cell->index());
        
      std::fill(ws.begin(), ws.end(), 1./cell->volume());
      dg_vector.add(ws.data(), dofs.size(), dofs.data());      
    }  
    dg_vector.apply("insert");  
  }
  void compute_DG0_to_CG1_weight_matrix(GenericMatrix& A, Function& DG)
  {
    compute_weight(DG);
    std::vector<std::size_t> columns;
    std::vector<double> values;
    std::size_t M = A.size(0);
    const std::pair<std::size_t, std::size_t> row_range = A.local_range(0);
    const std::size_t m = row_range.second - row_range.first;
    GenericVector& weight = *DG.vector();
    for (std::size_t row=0; row<m; row++)
    {   
      // Get global row number
      const std::size_t global_row = row + row_range.first;
      
      A.getrow(global_row, columns, values);
      for (std::size_t i=0; i<values.size(); i++)
      {
        double w = weight[columns[i]];
        values.at(i) = w;
      }
      double s = std::accumulate(values.begin(), values.end(), 0.0);
      std::transform(values.begin(), values.end(), values.begin(),
                     std::bind2nd(std::multiplies<double>(), 1./s));
      
      for (std::size_t i=0; i<values.size(); i++)
      {
        double w = weight[columns[i]];
        values.at(i) = values[i]*w;
      }

      A.setrow(global_row, columns, values);
      A.apply("insert");
    }
  }  
  void compute_weighted_gradient_matrix(GenericMatrix& A, GenericMatrix& dP, GenericMatrix& C, Function& DG)
  {
    compute_DG0_to_CG1_weight_matrix(A, DG);
    const dolfin::PETScMatrix* Ap = &as_type<const dolfin::PETScMatrix>(A);
    const dolfin::PETScMatrix* Pp = &as_type<const dolfin::PETScMatrix>(dP);
    dolfin::PETScMatrix* Cp = &as_type<dolfin::PETScMatrix>(C);  
    PetscErrorCode ierr = MatMatMult(*Ap->mat(), *Pp->mat(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(*Cp->mat()));
  }  
}        
