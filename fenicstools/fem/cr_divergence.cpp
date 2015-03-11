#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/SubSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/plot/plot.h>

namespace dolfin
{
  // Base case for all divergence computations. 
  // Compute divergence of vector field u.
  void cr_divergence(Function& divu, const Function& u)
  {
    std::shared_ptr<const GenericDofMap>
      CR1_dofmap = u.function_space()->dofmap(),
      DG0_dofmap = divu.function_space()->dofmap();

    // Get u and divu as vectors
    std::shared_ptr<const GenericVector> U = u.vector();
    std::shared_ptr<GenericVector> DIVU = divu.vector();

    // Figure out about the local dofs of DG0 
    std::pair<std::size_t, std::size_t>
    first_last_dof = DG0_dofmap->ownership_range();
    std::size_t first_dof = first_last_dof.first;
    std::size_t last_dof = first_last_dof.second;
    std::size_t n_local_dofs = last_dof - first_dof;

    // Make room for local values of U
    std::vector<double> DIVU_values(DIVU->local_size());

    // Get topological dimension so that we know what Facet is
    const Mesh mesh = *u.function_space()->mesh();
    std::size_t tdim = mesh.topology().dim(); 
    // Get the info on length of u and gdim for the dot product
    std::size_t gdim = mesh.geometry().dim();
    std::size_t udim = u.value_dimension(0); // Rank 1!
    std::size_t i_max = std::min(gdim, udim);

    // Fill the values
    for(CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const ArrayView<const dolfin::la_index>
        cell_dofs = DG0_dofmap->cell_dofs(cell->index());
      // There is only one DG0 dof per cell
      dolfin::la_index cell_dof = cell_dofs[0];

      if((first_dof <= cell_dof) and (cell_dof < last_dof))
      {
        Point cell_mp = cell->midpoint();
        double cell_volume = cell->volume();
        
        // Dofs of CR on all facets of the cell, global order
        const ArrayView<const dolfin::la_index>
          facets_dofs = CR1_dofmap->cell_dofs(cell->index());
        
        double cell_integral = 0;
        std::size_t local_facet_index = 0;
        for(FacetIterator facet(*cell); !facet.end(); ++facet)
        {
          double facet_measure=0;
          if(tdim == 2)
            facet_measure = Edge(mesh, facet->index()).length();
          else if(tdim == 3)
            facet_measure = Face(mesh, facet->index()).area();
          // Tdim 1 will not happen because CR is not defined there
          
          Point facet_normal = facet->normal();

          // Flip the normal if it is not outer already
          Point facet_mp = facet->midpoint();
          int sign = (facet_normal.dot(facet_mp - cell_mp) > 0) ? 1 : -1;
          facet_normal *= sign;

          // Dofs of CR on the facet, local order
          std::vector<std::size_t> facet_dofs;
          CR1_dofmap->tabulate_facet_dofs(facet_dofs, local_facet_index);

          // Do the dot product u_i*n_i*meas(facet)
          double facet_integral = 0;
          for(std::size_t i = 0; i < i_max; i++)
            facet_integral += (*U)[facets_dofs[facet_dofs[i]]]*facet_normal[i];
          facet_integral *= facet_measure;

          cell_integral += facet_integral;
          local_facet_index += 1;
        }
        cell_integral /= cell_volume;
        DIVU_values[cell_dof - first_dof] = cell_integral;
      }
    }
    DIVU->set_local(DIVU_values);
    DIVU->apply("insert");
  }
  
  //---------------------------------------------------------------------------

  // Compute divergence of scalar/vector/tensor. Uses vector divergence with
  // specialized vectors.
  void cr_divergence(Function& divu, const Function& u,
                  const FunctionSpace& DGscalar, const FunctionSpace& CRvector)
  {
    // Divu scalar from u vector 
    std::size_t u_rank = u.value_rank();
    std::size_t divu_rank = divu.value_rank();
    if((divu_rank == 0) and (u_rank == 1))
    {
      cr_divergence(divu, u);
    }
    else if(divu_rank == 1)
    {
      // Divu is a vector with components divu_i in space DGscalar
      Function divu_i(DGscalar);
      std::shared_ptr<GenericVector> DIVU_i = divu_i.vector();

      // The components divu_i are created from divergence of special
      // vector in u_i which is space CRvector
      Function u_i(CRvector);
      std::shared_ptr<GenericVector> U_i = u_i.vector();

      // Get vectors of divu and u
      std::shared_ptr<const GenericVector> U = u.vector();
      std::shared_ptr<GenericVector> DIVU = divu.vector();

      // Get dofmaps 
      std::shared_ptr<const FunctionSpace> DGvector = divu.function_space();
      std::shared_ptr<const FunctionSpace> CRtensor = u.function_space();
      
      std::size_t len_divu = divu.value_dimension(0);
      // With scalar U can be extracted only once
      std::vector<double> U_values;
      if(u_rank == 0)
      {
        U->get_local(U_values);
      }

      for(std::size_t i = 0; i < len_divu; i++)
      {
        // Build U_i
        // U_i looks different for u scalar and u tensor
        // For scalar it is U_i = U*e_i + 0, i.e. U_i = (U, 0, 0), (0, U, 0) etc
        if(u_rank == 0)
        {
          // Local dofs of CRvector component
          std::vector<dolfin::la_index>
            CRi_rows = CRvector[i]->dofmap()->dofs();
          std::size_t m = CRi_rows.size();
          U_i->zero();
          U_i->set(U_values.data(), m, CRi_rows.data()); 
          U_i->apply("insert");
        }
        // For tensor, U_i represents T_ij, (T_i0, T_i1, T_i2)
        else if(u_rank == 2)
        {
          // Build the vector U_i component by component
          for(std::size_t j = 0; j < len_divu; j++)
          {
            // Local dofs for ith row's jth component
            std::vector<dolfin::la_index>
              CRij_rows = (*CRtensor)[i*len_divu + j]->dofmap()->dofs();
            U->gather(U_values, CRij_rows);

            std::size_t m = CRij_rows.size();

            // Local dofs of CRvector component
            std::vector<dolfin::la_index>
              CRj_rows = CRvector[j]->dofmap()->dofs();

            U_i->set(U_values.data(), m, CRj_rows.data()); 
            U_i->apply("insert");
          }
        }

        // Compute the component of divergence from u_i
        cr_divergence(divu_i, u_i);

        // Assemble divu_i into divu
        std::vector<dolfin::la_index>
          DGi_rows = (*DGvector)[i]->dofmap()->dofs();
        std::size_t m = DGi_rows.size();
        std::vector<double> DIVU_i_values;
        DIVU_i->get_local(DIVU_i_values);
        DIVU->set(DIVU_i_values.data(), m, DGi_rows.data()); 
        DIVU->apply("insert");
      }
    }
  }
  
  std::shared_ptr<GenericMatrix> MatMatMult(GenericMatrix& A, GenericMatrix& B)
  {
    const dolfin::PETScMatrix* Ap = &as_type<const dolfin::PETScMatrix>(A);
    const dolfin::PETScMatrix* Bp = &as_type<const dolfin::PETScMatrix>(B);
    Mat CC;
    PetscErrorCode ierr = MatMatMult(Ap->mat(), Bp->mat(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CC);
    dolfin::PETScMatrix CCC = PETScMatrix(CC);
    return CCC.copy();  
  }
  
  // Base case for all divergence computations. 
  // Compute divergence of vector field u.
  std::shared_ptr<GenericMatrix> 
    cr_divergence_matrix(GenericMatrix& M, GenericMatrix& A,
                         const FunctionSpace& DGscalar, 
                         const FunctionSpace& CRvector)
  {
    std::shared_ptr<const GenericDofMap>
      CR1_dofmap = CRvector.dofmap(),
      DG0_dofmap = DGscalar.dofmap();

    // Figure out about the local dofs of DG0 
    std::pair<std::size_t, std::size_t>
    first_last_dof = DG0_dofmap->ownership_range();
    std::size_t first_dof = first_last_dof.first;
    std::size_t last_dof = first_last_dof.second;
    std::size_t n_local_dofs = last_dof - first_dof;

    // Get topological dimension so that we know what Facet is
    const Mesh mesh = *DGscalar.mesh();
    std::size_t tdim = mesh.topology().dim(); 
    std::size_t gdim = mesh.geometry().dim();
    
    std::vector<std::size_t> columns;
    std::vector<double> values;
    
    // Fill the values
    for(CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const ArrayView<const dolfin::la_index>
        dg_dofs = DG0_dofmap->cell_dofs(cell->index());
      // There is only one DG0 dof per cell
      dolfin::la_index cell_dof = dg_dofs[0];
      
      Point cell_mp = cell->midpoint();
      double cell_volume = cell->volume();
      std::size_t local_facet_index = 0;      
      
      const ArrayView<const dolfin::la_index>
        cr_dofs = CR1_dofmap->cell_dofs(cell->index());
      
      for(FacetIterator facet(*cell); !facet.end(); ++facet)
      {
        double facet_measure=0;
        if(tdim == 2)
          facet_measure = Edge(mesh, facet->index()).length();
        else if(tdim == 3)
          facet_measure = Face(mesh, facet->index()).area();
        // Tdim 1 will not happen because CR is not defined there
        
        Point facet_normal = facet->normal();

        // Flip the normal if it is not outer already
        Point facet_mp = facet->midpoint();
        double sign = (facet_normal.dot(facet_mp - cell_mp) > 0.0) ? 1.0 : -1.0;
        facet_normal *= (sign*facet_measure/cell_volume);
        
        // Dofs of CR on the facet, local order
        std::vector<std::size_t> facet_dofs;
        CR1_dofmap->tabulate_facet_dofs(facet_dofs, local_facet_index);
        
        for (std::size_t j = 0 ; j < facet_dofs.size(); j++)
        {   
          columns.push_back(cr_dofs[facet_dofs[j]]);
          values.push_back(facet_normal[j]);
        }        
        local_facet_index += 1;
      }
      M.setrow(cell_dof, columns, values);
      columns.clear();
      values.clear();
    }
    M.apply("insert");
    std::shared_ptr<GenericMatrix> Cp = MatMatMult(M, A);
    return Cp;
  }
} 
