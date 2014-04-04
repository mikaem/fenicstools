#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/SubSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
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
      std::vector<dolfin::la_index>
      cell_dofs = DG0_dofmap->cell_dofs(cell->index());
      // There is only one DG0 dof per cell
      dolfin::la_index cell_dof = cell_dofs[0];

      if((first_dof <= cell_dof) and (cell_dof < last_dof))
      {
        Point cell_mp = cell->midpoint();
        double cell_volume = cell->volume();
        
        // Dofs of CR on all facets of the cell, global order
        std::vector<dolfin::la_index>
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
    if((divu.value_rank() == 0) and (u.value_rank() == 1))
      cr_divergence(divu, u);
    else if(divu.value_rank() == 1)
    {
      // Divu is a vector with components divu_i in DG that will be computed
      Function divu_i(DGscalar);
      std::shared_ptr<GenericVector> DIVU_i = divu_i.vector();

      // The components are created from divergence of special vector in CR
      Function u_i(CRvector);
      std::shared_ptr<GenericVector> U_i = u_i.vector();

      // Get vectors to divu and u
      std::shared_ptr<const GenericVector> U = u.vector();
      std::shared_ptr<GenericVector> DIVU = divu.vector();

      // Get dofmaps of ...
      std::shared_ptr<const FunctionSpace> DGvector = divu.function_space();
      std::shared_ptr<const FunctionSpace> CRtensor = u.function_space();
      
      std::size_t len_divu = divu.value_dimension(0);
      if(u.value_rank() == 0)
      {
        // U stays same, only builds U_I
        std::vector<double> U_values;
        U->get_local(U_values);

        for(std::size_t i = 0; i < len_divu; i++)
        {
          // Build special vector, U ---> U_I
          // Local dofs of CR component
          std::vector<dolfin::la_index> CRi_rows = CRvector[i]->dofmap()->dofs();
          std::size_t CRi_m = CRi_rows.size();
          // Make U_i = U*e_i + 0
          U_i->zero();
          U_i->set(U_values.data(), CRi_m, CRi_rows.data()); 
          U_i->apply("insert");

          // Compute the component of divergence
          u_i.update();
          cr_divergence(divu_i, u_i);
  
          // Set the final divergence using new component, DIVU_I ---> DIVU
          std::vector<dolfin::la_index>
            DGi_rows = (*DGvector)[i]->dofmap()->dofs();
          std::size_t DGi_m = DGi_rows.size();
          std::vector<double> DIVU_i_values;
          DIVU_i->get_local(DIVU_i_values);
          DIVU->set(DIVU_i_values.data(), DGi_m, DGi_rows.data()); 
          DIVU->apply("insert");
        }
      }
      else if(u.value_rank() == 2)
      {
        for(std::size_t i = 0; i < len_divu; i++)
        {
          for(std::size_t j = 0; j < len_divu; j++)
          {
            // Build special vector, U ---> U_I
            // Local dofs of CR component
            //info("len %d, i=%d, j=%d--> %d", len_divu, i, j, i*len_divu + j);
            std::vector<dolfin::la_index>
              CRi_rows = (*CRtensor)[i*len_divu + j]->dofmap()->dofs();
            std::vector<double> U_values;
            U->gather(U_values, CRi_rows);

            std::size_t CRi_m = CRi_rows.size();
            std::vector<dolfin::la_index>
              xxx = CRvector[j]->dofmap()->dofs();
            // TODO, test clean up


            U_i->set(U_values.data(), CRi_m, xxx.data()); 
            U_i->apply("insert");
          }

          // Compute the component of divergence
          u_i.update();
          cr_divergence(divu_i, u_i);
  
          // Set the final divergence using new component, DIVU_I ---> DIVU
          std::vector<dolfin::la_index>
            DGi_rows = (*DGvector)[i]->dofmap()->dofs();
          std::size_t DGi_m = DGi_rows.size();
          std::vector<double> DIVU_i_values;
          DIVU_i->get_local(DIVU_i_values);
          DIVU->set(DIVU_i_values.data(), DGi_m, DGi_rows.data()); 
          DIVU->apply("insert");
        }
      }
    }
  }
} 
