#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>

namespace dolfin
{
  void cr_divergence(Function& divu, const Function& u)
  {
    // Get the dofmaps
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
          // Tdim will not happen because CR is not defined there
          
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
}
