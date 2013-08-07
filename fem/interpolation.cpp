
#include "dolfin.h"

namespace dolfin
{
    
  void extract_dof_component_map(std::map<unsigned int, unsigned int>& dof_component_map, const FunctionSpace& VV, int* component)
  { // Extract sub dofmaps recursively and store dof to component map
    boost::unordered_map<std::size_t, std::size_t> collapsed_map;
    boost::unordered_map<std::size_t, std::size_t>::iterator map_it;
    std::vector<std::size_t> comp(1);
    
    if (VV.element()->num_sub_elements() == 0)
    {
      boost::shared_ptr<GenericDofMap> dummy = VV.dofmap()->collapse(collapsed_map, *VV.mesh());        
      (*component)++;
      for (map_it =collapsed_map.begin(); map_it!=collapsed_map.end(); ++map_it)
        dof_component_map[map_it->second] = (*component);  
    }
    else
    {
      for (unsigned int i=0; i<VV.element()->num_sub_elements(); i++)
      {
        comp[0] = i;
        boost::shared_ptr<FunctionSpace> Vs = VV.extract_sub_space(comp);
        extract_dof_component_map(dof_component_map, *Vs, component);
      }
    }
  }
    
  void interpolate_nonmatching_mesh(const Function& u0, Function& u) 
  {
    // Interpolate from Function u0 to FunctionSpace of Function u
    boost::shared_ptr<const FunctionSpace> V = u.function_space();

    // Get mesh and dimension
    const Mesh& mesh1 = *V->mesh();
    const std::size_t gdim = mesh1.geometry().dim();

    // Create arrays used to eval one point
    Array<double> values(u.value_size());
    Array<double> x(gdim);
    
    // Create vector to hold all local values of u 
    std::vector<double> local_u_vector(u.vector()->local_size());    
    
    // Get coordinates of all dofs on mesh of this processor
    std::vector<double> coords = V->dofmap()->tabulate_all_coordinates(mesh1);
    
    // Get local dof ownership range
    std::pair<std::size_t, std::size_t> owner_range = V->dofmap()->ownership_range();
        
    // Get a map from global dofs to component number in mixed space
    std::map<unsigned int, unsigned int> dof_component_map;
    int component = -1;
    extract_dof_component_map(dof_component_map, *V, &component);
        
    // Run over meshes on all processors, one mesh at the time
    for (unsigned int i=0; i<MPI::num_processes(); i++)
    {
      // Assign mesh from processor i and broadcast to all processors
      std::vector<double> coords_all;    
      if (MPI::process_number() == i)
        coords_all.assign(coords.begin(), coords.end());        
      
      MPI::broadcast(coords_all, int(i));
      
      // Perform safe eval on all processes to get Function values for all coords
      std::vector<unsigned int> ids;
      std::vector<std::vector<double> > data;
      for (unsigned int j=0; j<coords_all.size()/gdim; j++)
      {
      
        for (unsigned int jj=0; jj<gdim; jj++)
          x[jj] = coords_all[j*gdim+jj];
        
        try
        { // push back when point is found
          u0.eval(values, x);
          std::vector<double> vals;
          for (unsigned int k=0; k<values.size(); k++)
            vals.push_back(values[k]);
          data.push_back(vals);
          ids.push_back(j);
        } 
        catch (std::exception &e)
        { // do-nothing if not found
        }
      }     
      // Gather all found data on process i
      std::vector<std::vector<unsigned int> > all_ids;
      std::vector<std::vector<std::vector<double> > > all_data;
      MPI::gather(ids, all_ids, i);
      MPI::gather(data, all_data, i);
      
      // Move all_data onto the local_u_vector (local on proc i)
      // Choose the correct component using dof_component_map  
      if (MPI::process_number() == i)
      { 
        for (unsigned int j=0; j<MPI::num_processes(); j++)
        {
          for (unsigned int k=0; k<all_ids[j].size(); k++)
          {
            unsigned int m = all_ids[j][k];
            local_u_vector[m] = all_data[j][k][dof_component_map[m+owner_range.first]];
          }
        }
        u.vector()->set_local(local_u_vector);
      }
    }
  }
}
