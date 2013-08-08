
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
    
  void interpolate_nonmatching_mesh(const GenericFunction& u0, Function& u) 
  {
    // Interpolate from GenericFunction u0 to FunctionSpace of Function u
    // The FunctionSpace of u can have a different mesh than that of u0 
    
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
    
    // Get dof ownership range
    std::pair<std::size_t, std::size_t> owner_range = V->dofmap()->ownership_range();
        
    // Get a map from global dofs to component number in mixed space
    std::map<unsigned int, unsigned int> dof_component_map;
    int component = -1;
    extract_dof_component_map(dof_component_map, *V, &component);
        
    // Search this process first for all coordinates in new mesh
    std::vector<unsigned int> ids;
    std::vector<std::vector<double> > data;
    std::vector<unsigned int> ids_not_found;
    std::vector<double> coords_not_found;
    for (unsigned int j=0; j<coords.size()/gdim; j++)
    {    
      for (unsigned int jj=0; jj<gdim; jj++)
        x[jj] = coords[j*gdim+jj];
    
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
      { // If not found then it must be seached on the other processes
        ids_not_found.push_back(j+owner_range.first);
        for (unsigned int jj=0; jj<gdim; jj++)
          coords_not_found.push_back(x[jj]);
      }
    }
    
    // Store the points found on local_u_vector
    for (unsigned int k=0; k<ids.size(); k++)
    {
      unsigned int m = ids[k];
      local_u_vector[m] = data[k][dof_component_map[m+owner_range.first]];
    }
    
    // Send all points not found to processor with one higher rank
    // Search there and then send found points back and not found to 
    // next processor in line. By the end of this loop all processors 
    // will have been searched and thus if not found the point is not
    // in the mesh of Function u0. In that case the point will take
    // the value of zero.
    for (unsigned int k = 1; k < MPI::num_processes(); ++k)
    {
      std::vector<double> coords_not_found_recv;
      std::vector<unsigned int> ids_not_found_recv;
           
      int src = (MPI::process_number()-1+MPI::num_processes()) % MPI::num_processes();
      int dest =  (MPI::process_number()+1) % MPI::num_processes();
      
      MPI::send_recv(ids_not_found, dest, ids_not_found_recv, src);      
      MPI::send_recv(coords_not_found, dest, coords_not_found_recv, src);
      
      ids_not_found.clear();
      coords_not_found.clear();
      
      // Search this processor for points
      std::vector<unsigned int> found_ids;
      std::vector<std::vector<double> > found_data;
      for (unsigned int j=0; j<coords_not_found_recv.size()/gdim; j++)
      {        
        unsigned int m = ids_not_found_recv[j];
        for (unsigned int jj=0; jj<gdim; jj++)
          x[jj] = coords_not_found_recv[j*gdim+jj];

        try
        { // push back when point is found
          u0.eval(values, x);
          std::vector<double> vals;
          for (unsigned int jj=0; jj<values.size(); jj++)
            vals.push_back(values[jj]);
          found_data.push_back(vals);
          found_ids.push_back(m);
        } 
        catch (std::exception &e)
        { // If not found then it will be sent to next rank
          ids_not_found.push_back(m);
          for (unsigned int jj=0; jj<gdim; jj++)
            coords_not_found.push_back(x[jj]);
        }
      }     
      // Send found data back to owner (MPI::process_number()-k)        
      std::vector<unsigned int> found_ids_recv;
      std::vector<std::vector<double> > found_data_recv;
      dest = (MPI::process_number()-k+MPI::num_processes()) % MPI::num_processes();
      src  = (MPI::process_number()+k) % MPI::num_processes();
      MPI::send_recv(found_ids, dest, found_ids_recv, src);
      MPI::send_recv(found_data, dest, found_data_recv, src);

      // Move all_data onto the local_u_vector
      // Choose the correct component using dof_component_map  
      for (unsigned int kk=0; kk<found_ids_recv.size(); kk++)
      {
        unsigned int m = found_ids_recv[kk]-owner_range.first;
        local_u_vector[m] = found_data_recv[kk][dof_component_map[m+owner_range.first]];
      }
    }
    u.vector()->set_local(local_u_vector);
  }
}
