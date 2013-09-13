
#include "dolfin.h"

namespace dolfin
{
    
  void extract_dof_component_map(std::map<std::size_t, std::size_t>& dof_component_map, 
                                 const FunctionSpace& VV, int* component)
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
      for (std::size_t i=0; i<VV.element()->num_sub_elements(); i++)
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
    // (if u0 has a mesh)
    //
    // The algorithm is like this
    //
    //   1) Tabulate all coordinates for all dofs in u.function_space()
    //   2) Create a map from dof to component number in Mixed Space.
    //   3) Evaluate u0 for all coordinates in u (computed in 1)). 
    //        Problem here is that u0 and u will have different meshes 
    //        and as such a vertex in u will not necessarily be found 
    //        on the same processor for u0. Hence the vertex will be 
    //        passed around and searched on all ranks until found.
    //   4) Set all values in local u using the dof to component map
    
    // Get the function space interpolated to
    boost::shared_ptr<const FunctionSpace> V = u.function_space();
    
    // Get mesh and dimension of the FunctionSpace interpolated to
    const Mesh& mesh = *V->mesh();
    const std::size_t gdim = mesh.geometry().dim();

    // Create arrays used to evaluate one point
    std::vector<double> x(gdim);
    std::vector<double> values(u.value_size());
    Array<double> _x(gdim, x.data());
    Array<double> _values(u.value_size(), values.data());
    
    // Create vector to hold all local values of u 
    std::vector<double> local_u_vector(u.vector()->local_size());
    
    // Get coordinates of all dofs on mesh of this processor
    std::vector<double> coords = V->dofmap()->tabulate_all_coordinates(mesh);
    
    // Get dof ownership range
    std::pair<std::size_t, std::size_t> owner_range = V->dofmap()->ownership_range();
        
    // Get a map from global dofs to component number in mixed space
    std::map<std::size_t, std::size_t> dof_component_map;
    int component = -1;
    extract_dof_component_map(dof_component_map, *V, &component);
        
    // Search this process first for all coordinates in u's local mesh
    std::vector<std::size_t> global_dofs_not_found;
    std::vector<double> coords_not_found;
    for (std::size_t j=0; j<coords.size()/gdim; j++)
    {    
      std::copy(coords.begin()+j*gdim, coords.begin()+(j+1)*gdim, x.begin());
      try
      { // store when point is found
        u0.eval(_values, _x);
        local_u_vector[j] = values[dof_component_map[j+owner_range.first]];
      } 
      catch (std::exception &e)
      { // If not found then it must be seached on the other processes
        global_dofs_not_found.push_back(j+owner_range.first);
        for (std::size_t jj=0; jj<gdim; jj++)
          coords_not_found.push_back(x[jj]);
      }
    }
    
    // Send all points not found to processor with one higher rank.
    // Search there and send found points back to owner and not found to 
    // next processor in line. By the end of this loop all processors 
    // will have been searched and thus if not found the point is not
    // in the mesh of Function u0. In that case the point will take
    // the value of zero.
    for (std::size_t k = 1; k < MPI::num_processes(); ++k)
    {
      std::vector<double> coords_recv;
      std::vector<std::size_t> global_dofs_recv;
           
      std::size_t src = (MPI::process_number()-1+MPI::num_processes()) % MPI::num_processes();
      std::size_t dest =  (MPI::process_number()+1) % MPI::num_processes();
      
      MPI::send_recv(global_dofs_not_found, dest, global_dofs_recv, src);
      MPI::send_recv(coords_not_found, dest, coords_recv, src);
      
      global_dofs_not_found.clear();
      coords_not_found.clear();
      
      // Search this processor for received points
      std::vector<std::size_t> global_dofs_found;
      std::vector<std::vector<double> > coefficients_found;
      for (std::size_t j=0; j<coords_recv.size()/gdim; j++)
      {        
        std::size_t m = global_dofs_recv[j];
        std::copy(coords_recv.begin()+j*gdim, coords_recv.begin()+(j+1)*gdim, x.begin());

        try
        { // push back when point is found
          u0.eval(_values, _x);
          coefficients_found.push_back(values);
          global_dofs_found.push_back(m);
        } 
        catch (std::exception &e)
        { // If not found then collect and send to next rank
          global_dofs_not_found.push_back(m);
          for (std::size_t jj=0; jj<gdim; jj++)
            coords_not_found.push_back(x[jj]);
        }
      }     
      
      // Send found coefficients back to owner (dest)
      std::vector<std::size_t> global_dofs_found_recv;
      std::vector<std::vector<double> > coefficients_found_recv;
      dest = (MPI::process_number()-k+MPI::num_processes()) % MPI::num_processes();
      src  = (MPI::process_number()+k) % MPI::num_processes();
      MPI::send_recv(global_dofs_found, dest, global_dofs_found_recv, src);
      MPI::send_recv(coefficients_found, dest, coefficients_found_recv, src);

      // Move all found coefficients onto the local_u_vector
      // Choose the correct component using dof_component_map  
      for (std::size_t j=0; j<global_dofs_found_recv.size(); j++)
      {
        std::size_t m = global_dofs_found_recv[j]-owner_range.first;
        std::size_t n = dof_component_map[m+owner_range.first];
        local_u_vector[m] = coefficients_found_recv[j][n];
      }
      
      // Note that this algorithm computes and sends back all values, 
      // i.e., coefficients_found pushes back the entire vector for all 
      // components in mixed space. An alternative algorithm is to send 
      // around the correct component number in addition to global dof number 
      // and coordinates and then just send back the correct value.
    }
    u.vector()->set_local(local_u_vector);
  }
}
