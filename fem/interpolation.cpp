
#include "dolfin.h"

namespace dolfin
{
    
  void extract_dof_component_map(std::map<unsigned int, unsigned int>& dof_component_map, const FunctionSpace& VV, int* component)
  {    
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
    
  void interpolate_nonmatching_mesh(Function& u, const Function& u0, const FunctionSpace& V) 
  {
    boost::shared_ptr<const FunctionSpace> V0 = u0.function_space();
    const Mesh& mesh1 = *V.mesh();
    const std::size_t gdim = mesh1.geometry().dim();
    std::vector<double> coords = V.dofmap()->tabulate_all_coordinates(mesh1);
    std::vector<double> coords_all;    
    std::pair<std::size_t, std::size_t> owner_range = V.dofmap()->ownership_range();
    Array<double> values(u.value_size());
    Array<double> x(gdim);
    MPICommunicator mpi_comm;
    boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);
        
    // Get a map from global dofs to component number in mixed space
    std::map<unsigned int, unsigned int> dof_component_map;
    int component = -1;
    extract_dof_component_map(dof_component_map, V, &component);
    
    // Vectors to store evaluated solutions on all processes
    std::vector<unsigned int> ids;
    std::vector<unsigned int> ids_recv;
    std::vector<std::vector<unsigned int> > all_ids;
    std::vector<int> recvfrom;
    std::vector<std::vector<double> > data;    
    std::vector<std::vector<double> > data_recv;    
    std::vector<std::vector<std::vector<double> > > all_data;
    
    std::vector<double> vals(u0.value_size());
    std::vector<double> local_u_vector(u.vector()->local_size());
    
    // Run over meshes on all processors, one mesh at the time
    for (unsigned int i=0; i<MPI::num_processes(); i++)
    {
      // Assign mesh from processor i and broadcast to all processors
      if (MPI::process_number() == i)
        coords_all.assign(coords.begin(), coords.end());
        
      MPI::broadcast(coords_all, int(i));
      
      // Perform safe eval on all processes to get Function values for all coords
      data.clear();
      ids.clear();
      cout << "Num points = " << coords_all.size()/gdim << endl;
      for (unsigned int j=0; j<coords_all.size()/gdim; j++)
      {
        for (unsigned int jj=0; jj<gdim; jj++)
          x[jj] = coords_all[j*gdim+jj];
        
        try
        {
          // push back when point is found
          u0.eval(values, x);
          vals.clear();
          for (unsigned int k=0; k<values.size(); k++)
            vals.push_back(values[k]);
          data.push_back(vals);
          ids.push_back(j);
        } 
        catch (std::exception &e)
        { // do-nothing
        }
      }      
      // Send all found data to process i where it will be put in local u
      MPI::gather(ids, all_ids, i);
      MPI::gather(data, all_data, i);
      if (MPI::process_number() == i)
      {
        // Move all_data onto the local_u_vector (local on proc i)
        // Choose the correct component using dof_component_map
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
