#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/DistributedMeshTools.h>

// Make DistributedMeshTools::number_entities available to python

namespace dolfin
{
  void dmt_number_entities(const Mesh& mesh, std::size_t d)
  {
    return DistributedMeshTools::number_entities(mesh, d); 
  }
}
