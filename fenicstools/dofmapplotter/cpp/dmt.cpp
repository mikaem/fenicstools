/*
<%
from dolfin.jit.jit import dolfin_pc
setup_pybind11(cfg)
cfg['libraries'] = dolfin_pc['libraries']
cfg['include_dirs'] = dolfin_pc['include_dirs']
cfg['library_dirs'] = dolfin_pc['library_dirs']
%>
*/
#include <pybind11/pybind11.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/DistributedMeshTools.h>

// Make DistributedMeshTools::number_entities available to python

using namespace dolfin

void dmt_number_entities(const Mesh& mesh, std::size_t d)
{
  return DistributedMeshTools::number_entities(mesh, d); 
}

namespace py = pybind11;

PYBIND11_MODULE(dmt, m)
{
    m.def("dmt_number_entities", &dmt_number_entities);
}

