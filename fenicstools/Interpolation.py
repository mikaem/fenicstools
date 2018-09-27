__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-12-13"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
from dolfin import Function
import cppimport

compiled_fem_module = cppimport.imp('fenicstools.fem.interpolation')

def interpolate_nonmatching_mesh(u0, V):
    """Interpolate from GenericFunction u0 to FunctionSpace V.

    The FunctionSpace V can have a different mesh than that of u0, if u0
    has a mesh.

    """
    u = Function(V)
    compiled_fem_module.interpolate(u0, u)
    return u

def interpolate_nonmatching_mesh_any(u0, V):
    """Interpolate from GenericFunction u0 to FunctionSpace V.

    The FunctionSpace V can have a different mesh than that of u0, if u0
    has a mesh.

    This function works for any finite element space, not just Lagrange.

    """
    u = Function(V)
    compiled_fem_module.interpolate_any(u0, u)
    return u
