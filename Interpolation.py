__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-12-13"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
import os, inspect
from dolfin import Function, compile_extension_module

fem_folder = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), "../fem"))
fem_code = open(os.path.join(fem_folder, 'interpolation.cpp'), 'r').read()
compiled_fem_module = compile_extension_module(code=fem_code)
                                           
def interpolate_nonmatching_mesh(u0, V):
    """Interpolate from GenericFunction u0 to FunctionSpace V.
    
    The FunctionSpace V can have a different mesh than that of u0, if u0 
    has a mesh.
    
    """
    u = Function(V)
    compiled_fem_module.interpolate_nonmatching_mesh(u0, u)
    return u

if __name__ == "__main__":
    from dolfin import *
    # Test for nonmatching mesh and FunctionSpace
    mesh = UnitCubeMesh(16, 16, 16)
    mesh2 = UnitCubeMesh(32, 32, 32)
    V = FunctionSpace(mesh, 'CG', 1)
    V2 = FunctionSpace(mesh2, 'CG', 1)
    # Just create some random data to be used for probing
    x0 = interpolate(Expression('x[0]'), V)
    u = interpolate_nonmatching_mesh(x0, V2)
    
    VV = VectorFunctionSpace(mesh, 'CG', 1)
    VV2 = VectorFunctionSpace(mesh2, 'CG', 1)
    v0 = interpolate(Expression(('x[0]', '2*x[1]', '3*x[2]')), VV)    
    v = interpolate_nonmatching_mesh(v0, VV2)
    