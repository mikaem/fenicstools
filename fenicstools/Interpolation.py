__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-12-13"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
import os, inspect
from dolfin import Function, compile_extension_module

fem_folder = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), "../fem"))
fem_code = open(os.path.join(fem_folder, 'interpolation.cpp'), 'r').read()
#fem_code = open(os.path.join(fem_folder, 'interpolation_nonmatching2.cpp'), 'r').read()
compiled_fem_module = compile_extension_module(code=fem_code)

def interpolate_nonmatching_mesh(u0, V):
    """Interpolate from GenericFunction u0 to FunctionSpace V.
    
    The FunctionSpace V can have a different mesh than that of u0, if u0 
    has a mesh.
    
    """
    u = Function(V)
    compiled_fem_module.interpolate(u0, u)
    return u
