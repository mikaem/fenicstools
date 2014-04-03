import inspect
from os.path import abspath, join
from dolfin import FunctionSpace, VectorFunctionSpace, Function, interpolate,\
                   compile_extension_module


folder = abspath(join(inspect.getfile(inspect.currentframe()), '../fem'))
code = open(join(folder, 'cr_divergence.cpp'), 'r').read()
compiled_cr_module = compile_extension_module(code=code)

def cr_divergence(u, mesh=None):
    '''
    Compute divergence of a vector field on each cell by integrating the normal
    fluxes across the cell boundary. Integration uses midpoint rule and as such
    is exact for linear vector fields.
    '''

    # Make sure u is a vector
    assert u.rank() == 1

    # See if we have Expression or Function
    try:  # Function?
        _mesh = u.function_space().mesh()
    except AttributeError:  # Maybe Expression
        if mesh is not None:
            _mesh = mesh

    # Interpolate u to Crouzeix-Raviart space
    U = VectorFunctionSpace(_mesh, 'CR', 1)
    _u = interpolate(u, U)
    _u.update()

    # Create a Discontinuous Galerkin function of order 0 to hold the
    # divergence
    V = FunctionSpace(_mesh, 'DG', 0)
    divu = Function(V)

    # Fill in the values
    compiled_cr_module.cr_divergence(divu, _u)

    return divu
