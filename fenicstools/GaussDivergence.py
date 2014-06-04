__author__ = "Miroslav Kuchta <mirok@math.uio.no>"
__date__ = "2014-04-05"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import inspect
from dolfin import TensorFunctionSpace, VectorFunctionSpace, FunctionSpace,\
    Function, interpolate, compile_extension_module, GenericFunction, assemble, \
    TrialFunction, TestFunction, dx, Matrix, dot, div
from fenicstools import SetMatrixValue
from os.path import abspath, join

folder = abspath(join(inspect.getfile(inspect.currentframe()), '../fem'))
code = open(join(folder, 'cr_divergence.cpp'), 'r').read()
compiled_cr_module = compile_extension_module(code=code)


def gauss_divergence(u, mesh=None):
    '''
    This function uses Gauss divergence theorem to compute divergence of u
    inside the cell by integrating normal fluxes across the cell boundary.
    If u is a vector or tensor field the result of computation is diverence
    of u in the cell center = DG0 scalar/vector function. For scalar fields,
    the result is grad(u) = DG0 vector function. The fluxes are computed by
    midpoint rule and as such the computed divergence is exact for linear
    fields.
    '''

    # Require u to be GenericFunction
    assert isinstance(u, GenericFunction)

    # Require u to be scalar/vector/rank 2 tensor
    rank = u.value_rank()
    assert rank in [0, 1, 2]

    # For now, there is no support for manifolds
    if mesh is None:
        _mesh = u.function_space().mesh()
    else:
        _mesh = mesh
    tdim = _mesh.topology().dim()
    gdim = _mesh.geometry().dim()
    assert tdim == gdim

    for i in range(rank):
        assert u.value_dimension(i) == gdim

    # Based on rank choose the type of CR1 space where u should be interpolated
    # to to get the midpoint values + choose the type of DG0 space for
    # divergence
    if rank == 1:
        DG = FunctionSpace(_mesh, 'DG', 0)
        CR = VectorFunctionSpace(_mesh, 'CR', 1)
    else:
        DG = VectorFunctionSpace(_mesh, 'DG', 0)
        if rank == 0:
            CR = FunctionSpace(_mesh, 'CR', 1)
        else:
            CR = TensorFunctionSpace(_mesh, 'CR', 1)

    divu = Function(DG)
    _u = interpolate(u, CR)

    # Use Gauss theorem cell by cell to get the divergence. The implementation
    # is based on divergence(vector) = scalar and so the spaces for these
    # two need to be provided
    if rank == 1:
        pass  # CR, DG are correct already
    else:
        DG = FunctionSpace(_mesh, 'DG', 0)
        CR = VectorFunctionSpace(_mesh, 'CR', 1)
    compiled_cr_module.cr_divergence(divu, _u, DG, CR)

    return divu

from CRInterpolation import cg1_cr_interpolation_matrix
def divergence_matrix(mesh):
    CR = VectorFunctionSpace(mesh, 'CR', 1)
    DG = FunctionSpace(mesh, 'DG', 0)
    A = cg1_cr_interpolation_matrix(mesh)    
    M  = assemble(dot(div(TrialFunction(CR)), TestFunction(DG))*dx())
    C = compiled_cr_module.cr_divergence_matrix(M, A, DG, CR)
    return C
