import cppimport
from dolfin import Function, FunctionSpace, assemble,\
                   TrialFunction, TestFunction, dx, PETScMatrix, VectorFunctionSpace, dot

compiled_i_module = cppimport.imp('fenicstools.fem.cr_interpolation')

def cg1_cr_interpolation_matrix(mesh, constrained_domain=None):
    '''
    Compute matrix that allows fast interpolation of CG1 function to
    Couzeix-Raviart space.
    '''

    CG1 = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain=constrained_domain)
    CR = VectorFunctionSpace(mesh, 'CR', 1, constrained_domain=constrained_domain)

    # Get the matrix with approximate sparsity as the interpolation matrix
    u = TrialFunction(CG1)
    v = TestFunction(CR)
    I = PETScMatrix()
    assemble(dot(u, v)*dx, tensor=I)

    # Fill the interpolation matrix
    d = mesh.geometry().dim()
    compiled_i_module.compute_cg1_cr_interpolation_matrix(I, d)

    return I


