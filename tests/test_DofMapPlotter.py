import subprocess
from fenicstools import DofMapPlotter
import nose

def test_DofMapPlotter():
    '''Test logic used in DofMapPlotter by comparing its string representation
    with FunctionSpace.print_dofmap.'''

    command =\
        '''
from dolfin import * ;
mesh = UnitSquareMesh(1, 1);
S = FunctionSpace(mesh, 'CG', 1);
V = VectorFunctionSpace(mesh, 'CR', 1);
T = TensorFunctionSpace(mesh, 'DG', 0);
M = MixedFunctionSpace([S, V, T]);
M.print_dofmap()
        '''
    # Get output of FunctionSpace.print_dofmap (only print to terminal)
    # Pull back
    process = subprocess.Popen(['python', '-c', command],
                               stdout=subprocess.PIPE)
    dolfin_out, _ = process.communicate()

    # Run the command here to create spaces etc.
    exec(command)

    dmp = DofMapPlotter(M)
    dmp_out = dmp.__str__()

    # The string should match (can't make simple == for strings to work)
    maps_match = all(x == y for x, y in zip(dmp_out, dolfin_out))

    nose.tools.assert_true(maps_match)



