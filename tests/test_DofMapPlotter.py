#!/usr/bin/env py.test

import pytest
import subprocess
from fenicstools import DofMapPlotter
from dolfin import MPI, mpi_comm_world

def test_DofMapPlotter():
    '''Test logic used in DofMapPlotter by comparing its string representation
    with FunctionSpace.print_dofmap.'''
    
    if MPI.size(mpi_comm_world()) == 1:
        command =\
        '''
from dolfin import * ;
mesh = UnitSquareMesh(4, 4);
S = FunctionSpace(mesh, 'CG', 1);
V = VectorFunctionSpace(mesh, 'CR', 1);
T = TensorFunctionSpace(mesh, 'DG', 0);
M = MixedFunctionSpace([S, V, T]);
M.print_dofmap()
        '''
        # Get output of FunctionSpace.print_dofmap (only print to terminal)
        # Pull back
        # This is run two times just in case FFC sees the spaces for the first time
        # in which case there are some messages printed to terminal and these
        # screw up with the test.
        for i in range(2):
            process = subprocess.Popen(['python', '-c', command],
                                    stdout=subprocess.PIPE)
            dolfin_out, _ = process.communicate()
            #print dolfin_out

        # Run the command here to create spaces etc.
        exec(command)

        dmp = DofMapPlotter(M)
        dmp_out = dmp.__str__()

        # The string should match (can't make simple == for strings to work)
        maps_match = all(x == y for x, y in zip(dmp_out, dolfin_out))

        # Visual check
        # dmp_lines = dmp_out.split('\n')
        # dolfin_lines = dolfin_out.split('\n')
        # for dmp_line, dolfin_line in zip(dmp_lines, dolfin_lines):
        #     print dmp_line, ' vs. ', dolfin_line

        assert maps_match
    else:
        pass
