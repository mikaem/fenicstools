from dolfin import *
import pytest
from os import path, makedirs

fixture = pytest.fixture(scope="module")

@fixture
def dirpath():
    dirpath = path.join(path.dirname(path.abspath(__file__)), "tmp", "")
    if not path.isdir(dirpath):
        makedirs(dirpath)
    return dirpath

@fixture
def mesh_2D():
    return UnitSquareMesh(4, 4)

@fixture
def mesh_3D():
    return UnitCubeMesh(4, 4, 4)

@fixture
def V2(mesh_2D):
    return FunctionSpace(mesh_2D, 'CG', 1)

@fixture
def V3(mesh_3D):
    return FunctionSpace(mesh_3D, 'CG', 1)

@fixture
def VF2(mesh_2D):
    return VectorFunctionSpace(mesh_2D, "CG", 1)

@fixture
def VF3(mesh_3D):
    return VectorFunctionSpace(mesh_3D, "CG", 1)

@fixture
def mesh_2D_self():
    return UnitSquareMesh(mpi_comm_self(), 4, 4)

@fixture
def mesh_3D_self():
    return UnitCubeMesh(mpi_comm_self(), 4, 4, 4)

@fixture
def V2_self(mesh_2D_self):
    return FunctionSpace(mesh_2D_self, 'CG', 1)

@fixture
def V3_self(mesh_3D_self):
    return FunctionSpace(mesh_3D_self, 'CG', 1)

@fixture
def VF2_self(mesh_2D_self):
    return VectorFunctionSpace(mesh_2D_self, "CG", 1)

@fixture
def VF3_self(mesh_3D_self):
    return VectorFunctionSpace(mesh_3D_self, "CG", 1)
