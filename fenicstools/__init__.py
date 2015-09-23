from Probe import Probe, Probes, StatisticsProbe, StatisticsProbes
from StructuredGrid import StructuredGrid, ChannelGrid
from WeightedGradient import weighted_gradient_matrix, compiled_gradient_module
from common import getMemoryUsage, SetMatrixValue
from Streamfunctions import StreamFunction, StreamFunction3D
from GaussDivergence import gauss_divergence, divergence_matrix
from Interpolation import interpolate_nonmatching_mesh, interpolate_nonmatching_mesh_any
try:
    from DofMapPlotter import DofMapPlotter
except:
    pass # Probably missing dependency
