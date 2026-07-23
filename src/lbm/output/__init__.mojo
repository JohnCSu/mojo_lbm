"""Reserved for post-processing helpers that derive fields from raw LBM output.

Currently empty; Q-criterion, drag, and velocity/density extraction live in
`src/lbm/kernels/output.mojo`.
"""

from .velocity import calculate_rho_and_velocity
from .drag import calculate_drag_around_object