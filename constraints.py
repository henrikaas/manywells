"""
Boundary constants for the constraints in the optimization problem
"""
from dataclasses import dataclass

@dataclass
class OptimizationProblemProperties:
    
    # Choke
    u_max: float = 1.
    u_min: float = 0.

    # Gas lift
    gl_max: float = 5.
    gl_min: float = 0.
    comb_gl_max: float = 5.  # Combined gas lift max

    # Water in separator
    wat_max: float = 72.

    # Constraints on move lengths
    r_max: float = 1.
    r_min: float = 0.

    # Number of moves
    max_moves = 8
    max_wells = 20