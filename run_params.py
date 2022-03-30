"""
Parameters for rb_convect.py
"""
import numpy as np

a = 3
Ny, Nz = 256, 64
Pr = 1.0
Ra = 1e6

dt = 1e-5
max_dt = 5e-4

end_sim_time = 1.6
end_wall_time = np.inf
end_iteration = np.inf

snapshot_iter = 50
analysis_iter = 50
