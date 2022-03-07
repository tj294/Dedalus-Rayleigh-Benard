"""
Parameters for rb_convect.py
"""
import numpy as np

a = 3
Nx, Nz = 256, 64
Pr = 0.8
Ra = 1e6

dt = 3e-6
max_dt = 1e-5

end_sim_time = np.inf
end_wall_time = np.inf
end_iteration = 1e5

snapshot_iter = 50
analysis_iter = 50
