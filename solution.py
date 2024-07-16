import numpy as np
import matplotlib.pyplot as plt
from sfepy import data_dir
from sfepy.discrete import Problem
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.conditions import EssentialBC
from sfepy.solvers import Solver
from sfepy.discrete import Function
from sfepy.discrete.projections import make_l2_projection
from sfepy.base.base import Struct

mesh = Mesh.from_file(data_dir + '/meshes/1d/special/cube_1d.mesh')
domain = FEDomain('domain', mesh)
min_x, max_x = domain.get_mesh_bounding_box()[:, 0]
print(f"Domain boundaries: {min_x}, {max_x}")

field = Field.from_args('f', np.float64, 1, domain, approx_order=1)
u = Field.Variable('u', 'unknown', field)
v = Field.Variable('v', 'test', field, primary_var_name='u')

# Define the initial condition - delta function (kind of)
class DeltaInitialCondition(Function):
    def __call__(self, coors, mode=None, **kwargs):
        if mode == 'qp':
            x = coors[:, 0]
            eps = 1e-4
            return np.exp(-((x - 0.5)**2) / (2.0 * eps)) / np.sqrt(2.0 * np.pi * eps)

initial_condition = DeltaInitialCondition()
u.set_data(make_l2_projection(u, initial_condition, order=2))

# Set parameters
sigma = 1.0
mu = 0.8
T = 5.0
dt = 1e-2
theta = 1.0 

# Diffusion and drift terms
C = sigma / 2.0
FX_expr = f"3.0 * ({mu} - x[0])"
FX = Struct(name='FX', vals={0: FX_expr})

# PDE setup
integral = domain.create_integral('i', order=2)
t1 = Term.new('dw_laplace(v, u)', integral, domain, u=u, v=v)
t2 = Term.new('dw_dot(v, u)', integral, domain, u=u, v=v)
t3 = Term.new('dw_lin_elastic(v, u)', integral, domain, u=u, v=v, val=C)
t4 = Term.new('dw_lin_convect(v, u)', integral, domain, u=u, v=v, val=FX)

# Problem definition
equations = {
    'balance': t1 + theta * (t2 + t3 + t4)
}
problem = Problem('fokker_planck', equations=equations)

# Boundary conditions (setting to 0 at x_th)
def boundary(x):
    return np.abs(x - 1.0) < 1e-3

ebc = EssentialBC('fix_u', domain.regions['All'], {'u.all': 0.0})
problem.time_update(ebcs=[ebc])

# Solvers setup
ls = Solver.any_from_conf(problem.ls_conf, problem, u=u)
nls = Solver.any_from_conf(problem.nls_conf, problem, u=u)
problem.set_solvers(nls, ls)

# Time-stepping loop
times = np.arange(0, T + dt, dt)
for t in times:
    problem.time_update(t=t, dt=dt, u=u)
    problem.solve()

    # Calculate the FPT probability
    FPT_probability = 1.0 - u.integral()
    print(f"Time: {t:.2f}, FPT Probability: {FPT_probability:.6f}")

    # Plotting the solution every second
    if np.isclose(t % 1.0, 0.0):
        plt.plot(domain.mesh.coors, u())
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.title(f'Solution at t = {t:.2f}')
        plt.show()
