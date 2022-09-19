import numpy as np
import matplotlib.pyplot as plt
from src import utils

def sol_a(t):
    # Analytical solution of the mass-spring damper system. Seen in class on 15/09/2022
    return -(5.0 / 6.0) * np.exp(-t / 10.0) * (3.0 * np.cos(3.0 * t / 10.0) + np.sin(3.0 * t / 10.0)) + 5.0 / 2.0


class SpringDamper(utils.Model):
    """
    Child class of utils.Model. Describes the Right Hand Side of a continuous time system \dot{x} = f(x,u).
    Right now, integrator is either RK4 or Euler.
    """
    def __init__(self, Delta, integrator_type="RK4"):
        super().__init__(Delta, integrator_type=integrator_type)
        self.Nx = 2
        self.Nu = 1
        self.x0 = np.zeros((self.Nx,))

    def rhs(self, x, u):
        # This method overrides the parent class method.
        c = 4.0  # Damping constant
        k = 2.0  # Stiffness of the spring
        m = 20.0  # Mass
        # F = 5.0  # Force
        A = np.array([[0.0, 1.0], [-k / m, -c / m]])
        B = np.array([[0.0], [1.0 / m]])
        return np.array(np.matmul(A, x) + np.matmul(B, u)).flatten()


Delta = 0.5
Delta_1 = 0.25

tstart = 0
tstop = 60

t = np.arange(tstart, tstop + Delta, Delta)
t_1 = np.arange(tstart, tstop + Delta_1, Delta_1)
N = int((tstop-tstart)/Delta_1) # Simulation length

# Analytical solution simulation
x_a = sol_a(t)


m_rk4 = SpringDamper(Delta=Delta_1, integrator_type="RK4")
m_euler = SpringDamper(Delta=Delta_1, integrator_type="Euler")

# Array of system states
x_sim_rk4 = np.zeros((N + 1, m_rk4.Nx))
x_sim_euler = np.zeros((N+1, m_euler.Nx))
# System input. Constant force of 5
u_sim = np.ones((N, m_rk4.Nu)) * 5.0

for k in range(N):
    # x_sim_rk4[k + 1] = m_rk4.simulate(u_sim[k])
    x_sim_euler[k + 1] = m_euler.simulate(u_sim[k])


plt.figure(1)
plt.plot(t, x_a, label="x1 (a)", marker='o', markevery=10)

plt.plot(t_1, x_sim_rk4[:, 0], label="x1 (RK4)", marker=".", markevery=10)
plt.plot(t_1, x_sim_rk4[:, 1], label="x2 (RK4)")

plt.plot(t_1, x_sim_euler[:, 0], label="x1 (Euler)", marker=".", markevery=10)
plt.plot(t_1, x_sim_euler[:, 1], label="x2 (Euler)")

plt.title('Simulation of the Mass-Spring-Damper System')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid()
plt.legend()
plt.show()
