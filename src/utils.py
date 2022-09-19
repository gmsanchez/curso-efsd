import numpy as np


class Model(object):
    """
    A general class for continuous time models. Should support various types of integrators.
    """

    def __init__(self, Delta, integrator_type="RK4"):
        self.__Delta = Delta
        self.__integrator = get_integrator(self.rhs, Delta, integrator_type=integrator_type)
        self.x0 = None

    def rhs(self, *args):
        raise NotImplementedError("Method RHS should be implemented in child class.")

    def simulate(self, u):
        if self.x0 is None:
            raise ValueError("x0 should be set before trying to simulate.")
        else:
            x1 = self.__integrator.call(self.x0, u)
            self.x0 = x1
            return np.array(self.x0).flatten()


class Rk4Integrator(object):

    def __init__(self, ode, Delta):
        self.__ode = ode
        self.__Delta = Delta

    def call(self, x, u):
        raise NotImplementedError("RK4 method should be implemented here.")


class EulerIntegrator(object):

    def __init__(self, ode, Delta):
        self.__ode = ode
        self.__Delta = Delta

    def call(self, x, u):
        raise NotImplementedError("Euler method should be implemented here.")


def get_integrator(ode, Delta, integrator_type="RK4"):
    if integrator_type=="RK4":
        return Rk4Integrator(ode, Delta)
    elif integrator_type=="Euler":
        return EulerIntegrator(ode, Delta)
    else:
        raise ValueError("%s is a wrong integrator type." % (integrator_type))