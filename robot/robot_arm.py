import multiprocessing
import numpy as np
import scipy.integrate as integrate
from sympy import Symbol, Function, sin, cos, simplify, symbols, Eq
from sympy.calculus.euler import euler_equations
from sympy import lambdify
from sympy.solvers import solve


class RobotArm:
  ''' A two-link robot arm with two point masses at the end of each link. The system is modeled as a two-link pendulum with damping coefficients. '''

  def __init__(self):
    self.t = Symbol('t')
    self.theta1 = Function('theta1')
    self.theta2 = Function('theta2')
    self._kinetic_energy = None
    self._potential_energy = None
    self._lagragian = None
    self._euler_lagrange = None
    self._euler_lagrange_dissipation = None

  def _post_init(self):
    self._init_coordinates()
    self._init_dissipation()

  def _init_dissipation(self):
    self.R1 = -1 / 2 * self.mu1 * self.theta1(self.t).diff(self.t)**2
    self.R2 = -1 / 2 * self.mu2 * self.theta2(self.t).diff(self.t)**2
    self.Q1 = self.R1.diff(self.theta1(self.t).diff(self.t))
    self.Q2 = self.R2.diff(self.theta2(self.t).diff(self.t))

  def _init_coordinates(self):
    self.x1 = self.l1 * sin(self.theta1(self.t))
    self.x2 = self.l1 * sin(self.theta1(self.t)) + self.l2 * sin(self.theta2(self.t))
    self.y1 = -self.l1 * cos(self.theta1(self.t))
    self.y2 = -self.l1 * cos(self.theta1(self.t)) - self.l2 * cos(self.theta2(self.t))

  def get_coordinates(self) -> tuple[tuple[float, float], tuple[float, float]]:
    return (self.x1, self.y1), (self.x2, self.y2)

  @property
  def kinetic_energy(self) -> Function:
    ''' Returns the kinetic energy of the system. '''
    if not self._kinetic_energy:
      T1 = Function('T1')
      T2 = Function('T2')
      T1 = 1 / 2 * self.m1 * (self.x1.diff(self.t)**2 + self.y1.diff(self.t)**2)
      T2 = 1 / 2 * self.m2 * (self.x2.diff(self.t)**2 + self.y2.diff(self.t)**2)
      self._kinetic_energy = simplify(T1 + T2)
    return self._kinetic_energy

  @property
  def potential_energy(self) -> Function:
    ''' Returns the potential energy of the system. '''
    if not self._potential_energy:
      self._potential_energy = simplify(self.m1 * self.g * self.y1 + self.m2 * self.g * self.y2)
    return self._potential_energy

  @property
  def lagrangian(self) -> Function:
    ''' Returns the Lagrangian of the system (L = T - V). '''
    if not self._lagragian:
      self._lagragian = simplify(self.kinetic_energy - self.potential_energy)
    return self._lagragian

  @property
  def euler_lagrange(self) -> tuple[Function, Function]:
    ''' Returns the Euler-Lagrange of the system. 

        The Euler-Lagrange equations: d/dt (∂L/∂q̇) - ∂L/∂q = 0
        
        In this implementation:
        - q corresponds to the generalized coordinates [θ1(t), θ2(t)].
        - L is the Lagrangian of the system, which is precomputed from kinetic and potential energy.

        Returns
        -------
        tuple[Function, Function]: Two Euler-Lagrange equations for θ1 and θ2.
    '''
    if not self._euler_lagrange:
      self._euler_lagrange = euler_equations(self.lagrangian, [self.theta1(self.t), self.theta2(self.t)], self.t)
      self._euler_lagrange = [self._euler_lagrange[0].lhs, self._euler_lagrange[1].lhs]
    return self._euler_lagrange

  @property
  def euler_lagrange_eq(self) -> list[Eq, Eq]:
    ''' Returns the Euler-Lagrange equations of the system.

        The Euler-Lagrange equation: d/dt (∂L/∂q̇) - ∂L/∂q = 0
    '''
    return [Eq(self.euler_lagrange[0], 0), Eq(self.euler_lagrange[1], 0)]

  @property
  def euler_lagrange_dissipation_eq(self) -> list[Eq, Eq]:
    ''' Returns the Euler-Lagrange equations of the system with dissipation.

        The Euler-Lagrange equation: d/dt (∂L/∂q̇) - ∂L/∂q = Q = -dR/dq̇

        In this implementation:
        - q corresponds to the generalized coordinates [θ1(t), θ2(t)].
        - L is the Lagrangian of the system, which is precomputed from kinetic and potential energy.
        - R is the dissipation function of the system.

        Returns
        -------
        tuple[Function, Function]: Two Euler-Lagrange equations for θ1 and θ2.
    '''
    # Through the symbolic computation, the euler lagrange equation was calculated by (-1), therfore the frictions are multiplied by (-1) either!
    return [Eq(self.euler_lagrange[0], -1 * self.Q1), Eq(self.euler_lagrange[1], -1 * self.Q2)]

  def solve_euler_lagrange(self, equation: list[Eq, Eq]) -> list[Function, Function]:
    ''' Solves the Euler-Lagrange equations of the system for the second derivatives of the angles.

        This method computes the angular accelerations (second derivatives) of the system's 
        generalized coordinates (θ1 and θ2) from the Euler-Lagrange equations.

        Parameters:
        ----------
        equation :  A list of symbolic equations representing the Euler-Lagrange equations. Each equation corresponds to one generalized coordinate (θ1 or θ2).

        Returns:
        -------
        A list containing the solutions for the second derivatives of θ1 and θ2 (i.e., [d²θ1/dt², d²θ2/dt²]).
        
        Example:
        --------
        If the input equations are: [Eq(d/dt(∂L/∂(dθ1/dt)) - ∂L/∂θ1, 0), Eq(d/dt(∂L/∂(dθ2/dt)) - ∂L/∂θ2, 0)]
        The method will return: [d²θ1/dt², d²θ2/dt²]
    '''
    return solve(equation, [self.theta1(self.t).diff(self.t, 2), self.theta2(self.t).diff(self.t, 2)])

  def first_order_transform(self, equation: Function) -> Function:
    ''' Substitutes the second order differential equation with first order differential equation 
    
        Rewrite the second-order system as a first-order system:
          z = [z1, z2, z3, z4] 
        where:
          z1 = theta1 
          z2 = theta2 
          z3 = d(theta1)/dt 
          z4 = d(theta2)/dt

        Parameters
        ----------
        equation: A symbolic equation representing the second order differential equation of the system: d²θ/dt² = f(θ, dθ/dt)

        Returns
        -------
        A first order differential equation of the system: dz/dt = f(z, t) = [dz1/dt, dz2/dt, dz3/dt, dz4/dt]
    '''
    z1, z2, z3, z4 = symbols('z1, z2, z3, z4')
    equation = equation.subs({self.theta1(self.t).diff(self.t): z3, self.theta2(self.t).diff(self.t): z4})
    equation = equation.subs({self.theta1(self.t): z1, self.theta2(self.t): z2})
    return simplify(equation)


class SymbolicRobotArm(RobotArm):
  ''' Symbolic Robot Arm '''

  def __init__(self):
    super().__init__()
    self.m1 = Symbol('m1')
    self.m2 = Symbol('m2')
    self.g = Symbol('g')
    self.l1 = Symbol('l1')
    self.l2 = Symbol('l2')
    self.mu1 = Symbol('mu1')
    self.mu2 = Symbol('mu2')
    super()._post_init()


class NumericRobotArm(RobotArm):
  ''' Numeric Robot Arm '''

  def __init__(self, m1: float, m2: float, l1: float, l2: float, mu1: float, mu2: float):
    ''' Initializes the numeric robot arm with the given parameters.
    
        Parameters
        ----------
        t_stop: Time to stop the simulation
        theta1: Initial angle of the first link
        theta2: Initial angle of the second link
        m1: Pointmass of the first link
        m2: Pointmass of the second link
        l1: Length of the first link
        l2: Length of the second link
        mu1: Damping coefficient of the first link
        mu2: Damping coefficient of the second link
    '''
    super().__init__()
    self.m1 = m1
    self.m2 = m2
    self.g = 9.81
    self.l1 = l1
    self.l2 = l2
    self.mu1 = mu1
    self.mu2 = mu2
    super()._post_init()

  def solve_odeint(self,
                   z0: np.array,
                   eq_theta1: Function,
                   eq_theta2: Function,
                   time_points: np.array,
                   pertubation_variance: float = None) -> np.array:
    ''' Solves the system of first order differential equations using odeint method.

        Parameters:
        ----------
        z0: Initial conditions of the system [θ1(0), θ2(0), dθ1/dt(0), dθ2/dt(0)]
        eq_theta1: First order differential equation for θ1
        eq_theta2: First order differential equation for θ2
        time_range: Time vector

        Returns:
        -------
        A numpy array containing the generalized coordinates [θ1, θ2, dθ1/dt, dθ2/dt] of the system.
    '''
    z1, z2, z3, z4 = symbols('z1 z2 z3 z4')
    func_theta1 = lambdify([z1, z2, z3, z4], eq_theta1, modules='numpy')
    func_theta2 = lambdify([z1, z2, z3, z4], eq_theta2, modules='numpy')

    def derivatives(state, t):
      z1, z2, z3, z4 = state
      dz1 = z3  # z1' = z3
      dz2 = z4  # z2' = z4
      dz3 = func_theta1(z1, z2, z3, z4)  # z3' from equation_theta1
      dz4 = func_theta2(z1, z2, z3, z4)  # z4' from equation_theta2
      return [dz1, dz2, dz3, dz4]

    trajectory = integrate.odeint(derivatives, z0, time_points)
    if pertubation_variance:
      noise = np.random.normal(0, pertubation_variance, trajectory.shape)
      trajectory += noise
    return trajectory

  def convert_to_coordinates(self, y: np.array) -> tuple[np.array, np.array, np.array, np.array]:
    ''' Converts the generalized coordinates to the x and y coordinates of the two links of the robot arm.

        Parameters:
        ----------
        y: A numpy array containing the generalized coordinates [θ1, θ2, dθ1/dt, dθ2/dt] of the system.

        Returns:
        -------
        A tuple containing the x and y coordinates of the two links of the robot arm.
    '''
    x1 = self.l1 * np.sin(y[:, 0])
    y1 = -self.l1 * np.cos(y[:, 0])
    x2 = self.l2 * np.sin(y[:, 1]) + x1
    y2 = -self.l2 * np.cos(y[:, 1]) + y1
    return x1, y1, x2, y2

  def energy_over_time(self, trajectories: list[np.array], time_points: np.array) -> np.array:
    ''' 
    Computes the energy of the system over time.
    
    Parameters:
    ----------
    trajectories: A list of numpy arrays containing the generalized coordinates [θ1, θ2, dθ1/dt, dθ2/dt] of the system.
    time_points:  Time vector
    
    Returns:
    -------
    Array of total energy values over time.
    '''
    theta1, theta2 = trajectories[:, 0], trajectories[:, 1]
    omega_1, omega2 = trajectories[:, 2], trajectories[:, 3]
    T_func = lambdify(
        [self.theta1(self.t),
         self.theta2(self.t),
         self.theta1(self.t).diff(self.t),
         self.theta2(self.t).diff(self.t)],
        self.kinetic_energy,
        modules='numpy')
    V_func = lambdify([self.theta1(self.t), self.theta2(self.t)], self.potential_energy, modules='numpy')
    T_vals = T_func(theta1, theta2, omega_1, omega2)
    V_vals = V_func(theta1, theta2)
    energy_vals = T_vals + abs(V_vals)
    return np.round(energy_vals, 0)


def get_equation_of_motion(system: RobotArm, equation: Function) -> tuple[Function, Function]:
  ''' Get the equation of motions dθ1/dt^2, dθ2/dt^2 '''
  eq_theta1 = equation[system.theta1(system.t).diff(system.t, 2)]
  eq_theta2 = equation[system.theta2(system.t).diff(system.t, 2)]
  return simplify(eq_theta1), simplify(eq_theta2)


def numeric_robot_arm(robot_arm: NumericRobotArm,
                      time_points: np.array,
                      z0: np.array,
                      dt: float,
                      pertubation_variance: float = None) -> np.array:
  ''' Solves the equations of motion for the robot arm using the numeric approach. 

      Parameters
      ----------
      t_stop:     Time to stop the simulation
      z0:         Initial conditions of the system [θ1(0), θ2(0), ω1(0) , ω2(0)]
      dt:         Time step

      Returns
      -------
      A numpy array containing the generalized coordinates [θ1, θ2, dθ1/dt, dθ2/dt] of the system.
  '''
  numeric_equation = robot_arm.solve_euler_lagrange(robot_arm.euler_lagrange_dissipation_eq)
  eq_theta1, eq_theta2 = get_equation_of_motion(system=robot_arm, equation=numeric_equation)
  eq_theta1 = robot_arm.first_order_transform(equation=eq_theta1)
  eq_theta2 = robot_arm.first_order_transform(equation=eq_theta2)
  trajectory = robot_arm.solve_odeint(z0=z0,
                                      eq_theta1=eq_theta1,
                                      eq_theta2=eq_theta2,
                                      time_points=time_points,
                                      pertubation_variance=pertubation_variance)
  return trajectory


def symbolic_robot_arm(log: bool = True) -> np.array:
  ''' Solves the equations of motion for the robot arm using the symbolic approach. '''
  robot_arm = SymbolicRobotArm()
  symbolic_equation = robot_arm.solve_euler_lagrange(robot_arm.euler_lagrange_dissipation_eq)
  eq_theta1, eq_theta2 = get_equation_of_motion(system=robot_arm, equation=symbolic_equation)
  eq_theta1 = robot_arm.first_order_transform(equation=eq_theta1)
  eq_theta2 = robot_arm.first_order_transform(equation=eq_theta2)
  if log:
    print(f'Euler-Lagrange Equation: {symbolic_equation}')
    print(f'First Order Equation respect to θ1: {eq_theta1}')
    print(f'First Order Equation repect to θ2: {eq_theta2}')


def process_trajectories(z0: list[np.array],
                         time_points: np.array,
                         dt: float,
                         trajectory_version: str,
                         create_trajectories: callable,
                         pertubation: bool = False) -> list[np.array]:
  '''
  Processes trajectory data in parallel and saves the generated trajectories to an .npz file.

  Parameters
  ----------
  z0:                  Initial conditions for the trajectories.
  time_points:         Time points for trajectory generation.
  dt:                  Time step for trajectory generation.
  trajectory_version:  Version number for the trajectory data.
  create_trajectories: Callable to generate trajectories for a given chunk.

  Returns
  -------
  List of generated trajectories.
  '''
  num_cpus = multiprocessing.cpu_count()
  z0_length = len(z0)
  chunk_size = z0_length // num_cpus
  if chunk_size == 0:
    chunks = [z0]
  else:
    chunks = [z0[i:i + chunk_size] for i in range(0, z0_length, chunk_size)]
    if z0_length % num_cpus != 0:
      chunks[-2].extend(chunks.pop(-1))
  file_name_base = f'pert_trajectories_v{trajectory_version}' if pertubation else f'trajectories_v{trajectory_version}'
  if chunk_size > 0:
    args_list = [(chunk, time_points, dt, False) for chunk in chunks]
    with multiprocessing.Pool(processes=num_cpus) as pool:
      results = pool.starmap(create_trajectories, args_list)
  else:
    results = [create_trajectories(chunk, time_points, dt, False) for chunk in chunks]
  final_trajectories = [item for sublist in results for item in sublist]
  output_file = f'data/trajectories/{file_name_base}.npz'
  np.savez(output_file, *final_trajectories)
  print(f"Trajectories created and saved to {output_file}.")
  return final_trajectories


def create_init_states(version: str | int) -> list[np.array]:
  ''' Creates initial states for the robot arm trajectories. '''
  version = int(version)
  np.random.seed(42)
  if version == 12:
    return [np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.0, 0.0]) for _ in range(1008)]
  elif version == 20:
    return [np.array([np.random.uniform(0, 0.1), np.random.uniform(0, 0.1), 0.0, 0.0]) for _ in range(1008)]
  elif version == 30:
    return [
        np.array(
            [np.random.uniform(-5, 5),
             np.random.uniform(-5, 5),
             np.random.uniform(-5, 5),
             np.random.uniform(-5, 5)]) for _ in range(1008)
    ]
