import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt


def animate_robot_arm(x1: np.array,
                      y1: np.array,
                      x2: np.array,
                      y2: np.array,
                      total_time: float = 30,
                      dt: float = 0.05) -> animation.FuncAnimation:
  ''' Animates the robot arm using the provided coordinates. '''
  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
  ax.set_aspect('equal')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.grid()
  line, = ax.plot([], [], 'o-', lw=2)
  time_template = 'time = %.1fs'
  time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
  frames = len(x1)
  interval = (total_time * 1000) / frames

  def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    return line, time_text

  return animation.FuncAnimation(fig, animate, len(x1), interval=interval, blit=True, repeat=False)


def plot_trajectory(y_analytic: np.ndarray, y_simulated: np.ndarray, time_points: np.ndarray, version: str) -> None:
  ''' Plots the analytic and simulated trajectories of the robot arm for each component of the state vector.
      The initial conditions are automatically extracted from the first entry of the trajectories.

      Parameters
      ----------
      y_analytic: Analytic trajectories, shape [time_steps, state_variables].
      y_simulated: Simulated trajectories, shape [time_steps, state_variables].
      time_points: Time points, shape [time_steps].
  '''
  initial_conditions = y_analytic[0]
  initial_conditions_str = f'[θ1(0)={initial_conditions[0]:.2f}, θ2(0)={initial_conditions[1]:.2f}, ω1(0)={initial_conditions[2]:.2f}, ω2(0)={initial_conditions[3]:.2f}]'
  num_state_variables = y_analytic.shape[1]  # Number of state variables (e.g., 4 for θ1, θ2, ω1, ω2)
  state_labels = [f'θ1', f'θ2', f'ω1', f'ω2']  # Customize as per your state variables
  fig, axs = plt.subplots(num_state_variables, 1, figsize=(12, 7), sharex=True)
  fig.suptitle(f'Analytic vs Simulated Trajectories\nInitial Conditions: {initial_conditions_str}, Model v{version}',
               fontsize=14)
  for i in range(num_state_variables):
    axs[i].plot(time_points, y_analytic[:, i], label=f'Analytic {state_labels[i]}', linestyle='-', color='blue')
    axs[i].plot(time_points, y_simulated[:, i], label=f'Simulated {state_labels[i]}', linestyle='--', color='orange')
    axs[i].set_ylabel(state_labels[i])
    axs[i].set_xlabel('Time (s)')
    axs[i].grid(True)
    axs[i].legend()
  axs[-1].set_xlabel('Time (s)')
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()


def plot_simulated_trajectory(data: np.ndarray, time_points: np.ndarray):
  ''' Plots the simulated trajectory of the robot arm for each component of the state vector.
  
    Parameters
    ----------
    simulated_trajectory: Simulated trajectories, shape [time_steps, state_variables].
    time_points: Time points, shape [time_steps].
  '''
  num_state_variables = data.shape[1]
  state_labels = [f'θ1 (grad)', f'θ2 (grad)', f'ω1 (grad/s)', f'ω2 (grad/s)']
  global_min = np.min(data)
  global_max = np.max(data)
  fig, axs = plt.subplots(num_state_variables, 1, figsize=(10, 6), sharex=True)
  fig.suptitle('Trajectories', fontsize=14)
  for i in range(num_state_variables):
    axs[i].plot(time_points, data[:, i], label=f'{state_labels[i]}', linestyle='-', color='blue')
    axs[i].set_ylabel(state_labels[i])
    axs[i].set_xlabel('Time (s)')
    axs[i].grid(True)
    axs[i].legend()
    axs[i].set_ylim(global_min, global_max)

  axs[-1].set_xlabel('Time (s)')
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()


def plot_loss_curves(losses: list[list[float]], epochs: int, labels: list[str], title: str):
  if len(labels) != len(losses):
    raise ValueError('Number of labels must match the number of loss curves.')
  plt.figure(figsize=(10, 6))
  for i, loss in enumerate(losses):
    if len(loss) < epochs:
      loss = loss + [np.nan] * (epochs - len(loss))
    elif len(loss) > epochs:
      loss = loss[:epochs]
    plt.plot(range(epochs), loss, label=labels[i])
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.show()


def plot_training_loss(training_losses: list[list[float]], epochs: int, labels: list[str]):
  plot_loss_curves(training_losses, epochs, labels, 'Training Loss Over Epochs')


def plot_evaluation_loss(evaluation_losses: list[list[float]], epochs: int, labels: list[str]):
  plot_loss_curves(evaluation_losses, epochs, labels, 'Evaluation Loss Over Epochs')


def plot_outer_points(trajectories: list[list[np.array]], robot_arm, group_labels: list):
  plt.figure(figsize=(10, 5))
  for i, trajectory in enumerate(trajectories):
    theta1, theta2 = trajectory[:, 0], trajectory[:, 1]
    x2 = robot_arm.l1 * np.sin(theta1) + robot_arm.l2 * np.sin(theta2)
    y2 = -robot_arm.l1 * np.cos(theta1) - robot_arm.l2 * np.cos(theta2)
    plt.plot(x2, y2, label=group_labels[i])
  plt.xlabel('x in m')
  plt.ylabel('y in m')
  plt.title('Outer Link l2 Trajectories (Grouped)')
  plt.legend()
  plt.grid(True)
  plt.show()


def plot_energies(energies: list[np.array], time_points: np.array, labels: list, version: str) -> None:
  plt.figure(figsize=(10, 6))
  max_energy = 0
  for idx, energy_values in enumerate(energies):
    plt.plot(time_points, energy_values, label=labels[idx])
    max_energy = max(max_energy, max(energy_values))
  plt.xlabel('Time (s)')
  plt.ylabel('Energy (J)')
  plt.title(f'Energy Over Time for Multiple Trajectories with Model v{version}')
  plt.ylim(max_energy - 10, max_energy + 10)
  plt.legend()
  plt.grid(True)
  plt.show()
