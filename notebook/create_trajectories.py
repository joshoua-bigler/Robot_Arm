import numpy as np
# local
from robot.robot_arm import NumericRobotArm, create_init_states, numeric_robot_arm, process_trajectories

robot_arm = NumericRobotArm(m1=10.0, m2=10.0, l1=1, l2=1, mu1=1, mu2=1)


def create_trajectories(z0: list,
                        time_points: np.array,
                        dt: float,
                        save: bool = True,
                        file_name: str = 'trajectories') -> list[np.array]:
  ''' Creates the trajectories for the robot arm. '''
  trajectories = [numeric_robot_arm(robot_arm=robot_arm, time_points=time_points, z0=z, dt=dt) for z in z0]
  if save:
    np.savez(f'data/trajectories/{file_name}.npz', *trajectories)
  return trajectories


def create_trajectories_with_pertubation(z0: list,
                                         time_points: np.array,
                                         dt: float,
                                         save: bool = True,
                                         file_name: str = 'trajectories',
                                         pertubation_variance: float = 0.01) -> list[np.array]:
  ''' Creates the trajectories for the robot arm. '''
  trajectories = [
      numeric_robot_arm(robot_arm=robot_arm,
                        time_points=time_points,
                        z0=z,
                        dt=dt,
                        pertubation_variance=pertubation_variance) for z in z0
  ]
  if save:
    np.savez(f'data/trajectories/{file_name}.npz', *trajectories)
  return trajectories


def main(t_stop: int, dt: float, init_version: str, trajectory_version: str, pertubation: bool = False):
  time_points = np.arange(0, t_stop, dt)
  z0 = create_init_states(version=init_version)
  process_trajectories(
      z0=z0,
      time_points=time_points,
      dt=dt,
      trajectory_version=trajectory_version,
      create_trajectories=create_trajectories if not pertubation else create_trajectories_with_pertubation,
      pertubation=pertubation)


if __name__ == '__main__':
  t_stop = 15
  main(t_stop=t_stop, dt=t_stop / 100, init_version='20', trajectory_version='41', pertubation=True)
