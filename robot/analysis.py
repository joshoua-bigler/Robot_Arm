import numpy as np
import torch
# local
from robot.robot_arm import NumericRobotArm


def neural_euler_integration(model: NumericRobotArm, z0: torch.Tensor, time_points: torch.Tensor) -> torch.Tensor:
  ''' Integrate the robot arm model using Euler's method. '''
  dt = time_points[1] - time_points[0]
  z = z0
  z_trajectory = [z0]
  for t in time_points[:-1]:
    z = z + dt * model(t, z)
    z_trajectory.append(z)
  return torch.stack(z_trajectory)


def rk4_step(model: NumericRobotArm, z0: torch.Tensor, time_points: torch.Tensor) -> torch.Tensor:
  ''' Integrate the robot arm model using the fourth-order Runge-Kutta method. '''
  h = time_points[1] - time_points[0]
  t = time_points[0]
  z = z0
  z_trajectory = [z0]

  def rk4(model, z, t, h):
    k1 = h * model(t, z)
    k2 = h * model(t + h / 2, z + k1 / 2)
    k3 = h * model(t + h / 2, z + k2 / 2)
    k4 = h * model(t + h, z + k3)
    return z + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

  for t in time_points[:-1]:
    z = rk4(model, z, t, h)
    z_trajectory.append(z)
  return torch.stack(z_trajectory)


def normalize(target_trajectories: list[np.array]) -> list[np.array]:
  ''' Normalize the target trajectories 

      Parameters
      ----------
      target_trajectories:  Trajectories to normalize

      Returns
      -------
      A list containing the normalized trajectories: [θ1(t), θ2(t), dθ1/dt, dθ2/dt] where θ1, θ2 are in radians [-1, +1] and dθ1/dt, dθ2/dt are in radians [-1, +1] per second.
  '''
  normalized_trajectories = []
  for trajectory in target_trajectories:
    angles = np.radians(trajectory[:2]) if trajectory.ndim == 1 else np.radians(trajectory[:, :2])
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    max_angles = np.abs(angles).max(axis=0)
    if max_angles.ndim == 1:
      max_angles[max_angles == 0] = 1.0
    else:
      if max_angles == 0:
        max_angles = 1.0
    angles_normalized = np.clip(angles / max_angles, -1, 1)
    angular_velocities = np.radians(trajectory[2:]) if trajectory.ndim == 1 else np.radians(trajectory[:, 2:])
    angular_velocities = (angular_velocities + np.pi) % (2 * np.pi) - np.pi
    max_velocities = np.abs(angular_velocities).max(axis=0)
    if max_velocities.ndim == 1:
      max_velocities[max_velocities == 0] = 1.0
    else:
      if max_velocities == 0:
        max_velocities = 1.0
    angular_velocities_normalized = np.clip(angular_velocities / max_velocities, -1, 1)
    trajectory_normalized = np.hstack([angles_normalized, angular_velocities_normalized])
    normalized_trajectories.append(trajectory_normalized)
  return normalized_trajectories


def denormalize(normalized_trajectory: np.array, original_trajectory: np.array) -> np.array:
  ''' Denormalize a single normalized trajectory using the corresponding original trajectory.

      Parameters
      ----------
      normalized_trajectory:  The normalized trajectory to denormalize
      original_trajectory:    The original trajectory used to calculate max values

      Returns
      -------
      The denormalized trajectory: [θ1(t), θ2(t), dθ1/dt, dθ2/dt] in their original ranges.
  '''
  # Calculate max_angles and max_velocities from the original trajectory
  angles = np.radians(original_trajectory[:2]) if original_trajectory.ndim == 1 else np.radians(
      original_trajectory[:, :2])
  angles = (angles + np.pi) % (2 * np.pi) - np.pi
  max_angles = np.abs(angles).max(axis=0)
  max_angles[max_angles == 0] = 1.0  # Avoid division by zero
  angular_velocities = np.radians(original_trajectory[2:]) if original_trajectory.ndim == 1 else np.radians(
      original_trajectory[:, 2:])
  angular_velocities = (angular_velocities + np.pi) % (2 * np.pi) - np.pi
  max_velocities = np.abs(angular_velocities).max(axis=0)
  max_velocities[max_velocities == 0] = 1.0  # Avoid division by zero
  # Split the normalized trajectory
  angles_normalized = normalized_trajectory[:2] if normalized_trajectory.ndim == 1 else normalized_trajectory[:, :2]
  velocities_normalized = normalized_trajectory[2:] if normalized_trajectory.ndim == 1 else normalized_trajectory[:, 2:]
  # Denormalize angles and velocities
  angles_denormalized = angles_normalized * max_angles
  angles_denormalized = np.degrees(angles_denormalized)
  velocities_denormalized = velocities_normalized * max_velocities
  velocities_denormalized = np.degrees(velocities_denormalized)  #
  # Combine into a single trajectory
  if normalized_trajectory.ndim > 1:
    trajectory_denormalized = np.hstack([angles_denormalized, velocities_denormalized])
  else:
    trajectory_denormalized = np.concatenate([angles_denormalized, velocities_denormalized])
  return trajectory_denormalized


def unit_normalize(target_trajectories: list[np.array]) -> list[np.array]:
  ''' Normalize the target trajectories with zero mean and unit variance

      Parameters
      ----------
      target_trajectories:  Trajectories to normalize

      Returns
      -------
      A list containing the normalized trajectories.
    '''
  normalized_trajectories = []
  for trajectory in target_trajectories:
    # Normalize angles
    angles = np.radians(trajectory[:2]) if trajectory.ndim == 1 else np.radians(trajectory[:, :2])
    angles = (angles + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    mean_angles = angles.mean(axis=0)
    std_angles = angles.std(axis=0)
    std_angles[std_angles == 0] = 1.0  # Prevent division by zero
    angles_normalized = (angles - mean_angles) / std_angles
    # Normalize angular velocities
    angular_velocities = np.radians(trajectory[2:]) if trajectory.ndim == 1 else np.radians(trajectory[:, 2:])
    angular_velocities = (angular_velocities + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    mean_velocities = angular_velocities.mean(axis=0)
    std_velocities = angular_velocities.std(axis=0)
    std_velocities[std_velocities == 0] = 1.0  # Prevent division by zero
    angular_velocities_normalized = (angular_velocities - mean_velocities) / std_velocities
    # Combine normalized angles and velocities
    trajectory_normalized = np.hstack([angles_normalized, angular_velocities_normalized])
    normalized_trajectories.append(trajectory_normalized)
  return normalized_trajectories
