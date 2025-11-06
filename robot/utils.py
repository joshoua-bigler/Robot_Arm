import numpy as np
import torch
from typing import Callable
# local
from robot.ode_net import OdeNet, OdeNet2, OdeNet3


def load_trajectories(file_name: str = 'trajectories') -> list[np.array]:
  data = np.load(f'data/trajectories/{file_name}.npz')
  trajectories = [data[key] for key in data]
  return trajectories


def save_txt_file(data: list[float], file_name: str) -> None:
  with open(f'data/training/{file_name}.txt', 'w') as f:
    for loss in data:
      f.write(f'{loss}\n')


def load_txt_file(file_name: str) -> list[float]:
  with open(f'data/training/{file_name}.txt', 'r') as f:
    training_losses = [float(line.strip()) for line in f]
  return training_losses


def create_trajectory(model: OdeNet | OdeNet2 | OdeNet3, z0: np.array, integrator: Callable,
                      time_points: np.array) -> np.array:
  model.eval()
  model.to('cpu')
  z0 = torch.tensor(z0, dtype=torch.float32)
  time_points = torch.tensor(time_points, dtype=torch.float32)
  z_trajectory = integrator(model, z0, time_points)
  return z_trajectory.squeeze().detach().numpy()
