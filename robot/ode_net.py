import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from torch.utils.data import Dataset, DataLoader, random_split


class OdeNet(nn.Module):

  def __init__(self, features: int = 4, latent_dim: int = 128):
    super().__init__()
    # yapf: disable
    self.net = nn.Sequential(nn.Linear(features, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, features))
    # yapf: enable

  def forward(self, t, z):
    return self.net(z)


class OdeNet2(nn.Module):

  def __init__(self, features: int = 4, latent_dim: int = 128):
    super().__init__()
    # yapf: disable
    self.net = nn.Sequential(nn.Linear(features, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, features))
    # yapf: enable

  def forward(self, t, z):
    return self.net(z)


class OdeNet3(nn.Module):

  def __init__(self, features: int = 4, latent_dim: int = 128):
    super().__init__()
    # yapf: disable
    self.net = nn.Sequential(nn.Linear(features, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, latent_dim),
                             nn.Tanh(),
                             nn.Linear(latent_dim, features))
    # yapf: enable

  def forward(self, t, z):
    return self.net(z)


def get_device():
  ''' Returns the device to be used for training. '''
  if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA device')

  elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using MPS device')
  else:
    print('Using CPU')
    device = torch.device('cpu')
  return device


class Trajectory(Dataset):
  ''' Dataset class for the robot arm trajectories. '''

  def __init__(self, init_states: torch.Tensor, trajectories: list[torch.Tensor]):
    ''' Initializes the dataset with the given initial states and trajectories.

        Parameters
        ----------
        init_states:  Tensor of initial states
        trajectories: List of tensors, each containing the trajectory data for one init state.
      '''
    self.init_states = torch.tensor(init_states, dtype=torch.float32)
    self.trajectories = torch.tensor(trajectories, dtype=torch.float32)

  def __len__(self):
    return len(self.init_states)

  def __getitem__(self, idx):
    return self.init_states[idx], self.trajectories[idx]


def create_data_batches(train_dataset: Dataset, test_dataset: Dataset,
                        batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
  ''' Creates data loaders for training, validation, and testing using the Trajectory class.

      Parameters
      ----------
      train_dataset:  Dataset object for training
      test_dataset:   Dataset object for testing
      batch_size:     Batch size for the data loaders

      Returns
      -------
      Tuple of DataLoader objects for train, validation, and test sets.
  '''
  len_train = int(0.8 * len(train_dataset))
  len_val = len(train_dataset) - len_train
  train_subset, val_subset = random_split(train_dataset, [len_train, len_val])
  train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
  val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, val_loader, test_loader


def train_network(model: OdeNet,
                  data_train: torch.utils.data.DataLoader,
                  data_eval: torch.utils.data.DataLoader,
                  device: str,
                  optimizer: optim.Optimizer,
                  time_points: np.array,
                  epochs: int,
                  integrator: odeint,
                  method: str = None,
                  epoch_print: int = 10,
                  eval: bool = True) -> tuple[list[float], list[float]]:
  ''' Trains the neural network model using the given data loaders.

      Parameters
      ----------
      model:                Neural network model
      data_train:           DataLoader object for training data
      data_eval:            DataLoader object for evaluation data
      device:               Device to be used for training
      optimizer:            Optimizer object
      time_points:          Array of time points
      epochs:               Number of epochs
      integrator:           ODE integrator
      method:               ODE solver method
      epoch_print:          Number of epochs after which to print the loss
      eval:                 Flag to evaluate the model
      
      Returns
      -------
      Tuple of training and evaluation losses.
  '''
  model.to(device)
  time_points = torch.tensor(time_points, dtype=torch.float32).to(device)
  criterion = torch.nn.MSELoss()
  training_losses = []
  eval_losses = []
  for epoch in range(epochs):
    model.train(True)
    epoch_train_loss = 0
    for _, (z0, data) in enumerate(data_train):
      z0 = z0.to(device, dtype=torch.float32)
      data = data.to(device, dtype=torch.float32)
      optimizer.zero_grad()
      outputs = integrator(model, z0.float(), time_points.float())
      outputs = outputs.squeeze()
      data = data.squeeze()
      if outputs.shape != data.shape:
        outputs = outputs.squeeze(1).permute(1, 0, 2)
      time_loss = criterion(outputs.squeeze(), data)
      # add spectral loss
      fft_outputs = torch.fft.rfft(outputs, dim=1)  # FFT along the time dimension
      fft_data = torch.fft.rfft(data, dim=1)
      magnitude_outputs = torch.abs(fft_outputs)
      magnitude_data = torch.abs(fft_data)
      spectral_loss = criterion(magnitude_outputs, magnitude_data)
      train_loss = spectral_loss + time_loss
      train_loss.backward()
      optimizer.step()
      epoch_train_loss += train_loss.item()
    avg_train_loss = epoch_train_loss / len(data_train)
    training_losses.append(avg_train_loss)
    if epoch % epoch_print == 0:
      print(f'Epoch: {epoch:03d}, Loss: {train_loss.item():.4f}')
    if eval:
      model.eval()
      epoch_eval_loss = 0
      for _, (z0, data) in enumerate(data_eval):
        z0 = z0.to(device, dtype=torch.float32)
        data = data.to(device, dtype=torch.float32)
        with torch.no_grad():
          outputs = odeint(model, z0.float(), time_points.float(), method=method if method else None)
          outputs = outputs.squeeze()
          if outputs.shape != data.shape:
            outputs = outputs.squeeze(1).permute(1, 0, 2)
          time_loss = criterion(outputs.squeeze(), data)
          # add spectral loss
          fft_outputs = torch.fft.rfft(outputs, dim=1)  # FFT along the time dimension
          fft_data = torch.fft.rfft(data, dim=1)
          magnitude_outputs = torch.abs(fft_outputs)
          magnitude_data = torch.abs(fft_data)
          spectral_loss = criterion(magnitude_outputs, magnitude_data)
          batch_eval_loss = spectral_loss + time_loss
          epoch_eval_loss += batch_eval_loss.item()
      avg_eval_loss = epoch_eval_loss / len(data_eval)
      eval_losses.append(avg_eval_loss)
  return training_losses, eval_losses
