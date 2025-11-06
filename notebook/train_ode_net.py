import numpy as np
import torch
import torch.optim as optim
from torchinfo import summary
from torchdiffeq import odeint_adjoint
# local
from robot.robot_arm import create_init_states
from robot.ode_net import get_device, Trajectory, create_data_batches, train_network, OdeNet3, OdeNet2
from robot.analysis import normalize
from robot.utils import load_trajectories, save_txt_file


def main(version: str,
         t_stop: int,
         dt: float,
         init_version: str,
         trajectory_version: str,
         epochs: int,
         batch_size: int,
         model: torch.nn.Module,
         pertubation: bool = False):
  time_points = np.arange(0, t_stop, dt)
  z0 = create_init_states(version=init_version)
  file_name = f'pert_trajectories_v{trajectory_version}' if pertubation else f'trajectories_v{trajectory_version}'
  data = load_trajectories(file_name=file_name)
  data_norm = normalize(data)
  z0_norm = normalize(z0)
  len_train = (int)(0.8 * len(data_norm))
  len_test = len(data_norm) - len_train
  train_dataset = Trajectory(init_states=z0_norm[:len_train], trajectories=data_norm[:len_train])
  test_dataset = Trajectory(init_states=z0_norm[-len_test:], trajectories=data_norm[-len_test:])
  train_loader, val_loader, _ = create_data_batches(train_dataset=train_dataset,
                                                    test_dataset=test_dataset,
                                                    batch_size=batch_size)
  print(summary(model))
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  training_losses, eval_losses = train_network(model=model,
                                               data_train=train_loader,
                                               data_eval=val_loader,
                                               device=get_device(),
                                               optimizer=optimizer,
                                               time_points=time_points,
                                               epochs=epochs,
                                               integrator=odeint_adjoint,
                                               epoch_print=1,
                                               eval=True)
  print('Save model and losses')
  torch.save(model.state_dict(), f'data/models/model_adjoint_v{version}.pth')
  save_txt_file(data=training_losses, file_name=f'model_adjoint_training_losses_v{version}')
  save_txt_file(data=eval_losses, file_name=f'model_adjoint_eval_losses_v{version}')


if __name__ == '__main__':
  print('CUDA Available:', torch.cuda.is_available())
  print('Current Device:', torch.cuda.current_device())
  print('Device Name:', torch.cuda.get_device_name(torch.cuda.current_device()))
  t_stop = 5
  main(version='33', init_version='20', trajectory_version='40', t_stop=t_stop, dt=t_stop / 100, epochs=200, batch_size=256, model=OdeNet3(features=4, latent_dim=128), pertubation=False) # yapf: disable
