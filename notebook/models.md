```python
  main(version='24', trajectory_version='12')
  model = OdeNet2(features=4, latent_dim=128)
  batch_size=256

  main(version='25', trajectory_version='20')
  OdeNet2(features=4, latent_dim=128)
  batch_size=256

  main(version='26', trajectory_version='20')
  model = OdeNet3(features=4, latent_dim=128)
  batch_size=256

  main(version='27', trajectory_version='20')
  model = OdeNet2(features=4, latent_dim=256)
  batch_size=256

  main(version='28', trajectory_version='20', pertubation=True)
  model = OdeNet2(features=4, latent_dim=128)
  batch_size=256

  main(version='29', trajectory_version='12', epochs=200)
  model = OdeNet2(features=4, latent_dim=128)
  batch_size=256

  main(version='30', trajectory_version='30', epochs=200, batch_size=256, pertubation=False)
  model = OdeNet2(features=4, latent_dim=128)
  batch_size=256

  main(version='31', trajectory_version='30', epochs=400, batch_size=256, model=OdeNet2(features=4, latent_dim=256), pertubation=False)
  
  main(version='32', trajectory_version='30', epochs=400, batch_size=256, model=OdeNet2(features=4, latent_dim=128), pertubation=False)
  
  t_stop = 5
  main(version='33', init_version='20', trajectory_version='40', t_stop=t_stop, dt=t_stop / 100, epochs=200, batch_size=256, model=OdeNet3(features=4, latent_dim=128), pertubation=False)
  
  t_stop = 30
  main(version='34', init_version='20', t_stop=t_stop, dt=t_stop / 300, trajectory_version='41', epochs=400, batch_size=256, model=OdeNet3(features=4, latent_dim=128), pertubation=False)

  t_stop = 15
  main(version='35', init_version='20', trajectory_version='41', t_stop=t_stop, dt=t_stop / 100, epochs=400, batch_size=256, model=OdeNet3(features=4, latent_dim=128), pertubation=False)

  t_stop = 15
  main(version='36', init_version='20', trajectory_version='41', t_stop=t_stop, dt=t_stop / 100, epochs=400, batch_size=256, model=OdeNet3(features=4, latent_dim=128), pertubation=True)
```
