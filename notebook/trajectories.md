````python
t_stop = 5
main(t_stop=t_stop, dt=t_stop / 30, init_version='30', trajectory_version='30', pertubation=False)

t_stop = 5
main(t_stop=t_stop, dt=t_stop / 100, init_version='20', trajectory_version='40', pertubation=False)

t_stop = 15
main(t_stop=t_stop, dt=t_stop / 100, init_version='20', trajectory_version='41', pertubation=False)

t_stop = 30
main(t_stop=t_stop, dt=t_stop / 300, init_version='20', trajectory_version='42', pertubation=False)
```