The goal of this project is to develop a comprehensive simulation of a robotic arm using a Physics-Based Machine Learning (PBML) approach. The simulation should effectively model the physical behavior of the robotic arm, incorporating a variety of disturbing factors to create a realistic environment for evaluating control algorithms and performance.
In order to achieve this goal, the robot arm is modeled using a Lagrangian formulation, which describes the dynamics of the system in terms of generalized coordinates and forces conditions.

# Setup
In order to run the [notebook](notebook/robot_arm.ipynb) you need to setup the virtual environment and install the required packages. 
You can do this by running the following commands from the **root** of the repository:
```bash
python -m venv venv
venv\Scripts\activate
pip install -e .
```
In order to run torch with CUDA support you need to install the appropriate version of [torch](https://pytorch.org/get-started/locally).
```bash
pip3 install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu<version>
```
## Documentation
[Robot Arm](/robot_arm.pdf)
