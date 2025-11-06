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