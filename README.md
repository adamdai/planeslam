# planeslam

LiDAR-based Simultaneous Localization and Mapping using Plane Features and Maps

Paper: https://arxiv.org/abs/2209.08248

## Setup
For Windows, may not work on other OSes.

Clone the GitHub repository:

    git clone https://github.com/adamdai/planeslam.git

Create conda environment from `.yml` file:

    conda env create -f environment.yml
    
(Alternatively, if not on Windows, create your own environement, and pip install the dependencies manually):

    conda create -n planeslam python=3.7
    pip install numpy scipy ipykernel ipympl plotly pandas open3d graphslam (matplotlib line_profiler)
    

Active the environment:
   
    conda activate planeslam
   
Install `planeslam` locally from directory containing `setup.py`
   
    pip install -e .
    
