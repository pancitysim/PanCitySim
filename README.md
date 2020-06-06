# PanCitySim
This repository contains a demo version of the codes used for the paper "Activity-based mechanistic model for high-resolution epidemic tracking and control in urban areas"


## Using colab(Recommended)
If the user wishes to use `google colab`, we have created a copy of the demo notebook in colab format. We recommend this method for quick testing of the script as the requirements are built-in and directory structure is already taken care of. The steps required to run the demo script using colab version are as following:
* open `demo_script_using_google_colab_light_version.ipynb` in browser; click on `open in colab`
* upload the `demo_dataset_30K_individuals.zip` file into the colab runtime environment
* run each cell of the notebook starting from top; output files and plots will be generated in the `ouput_files` folder. The output plots can also be previewed in the notebook

## Requirements 
The amount of dependencies has been minimised to make PanCitySim easily runnable. The script was tested using python3.6.9  and ipythonv7.13.0. Listed below are the packages that should be installed in order to run the accompanying ipython file (`demo_script.ipynb`) successfully. The packages can be installed using *pip* or *conda*(id using a conda environment). The links for installation of individual packages are provided below. For convenience, the pip commands for installing these have also been incorporated in the first cell of `demo_script.ipynb`.
### [matplotlib](https://matplotlib.org/3.1.1/users/installing.html)
### [numpy](https://pypi.org/project/numpy/)
### [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
### [altair](https://altair-viz.github.io/getting_started/installation.html)


## Directory tree
In order to run the script, the following directory strucure should be created in the working directory. The demo data is present in the a zipped folder called `demo_dataset_30K_individuals.zip`. When unzipped, this folder already contains the folders organised as shown below:

.
 * Working directory
   * demo_script.ipynb
   * output_files[directory]
   * SAMPLE_POPULATION  [directory]
     * running_statevectors [directory]
     * HOME_dicts [directory]
     * UNION_dicts [directory]
     * PT_dicts [directory]
     * ACT_dicts [directory]
 
Once the directory structure has been put into place, the entire `demo_script.ipynb` notebook can be run. The contact graphs are saved in the directories `Home_dicts`, `PT_dicts`, `ACT_dicts` and `UNION_dicts`. The contact graphs are later simulated to study the evolution of the epidemic. The last part of the notebook processes the outputs of the simulation and generates vital statistics which help us understand the spread of the epidemic. Plots are generated for analysis of the properties of the contact graph as well. After a succesful completion of the runs, the outputs plots are present in the `output_files` folder. The notebook takes as input the output files from the supply simulator of [SimMobility](https://github.com/smart-fm/simmobility-prod) and creates a contact network based on the trajectories. The demo data contains the relevant files from SimMobility output. We are working to release a version which is capable of working with a more generic format of input files so that this `PanCitySim` framework can be readily incorporated into other transport simulators as well. 

## Runtime
The entire dataset is too large to be incorporated here. We are trying to upload them to a file server. The link for the same will be provided here in the next few days. For demonstration purposes, we have provided a sub-sample of the population comprising 30K individuals. The run time for 30K indiviuals run time ~10 minutes using single thread for all processes on a machine running Ubuntu18.04.4 LTS. The RAM usage for demo data is less than 6GB. 

Processing of huge graphs is a memory-hungry operation. If the amount of RAM available is not sufficient, the Graphs can be loaded from the disk during the course of simulation. This can be tweaked by setting the `load_graphs_in_RAM` to `False`. The parallel loading of Graphs was tested but is not incorporated in the current version of the notebook. These are present in the `parallel-loading-of-Graphs` branch. A flowchart showing the parallelisation of loading of contact graphs is shown [here](https://user-images.githubusercontent.com/9101260/83565330-cfc54480-a550-11ea-87ac-7c00cdf17622.png)
