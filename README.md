# CoBa_CRAB
Python code accompanying the paper 'Automatic feedback control for resource-aware
characterisation of genetic circuits' (Kirill Sechkar and Harrison Steel, 2025).
Namely, the code implements a simulation of applying **Co**ntrol-**Ba**sed **C**ontinuation 
for **R**esource-**A**ware **B**ifurcation analysis of synthetic gene circuit modules.

## File organisation
The repository includes the following directories with Python scripts:
- _basic_model_case_ - Python scripts implementing simulations of a basic resource-aware
gene expression model described in Section II of the paper. The files include:
  - Model parameters:
    - _basic_model_parameters.md_ - a markdown file providing descriptions, values, and motivations
    for the parameters of the basic resource-aware gene expression model.
  - Model implementation:
    - _basic_model.py_ - implementation of the host cell gene expression model.
    - _basic_genetic_modules.py_ - implementations of models for different genetic modules which 
    can be expressed by the cell.
  - Simulation scripts, implemented as Jupyter Notebooks:
    - _basic_probe_characterisation.ipynb_ - simulation of the probe module characterisation step 
    described in Section III.A of the paper.
    - _basic_selfact_cbc.ipynb_ - simulation of the module-of-interest 
    (here, a self-activating genetic switch) characterisation step 
    described in Section III.B of the paper, as well as of using the obtained data to predict
    steady-state behaviour of synthetic gene circuits as decribed in Section III.C of the paper.
    - _basic_selfact_otherind_cbc.ipynb_ - simulation of the module-of-interest characterisation
    and prediction steps from the script above for a different level of chemical induction
    applied to the module of interest.
    - _basic_selfact_constreps.ipynb_ - simulation of the module-of-interest being characterised
    using conventional constitutive reporter assays as opposed to our control-based continuation
    method.
- _cell_model_case_ - Python scripts implementing analogous simulations with a coarse-grained
resource-aware mechanistic cell model first published in [Sechkar et al. 2024](https://www.nature.com/articles/s41467-024-46410-9). 
The program files include:
  - Model parameterisation:
    - _cell_model_parameters.md_ - a markdown file providing descriptions, values, and motivations
    for the parameters of the mechanistic cell model.
  - Model implementation:
    - _cell_model.py_ - implementation of the host cell model.
    - _cell_genetic_modules.py_ - implementations of models for different genetic modules which 
    can be expressed by the cell.
  - Simulation scripts, implemented as Jupyter Notebooks:
    - _cell_probe_characterisation.ipynb_ - simulation of the probe module characterisation step 
    described in Section III.A of the paper.
    - _cell_selfact_cbc.ipynb_ - simulation of the module-of-interest 
    (here, a self-activating genetic switch) characterisation step 
    described in Section III.B of the paper, as well as of using the obtained data to predict
    steady-state behaviour of synthetic gene circuits as decribed in Section III.C of the paper.
- _common_ - Python scripts common for both implementations. The program files include:
  - Model parameterisation tools:
    - _cell_to_basic.py_ - script for simulating the mechanistic cell model so as to obtain basic model
    parameters producing similar steady-state predictions.
  - Simulation tools:
    - _controllers.py_ - implementation of cybergenetic control algorithms applied to the simulated cells.
    - _reference_switchers.py_ - scirpts allowing to switch between different references tracked by the
    cybergenetic controller as the simulation progresses.
    - _ode_solvers.py_ - auxiliary file implementing ODE solvers for simulating the models.
  - Analytical and data processing tools:
    - _probe_char_tools.py_ - data processing functions which enable the probe characterisation step.
    - _selfact_an_bif.py_ - functions enabling the analytical retrieval of a bifurcation diagram 
    for a self-activating genetic switch (example module of interest in our simulations).
    - __jointexp.py_ - data processing functions for predicting, based on
    characterisation outcomes, how modules of interest will behave when found
    together in the same host cell.

## System requirements
The code was run and ensured to be functional with Python 3.12 
on a PCs running on Windows 11 Home 24H2. 
Software requirements can be found in the file _requirements.txt_. 
All scripts can be run on a normal PC CPU in under 5 minutes.