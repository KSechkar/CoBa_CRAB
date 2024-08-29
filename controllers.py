'''
CONTROLLERS.PY: Describing different cybergenetic controllers for the synthetic gene circuits
'''


# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

# CONSTANT CHEMICAL INPUT ----------------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def cci_initialise():
    default_par={}
    default_init_conds={}
    default_init_memory=[]
    name2pos={}

    # constant inducer concentration - zero by default
    default_par['inducer_level']=0.0

    return default_par, default_init_conds, default_init_memory, name2pos

# control action
def cci_action(t,x, # simulation time, system state
               ctrl_memo, # controller memory
               par, # system parameters
               modules_name2pos, # genetic module name to position decoder
               controller_name2pos # controller name to position decoder
               ):
    # constant inducer concentration
    u=par['inducer_level']
    return u

# ode
def cci_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # RETURN THE ODE
    return []

# update controller memory based on measurements
def cci_update(y, # measurements
               t, x, # time, cell state
               ctrl_memo # controller memory
               ):
    ctrl_memo=jnp.array([])
    return ctrl_memo