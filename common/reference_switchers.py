'''
REFERENCE_TRACKERS.PY: Different means of deciding when to track the next reference in the list
'''


# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import jax
import jax.numpy as jnp
import numpy as np

# general functions either doing the switching or not - used by all trackers
def switch(i_ref,  # current reference index
           t_last_ref_switch,  # time of last reference switch
           t # current time
           ):
    return (i_ref + 1, t)  # switch to the next reference

def no_switch(i_ref,  # current reference index
              t_last_ref_switch,  # time of last reference switch
              t  # current time
              ):
    return (i_ref, t_last_ref_switch)  # no switching


# NO SWITCHING BETWEEN REFERENCES --------------------------------------------------------------------------------------
# initialise
# -------- SPECIFY CONTROLLER COMPONENTS FROM HERE...
def no_switching_initialise():
    # default values of parameters
    default_par={}

    # -------- DEFAULT VALUES OF CONTROLLER PARAMETERS/INITIAL CONDITIONS CAN BE SPECIFIED FROM HERE...
    # -------- ...TO HERE

    return default_par


# switch if it's time to do so
def no_switching_switch(i_ref,  # current reference index
                        refs,  # list of references
                        t_last_ref_switch,  # time of last reference switch
                        t, x,
                        ctrl_memo,
                        par,  # system parameters
                        modules_name2pos,  # genetic module name to position decoder
                        controller_name2pos,  # controller name to position decoder
                        ctrledvar,  # name of the variable read and steered by the controller
                        meastimestep  # measurement time step
                        ):
    i_next = i_ref  # no switching
    t_last_ref_switch_next = t_last_ref_switch  # no switching
    return (i_next, t_last_ref_switch_next)

# SWITCH TO NEW REFERENCE EVERY T_SWITCH_REF HOURS ---------------------------------------------------------------------
# initialise
def timed_switching_initialise():
    # default values of parameters
    default_par={}

    # -------- DEFAULT VALUES OF CONTROLLER PARAMETERS/INITIAL CONDITIONS CAN BE SPECIFIED FROM HERE...
    # time interval between switching references
    default_par['t_switch_ref'] = 10.0 # 10 h by default
    # burn-in period for the first reference, meaning that we keep it t_burn_in hours before starting the timer
    default_par['t_burn_in'] = 0.0 # 0 h by default
    # -------- ...TO HERE

    return default_par

# switch if it's time to do so
def timed_switching_switch(i_ref,  # current reference index
                           refs,  # list of references
                           t_last_ref_switch,  # time of last reference switch
                           t, x,
                           ctrl_memo,
                           par,  # system parameters
                           modules_name2pos,  # genetic module name to position decoder
                           controller_name2pos,  # controller name to position decoder
                           ctrledvar,  # name of the variable read and steered by the controller
                           meastimestep  # measurement time step
                           ):
    # -------- DEFINE SPECIFIC SWITCHING CONDITIONS FROM HERE...
    condition=jnp.logical_or(
        jnp.logical_and(
            t_last_ref_switch <= par['t_burn_in'],
            t + meastimestep / 2 >= par['t_burn_in'] + par['t_switch_ref']
        ),
        jnp.logical_and(
            t_last_ref_switch > par['t_burn_in'],
            t + meastimestep / 2 >= t_last_ref_switch + par['t_switch_ref']
        )
    )
    # (half of the measurement time step is added to the current time to avoid rounding errors)
    # -------- ...TO HERE

    return jax.lax.cond(condition, switch, no_switch,
                        i_ref,t_last_ref_switch,t)
