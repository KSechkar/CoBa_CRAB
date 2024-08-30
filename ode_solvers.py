'''
CONTROLLERS.PY: Different ode solvers for simulation
'''


# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, Kvaerno3, ODETerm, SaveAt, PIDController

# DIFFRAX ODE SOLVER (NO SUPPORT FOR CONTROL ACTION DELAY!) ------------------------------------------------------------
# create a diffrax solver with required characteristics
def create_diffrax_solver(odeuus_complete,
                          control_delay,  # delay between control input calculation and its exertion on the system
                          meastimestep,  # measurement time step
                          rtol, atol,  # relative and absolute integration tolerances
                          solver_spec='Kvaerno3'  # ODE solver
                          ):
    # warn that the solver does not support control delay
    if(control_delay!=0):
        raise ValueError('Diffrax solver does not support control delay')

    # get the specified diffrax ODE solver
    if solver_spec == 'Kvaerno3':
        solver = Kvaerno3()
    else:
        raise ValueError('Specified Diffrax solver not supported')

    # define ODE term
    vector_field = lambda t, x, args: odeuus_complete(t,     # time
                                                     x,     # state vector
                                                     jnp.array([]),  # no control action record as zero control delay
                                                     args   # other arguments
                                                     )[0]
    term = ODETerm(vector_field)

    # define the time points at which we save the solution
    stepsize_controller = PIDController(rtol=rtol, atol=atol)

    # define ODE solver
    ode_solver = lambda t0, x0, u0, us0, args: run_diffrax(
        odeuus_complete=odeuus_complete,
        term=term,
        solver=solver,
        args=args,
        t0=t0, meastimestep=meastimestep,
        x0=x0,
        stepsize_controller=stepsize_controller)
    
    return (ode_solver, # return the ODE solver
            0)  # return the required control action memory size (0 for no control action delay)

# auxiliary: run diffrax solver and get the end-state and control action experienced in the end
def run_diffrax(odeuus_complete,  # complete ODE and control action calculation function
                term,
                solver,
                args,
                t0, meastimestep,
                x0, 
                stepsize_controller):
    # get next state vector by ODE integration
    next_x = diffeqsolve(
                    term,
                    solver,
                    args=args,
                    t0=t0, t1=t0+meastimestep,
                    dt0=0.1, y0=x0,
                    max_steps=None,
                    stepsize_controller=stepsize_controller).ys[-1]
    
    # get next experienced (as well as calculated, since there is no control action delay) control action
    next_u=odeuus_complete(t0+meastimestep, next_x, jnp.array([]), args)[1]
    
    # no calculated control actions recorded since there is no control action delay
    next_us = jnp.array([])
    
    # return
    return (next_x, next_u, next_us)


# EULER SOLVER (SUPPORTS CONTROL ACTION DELAY) -------------------------------------------------------------------------
# create an Euler solver with required characteristics
def create_euler_solver(odeuus_complete,  # complete ODE function
                        control_delay,  # delay between control action calculation and its exertion on the system]
                        meastimestep,  # measurement time step
                        euler_timestep  # Euler time step
                        ):
    # find how many Euler steps are needed to cover the measurement time step
    euler_steps_in_meastimestep = int((meastimestep+euler_timestep/2) / euler_timestep)

    # find the number of control actions to store in memory
    us_size = int((control_delay+euler_timestep/2) / euler_timestep)

    # define the Euler step as a function of step counter and (t, x, us, args) only
    euler_step_txuusargs = lambda step_cntr, euler_state, args: euler_step(step_cntr, odeuus_complete, euler_timestep, euler_state, args)

    # define the Euler loop as a function of (t0, x0, us0, args) only
    run_euler_txuus = lambda t0, x0, u0, us0, args: run_euler(euler_step_txuusargs, euler_steps_in_meastimestep, t0, x0, u0, us0, args)

    return (run_euler_txuus, # return the Euler loop function
            us_size)    # return the required control action memory size


# auxiliary: Euler integration loop, returning next state and control actions record
def run_euler(euler_step_txuusargs,  # Euler step function
                  euler_steps_in_meastimestep,  # number of Euler steps in the measurement time step
                  t0, x0, u0, us0, args  # initial time, state vector, control actions record, and arguments
                  ):
    next_txuus = jax.lax.fori_loop(0, euler_steps_in_meastimestep,  # loop over Euler steps
                                      lambda step_cntr, euler_state: euler_step_txuusargs(step_cntr, euler_state, args),  # Euler step function
                                      {'t': t0, 'x': x0, 'u': u0, 'us': us0}  # initial condition and arguments
                                      )
    # return
    return (next_txuus['x'],    # next state vector
            next_txuus['u'],    # next experienced control action
            next_txuus['us'])   # record of calculated control actions


# auxiliary: Euler step function
def euler_step(step_cntr,  # step counter (needed to run the fori_loop)
               odeuus_complete,  # complete ODE and control action calculation function
               euler_timestep,
               euler_state,  # time, state vector, control inputs - from one calculated control_delay hours ago to most recent calculation
               args
               ):
    # unpack the tuple
    t=euler_state['t']
    x=euler_state['x']
    us=euler_state['us']

    # get the ODE value, control action experienced at the time and the updated control actions record
    dxdt, u_next, us_next = odeuus_complete(t, x,  # time and state vector
                                           us,  # calculated control actions record
                                           args)  # other ODE arguments and control actions record

    # get the next state vector
    x_next = x + euler_timestep * dxdt

    # get the next time
    t_next = t + euler_timestep

    return {'t': t_next, 'x': x_next, 'u':u_next, 'us': us_next}  # return the next time, state vector, control actions record, and arguments
