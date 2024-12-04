# MPC_CONTROLLER.PY - MPC controller implementation

import jax
import jax.numpy as jnp
import numpy as np
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, io as bkio
from bokeh.colors import RGB as bkRGB

import time

# set up jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# set up bokeh
bkio.reset_output()
bkplot.output_notebook()


# MODEL SIMULATION FOR OPTIMISATION ------------------------------------------------------------------------------------
# simplified model of the system used in MPC
def mpc_sys_model(t, x,  # time, cell state
                  u,  # input to the system
                  par,  # system parameters
                  ):
    # unpack the state vector: self-activating switch
    p_s = x[0]  # switch protein concentration
    p_ofp = x[1]  # immature output fluorescence protein concentration
    ofp_mature = x[2]  # mature output fluorescence protein concentration
    # unpack the state vector: probe
    p_ta = x[3]  # transcriiption activator protein concentration
    p_b = x[4]  # gene b mRNA concentration
    # unpack the state vector: ribosomes
    R = x[5]  # ribosome count in the cell

    # get the transcription regulation function for the self-activating switch
    p_switch_term = p_s * par['mpc_I_switch'] / par['mpc_K_switch']
    F_s = par['mpc_baseline_switch'] + (1 - par['mpc_baseline_switch']) * (p_switch_term ** par['mpc_eta_switch']) / (
                p_switch_term ** par['mpc_eta_switch'] + 1)

    # get the transcription regulation function for the probe
    tai_conc = p_ta * u / (u + par['mpc_K_ta-i']) # concentration of the transcription activator protein
    F_b = par['mpc_baseline_tai-dna'] + (1 - par['mpc_baseline_tai-dna']) * (tai_conc ** par['mpc_eta_tai-dna']) / (
                tai_conc ** par['mpc_eta_tai-dna'] + par['mpc_K_tai-dna'] ** par['mpc_eta_tai-dna'])

    # get total resource competition
    Q = F_s*(par['mpc_q_s']+par['mpc_q_ofp']) + par['mpc_q_ta'] + F_b*par['mpc_q_b']
    # get total growth rate slowdown
    J = F_s*(par['mpc_j_s']+par['mpc_j_ofp']) + par['mpc_j_ta'] + F_b*par['mpc_j_b']

    # get the slowed-down growth rate
    l = par['mpc_l0'] / (1+J)

    dxdt=[
        # self-activating switch
        # switch protein
        par['mpc_v_s'] * (F_s*par['mpc_q_s']/(1+Q)) - par['mpc_deg_s'] * p_s - l * p_s,
        # immature output fluorescence protein
        par['mpc_v_ofp'] * (F_s*par['mpc_q_ofp']/(1+Q)) - par['mpc_deg_ofp'] * p_ofp - l * p_ofp - par['mpc_mu_ofp'] * p_ofp,
        # mature output fluorescence protein
        par['mpc_mu_ofp'] * p_ofp - par['mpc_deg_ofp_mature'] * ofp_mature - l * ofp_mature,
        # probe
        # transcription activator protein
        par['mpc_v_ta'] * (par['mpc_q_ta']/(1+Q)) - par['mpc_deg_ta'] * p_ta - l * p_ta,
        # gene b mRNA
        par['mpc_v_b'] * (F_b*par['mpc_q_b']/(1+Q)) - par['mpc_deg_b'] * p_b - l * p_b
    ]

    return jnp.array(dxdt)


def simulate_mpc_sys_model(x0,  # initial condition
                           ref,  # reference to be tracked
                           u0,  # original input (not subject to the current MPC iteration)
                           us,  # sequence of control inputs
                           control_delay,  # control delay
                           meastimestep,  # measurement time step
                           horizon_steps,  # prediction horizon in measurement time steps
                           dt,  # Euler time step (fits an integer number of times in the measurement time step)
                           par,  # system parameters
                           ):
    # define the arguments for finding the next state vector
    args = (dt, par, u0, us, control_delay, meastimestep, horizon_steps)

    # time points at which we save the solution
    ts = jnp.arange(0,meastimestep*(horizon_steps+0.5), meastimestep)

    # find the number of Euler steps in the measurement time step
    dts_in_meastimestep = int((meastimestep+dt/2) / dt)

    # make the retrieval of next simulator state a lambda-function for jax.lax.scanning
    scan_step = lambda sim_state, t: sim_step_mpc(sim_state, t,
                                                  args, dts_in_meastimestep)
    # define the jac.lax.scan function
    ode_scan_mpc = lambda sim_state_rec0, ts: jax.lax.scan(scan_step, sim_state_rec0, ts)
    ode_scan_mpc_jit = jax.jit(ode_scan_mpc)

    # initalise the simulator state: (t, x)
    sim_state = {'t': 0, 'x': x0}

    # run the simulation
    _, sim_outcome = ode_scan_mpc_jit(sim_state, ts)

    # unpack the simulation outcome
    xs=sim_outcome[1]   # system states
    u_exps=sim_outcome[2]   # control actions experienced by the system

    return ts, xs, u_exps

# one step of the MPC ode simulation loop (from one measurement to the next)
def sim_step_mpc(sim_state, t,
                 args,
                 dts_in_meastimestep,  # number of Euler steps in the measurement time step
                 ):
    # unpack the arguments
    dt = args[0]
    par = args[1]
    u0 = args[2]
    us = args[3]
    control_delay = args[4]
    meastimestep = args[5]
    horizon_steps = args[6]

    # get the input experienced at this time step
    u_exp=u_experienced_mpc(t, u0, us, control_delay, meastimestep, horizon_steps)

    # get the next measurement time
    t_next = t + meastimestep

    # get the next state vector
    x_next=jax.lax.fori_loop(0, dts_in_meastimestep,  # loop over Euler steps
                                lambda step_cntr, euler_state: euler_step_mpc(step_cntr, euler_state, args),  # Euler step function
                                {'t': t, 'x': sim_state['x']}  # initial condition
                                )['x']
    
    # get the overall next simulation state
    next_sim_state = {'t': t_next, 'x': x_next}
    
    return next_sim_state, (t, sim_state['x'], u_exp)


# one step of the Eulelr simulation for MPC
def euler_step_mpc(step_cntr, euler_state, args):
    # unpack the simulation state
    t = euler_state['t']
    x = euler_state['x']
    # unpack the arguments
    dt=args[0]
    par=args[1]
    u0=args[2]
    us=args[3]
    control_delay=args[4]
    meastimestep=args[5]
    horizon_steps=args[6]

    # get the experienced control action
    u_exp = u_experienced_mpc(t, u0, us, control_delay, meastimestep, horizon_steps)

    # get the ODE value
    dxdt = mpc_sys_model(t, x, u_exp, par)

    # get the next state vector
    x_next = x + dt * dxdt

    # get the next time
    t_next = t + dt

    return {'t': t_next, 'x': x_next}  # return the next time and state vector

# get the control action experienced at a given time
def u_experienced_mpc(t,  # current time
                      u0, us,
                      # initial control action, vector of control actions to be applied over the prediction horizon
                      control_delay,  # control delay
                      meastimestep,  # measurement time step
                      horizon_steps,  # prediction horizon (in number of measurement time steps)
                      ):
    # find the times at which the control action EXPERIENCED by the system is changed
    u_times = jnp.concatenate((jnp.array([0]), jnp.arange(control_delay, control_delay + meastimestep * (horizon_steps+0.5), meastimestep)))
    first_u_times=u_times[:-1]
    last_u_times=u_times[1:]
    u_array = jnp.concatenate((jnp.array([u0]), us))

    # find the control actions EXPERIENCED by the system (below, all but the right action are multiplied by 0)
    u_exp = jnp.sum(jnp.multiply(u_array,jnp.logical_and(t >= first_u_times, t < last_u_times)))

    # return
    return u_exp

# FUNCTIONS USED IN CBC SIMULATIONS ------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def mpc_initialise():
    # -------- SPECIFY CONTROLLER COMPONENTS FROM HERE...
    memos = ['ctrled_fp_level',
             'integral']  # names of memory entries, updated with every measurement - fluorescence of the controlled gene, integral of the error
    dynvars = []  # names of dynamic variables, simulated using ODEs
    ctrled_var = 'p_ofp_mature'  # name of the system's variabled read and steered by the controller
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in the memory vector or x (for dynvars)
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(memos)):
        name2pos[memos[i]] = i  # memory entries
    for i in range(0, len(dynvars)):
        name2pos[dynvars[
            i]] = 8 + i  # dynamic variables (name2pos will be updated once the cell's genetic modules are known)

    # default values of parameters
    default_par = {}

    # default initial memory entries
    default_init_memo = []
    for i in range(0, len(memos)):
        default_init_memo.append(0.0)

    # default initial conditions for dynamic variables
    default_init_conds = {}
    for i in range(0, len(dynvars)):
        default_init_conds[dynvars[i]] = 0.0

    # -------- DEFAULT VALUES OF CONTROLLER PARAMETERS/INITIAL CONDITIONS CAN BE SPECIFIED FROM HERE...
    default_par['mpc_Kp'] = 0.1  # proportional control gain
    default_par['mpc_Ki'] = 0.01  # integral control gain

    default_init_memo[name2pos['integral']] = 0.0  # initial integral value
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#de3163ff", '#ff6700ff', '#48d1ccff', '#bb3385ff', '#fcc200ff']
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    controller_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    # memory entry styles
    for i in range(0, len(memos)):
        controller_styles['colours'][memos[i]] = default_palette[i % len(default_palette)]
        controller_styles['dashes'][memos[i]] = default_dash[i % len(default_dash)]
    # dynamic variable styles
    for i in range(len(memos), len(memos) + len(dynvars)):
        controller_styles['colours'][dynvars[i - len(memos)]] = default_palette[i % len(default_palette)]
        controller_styles['dashes'][dynvars[i - len(memos)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS AND DASHING FOR PLOTTING FROM HERE...
    controller_styles['colours']['ofp_level'] = '#00af00ff'
    # -------- ...TO HERE

    return default_par, default_init_conds, default_init_memo, memos, dynvars, ctrled_var, name2pos, controller_styles


# control action
def mpc_action(t, x,  # simulation time, system state
                  ctrl_memo,  # controller memory
                  ref,  # currently tracked reference
                  par,  # system parameters
                  modules_name2pos,  # genetic module name to position decoder
                  controller_name2pos,  # controller name to position decoder
                  ctrled_var  # name of the system's variable read and steered by the controller
                  ):
    # value of the controlled variable
    x_ctrled = x[modules_name2pos[ctrled_var]]

    # calculate input to the system
    u_calc = par['mpc_Kp'] * (ref - x_ctrled) + par['mpc_Ki'] * ctrl_memo[controller_name2pos['integral']]

    # chemical inducer concentration must be non-negative
    u = u_calc * (u_calc >= 0)

    # return input
    return u


# ode
def mpc_ode(t, x,  # time, cell state
               ctrl_memo,  # controller memory
               ref,  # currently tracked reference
               e, l,  # translation elongation rate, growth rate
               R,  # ribosome count in the cell, resource
               k_het, D,
               # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
               p_prot,  # synthetic protease concentration
               par,  # system parameters
               modules_name2pos,  # genetic module name to position decoder
               controller_name2pos,  # controller name to position decoder
               ctrled_var  # name of the system's variable read and steered by the controller
               ):
    # RETURN THE ODE
    return []


#  update controller memory based on measurements
def mpc_update(t, x,  # time, cell state
                  ctrl_memo,  # controller memory
                  ref,  # currently tracked reference
                  par,  # system parameters
                  modules_name2pos,  # genetic module name to position decoder
                  controller_name2pos,  # controller name to position decoder
                  ctrled_var,  # name of the system's variable read and steered by the controller
                  meastimestep  # measurement time step
                  ):
    return []

# MAIN FUNCTION (FOR TESTING ONLY) -------------------------------------------------------------------------------------
def main():
    # SET SYSTEM PARAMETERS
    par={}
    # self-activating switch expression
    par['mpc_v_s']=1.0  # maximum protein synthesis rate, nM/h
    par['mpc_q_s']=1.0  # resource competition factor, unitless
    par['mpc_j_s']=1.0  # growth rate slowdown factor, unitless
    par['mpc_deg_s']=0.0    # protein degradation rate, 1/h

    # self-activating switch transcription regulation
    par['mpc_I_switch']=1.0
    par['mpc_K_switch']=1.0
    par['mpc_baseline_switch']=0.1
    par['mpc_eta_switch']=2.0

    # immature output fluorescent protein expression
    par['mpc_v_ofp']=1.0  # maximum protein synthesis rate, nM/h
    par['mpc_q_ofp']=1.0  # resource competition factor, unitless
    par['mpc_j_ofp']=1.0  # growth rate slowdown factor, unitless
    par['mpc_deg_ofp']=0.0    # protein degradation rate, 1/h

    # output fluorescent protein maturation
    par['mpc_mu_ofp']=1.0    # maturation rate, 1/h
    par['mpc_deg_ofp_mature']=0.0    # mature protein degradation rate, 1/h

    # probe transcription activation factor expression
    par['mpc_v_ta']=1.0  # maximum protein synthesis rate, nM/h
    par['mpc_q_ta']=1.0  # resource competition factor, unitless
    par['mpc_j_ta']=1.0  # growth rate slowdown factor, unitless
    par['mpc_deg_ta']=0.0    # protein degradation rate, 1/h

    # probe's burdensome protein expression
    par['mpc_v_b']=1.0  # maximum protein synthesis rate, nM/h
    par['mpc_q_b']=1.0  # resource competition factor, unitless
    par['mpc_j_b']=1.0  # growth rate slowdown factor, unitless
    par['mpc_deg_b']=0.0    # protein degradation rate, 1/h

    # probe transcription regulation
    par['mpc_K_ta-i']=1.0
    par['mpc_K_tai-dna']=1.0
    par['mpc_baseline_tai-dna']=0.1
    par['mpc_eta_tai-dna']=2.0

    # growth rate without burden
    par['mpc_l0']=1.5    # maximum growth rate, 1/h

    # SET THE INITIAL CONDITION
    x0=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])    # initial state vector

    # SET THE REFERENCE
    ref=1.0    # reference to be tracked

    # SET THE CONTROLLER PARAMETERS
    horizon_steps=10    # prediction horizon in measurement time steps
    meastimestep=0.1    # measurement time step
    dt=1e-5    # Euler time step
    u0=2.0
    us=jnp.array([1.0]*horizon_steps)
    control_delay=0.0

    # SIMULATE THE SYSTEM
    u0=u_experienced_mpc(0,  # current time
                      u0, us,
                      # initial control action, vector of control actions to be applied over the prediction horizon
                      control_delay,  # control delay
                      meastimestep,  # measurement time step
                      horizon_steps,  # prediction horizon (in number of measurement time steps)
                      )
    # ts_jnp, xs_jnp, u_exps_jnp=simulate_mpc_sys_model(x0, ref, u0, us,
    #                                                   control_delay, meastimestep, horizon_steps, dt, par)
    # ts=np.array(ts_jnp)
    # xs=np.array(xs_jnp)
    # u_exps=np.array(u_exps_jnp)

    # PLOT THE RESULTS
    bkplot.output_file(filename="mpc_sim.html",
                       title="MPC system model Simulation")  # set up bokeh output file
    # define circuit colours
    colours = {'switch': '#48d1ccff',
               'ofp': '#bb3385ff',
               'ofp_mature': '#000000ff',
               'ta': '#de3163ff',
               'b': '#ff6700ff'}

    # protein concentrations
    prot_fig=bkplot.figure(
            frame_width=320,
            frame_height=180,
            x_axis_label="t, hours",
            y_axis_label="Protein concs, nM",
            x_range=(0,meastimestep*horizon_steps),
            title='Protein concentrations',
            tools="box_zoom,pan,hover,reset"
        )
    prot_fig.output_backend = 'svg'
    # switch protein
    prot_fig.line(ts, xs[:,0],
                color=colours['switch'], line_width=2, legend_label='ps')
    # immature output fluorescent protein
    prot_fig.line(ts, xs[:,1],
                color=colours['ofp'], line_width=2, legend_label='pofp')
    # mature output fluorescent protein
    prot_fig.line(ts, xs[:,2],
                color=colours['ofp_mature'], line_width=2, legend_label='ofp_mature')
    # probe transcription activator protein
    prot_fig.line(ts, xs[:,3],
                color=colours['ta'], line_width=2, legend_label='pta')
    # probe's burdensome protein
    prot_fig.line(ts, xs[:,4],
                color=colours['b'], line_width=2, legend_label='pb')

    # legend formatting



    # save plots
    bkplot.save(prot_fig)

    # return
    return

# main call
if __name__ == '__main__':
    main()