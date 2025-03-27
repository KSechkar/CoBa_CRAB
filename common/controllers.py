'''
CONTROLLERS.PY: Describing different cybergenetic controllers for the synthetic gene circuits
'''


# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import jax
import jax.numpy as jnp
import numpy as np


# CONSTANT CHEMICAL INPUT ----------------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def cci_initialise():
    # -------- SPECIFY CONTROLLER COMPONENTS FROM HERE...
    memos = ['ofp_level']  # names of memory entries, updated with every measurement
    dynvars = ['inducer_level']  # names of dynamic variables, simulated using ODEs
    ctrled_var = ''  # name of the system's variabled read and steered by the controller - here, none as no feedback
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in the memory vector or x (for dynvars)
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(memos)):
        name2pos[memos[i]] = i  # memory entries
    for i in range(0, len(dynvars)):
        name2pos[dynvars[i]] = 8 + i  # dynamic variables (name2pos will be updated once the cell's genetic modules are known)

    # default values of parameters
    default_par={}

    # default initial memory entries
    default_init_memo=[]
    for i in range(0, len(memos)):
        default_init_memo.append(0.0)

    # default initial conditions for dynamic variables
    default_init_conds = {}
    for i in range(0, len(dynvars)):
        default_init_conds[dynvars[i]] = 0.0

    # -------- DEFAULT VALUES OF CONTROLLER PARAMETERS/INITIAL CONDITIONS CAN BE SPECIFIED FROM HERE...
    # constant inducer concentration - zero by default
    # default_par['inducer_level'] = 0.0
    default_init_conds['inducer_level'] = 10.0
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
def cci_action(t,x, # simulation time, system state
               ctrl_memo, # controller memory
               ref, # currently tracked reference
               par, # system parameters
               modules_name2pos,  # genetic module name to position decoder
               controller_name2pos,  # controller name to position decoder
               ctrled_var  # name of the system's variable read and steered by the controller
               ):
    # constant inducer concentration
    u=x[controller_name2pos['inducer_level']]
    return u

# ode
def cci_ode(t,  x,  # time, cell state
            ctrl_memo, # controller memory
            ref, # currently tracked reference
            e, l, # translation elongation rate, growth rate
            R,  # ribosome count in the cell, resource
            D,  # resource competition denominator
            par,  # system parameters
            modules_name2pos,  # genetic module name to position decoder
            controller_name2pos,  # controller name to position decoder
            ctrled_var  # name of the system's variable read and steered by the controller
            ):
    # RETURN THE ODE
    return [0.0]

#  update controller memory based on measurements
def cci_update(t, x,  # time, cell state
               ctrl_memo,  # controller memory
               ref,  # currently tracked reference
               par,  # system parameters
               modules_name2pos,  # genetic module name to position decoder
               controller_name2pos,  # controller name to position decoder
               ctrled_var,  # name of the system's variable read and steered by the controller
               meastimestep  # measurement time step
               ):
    return jnp.array([
        # fluorescence measurement
        x[modules_name2pos[ctrled_var]],
    ])


# CHEMICAL INPUT IS THE REFERENCE --------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def ciref_initialise():
    # -------- SPECIFY CONTROLLER COMPONENTS FROM HERE...
    memos = ['ofp_level']  # names of memory entries, updated with every measurement
    dynvars = []  # names of dynamic variables, simulated using ODEs
    ctrled_var = ''  # name of the system's variabled read and steered by the controller - here, none as no feedback
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in the memory vector or x (for dynvars)
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(memos)):
        name2pos[memos[i]] = i  # memory entries
    for i in range(0, len(dynvars)):
        name2pos[dynvars[i]] = 8 + i  # dynamic variables (name2pos will be updated once the cell's genetic modules are known)

    # default values of parameters
    default_par={}

    # default initial memory entries
    default_init_memo=[]
    for i in range(0, len(memos)):
        default_init_memo.append(0.0)

    # default initial conditions for dynamic variables
    default_init_conds = {}
    for i in range(0, len(dynvars)):
        default_init_conds[dynvars[i]] = 0.0

    # -------- DEFAULT VALUES OF CONTROLLER PARAMETERS/INITIAL CONDITIONS CAN BE SPECIFIED FROM HERE...
    # constant inducer concentration - zero by default
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
def ciref_action(t, x,  # simulation time, system state
                 ctrl_memo,  # controller memory
                 ref,  # currently tracked reference
                 par,  # system parameters
                 modules_name2pos,  # genetic module name to position decoder
                 controller_name2pos,  # controller name to position decoder
                 ctrled_var  # name of the system's variable read and steered by the controller
               ):
    # constant inducer concentration
    u=ref
    return u

# ode
def ciref_ode(t, x,  # time, cell state
              ctrl_memo,  # controller memory
              ref,  # currently tracked reference
              e, l,  # translation elongation rate, growth rate
              R,  # ribosome count in the cell, resource
              D,  # resource competition denominator
              par,  # system parameters
              modules_name2pos,  # genetic module name to position decoder
              controller_name2pos,  # controller name to position decoder
              ctrled_var  # name of the system's variable read and steered by the controller
              ):
    # RETURN THE ODE
    return []

#  update controller memory based on measurements
def ciref_update(t, x,  # time, cell state
                 ctrl_memo,  # controller memory
                 ref,  # currently tracked reference
                 par,  # system parameters
                 modules_name2pos,  # genetic module name to position decoder
                 controller_name2pos,  # controller name to position decoder
                 ctrled_var,  # name of the system's variable read and steered by the controller
                 meastimestep  # measurement time step
                 ):
    return jnp.array([
        # fluorescence measurement
        x[modules_name2pos[ctrled_var]]
    ])


# BANG-BANG CHEMICAL CONTROL -------------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def bangbangchem_initialise():
    # -------- SPECIFY CONTROLLER COMPONENTS FROM HERE...
    memos = ['ctrled_fp_level']  # names of memory entries, updated with every measurement - fluorescence of the controlled gene, integral of the error
    dynvars = []  # names of dynamic variables, simulated using ODEs
    ctrled_var = 'p_b'  # name of the system's variabled read and steered by the controller
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
    default_par['inducer_level_on'] = 1e3  # inducer concentration (nM) when the input is ON, aka 1
    default_par['on_when_below_ref'] = True  # ON when the reference is greater than the current value - set to False for OFF in this case

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
def bangbangchem_action(t, x,  # simulation time, system state
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
    u = par['inducer_level_on'] * ((ref > x_ctrled) * par['on_when_below_ref'] + (ref < x_ctrled) * (not par['on_when_below_ref']))

    # return input
    return u


# ode
def bangbangchem_ode(t, x,  # time, cell state
                     ctrl_memo,  # controller memory
                     ref,  # currently tracked reference
                     e, l,  # translation elongation rate, growth rate
                     R,  # ribosome count in the cell, resource
                     D,  # resource competition denominator
                     par,  # system parameters
                     modules_name2pos,  # genetic module name to position decoder
                     controller_name2pos,  # controller name to position decoder
                     ctrled_var  # name of the system's variable read and steered by the controller
                     ):
    # RETURN THE ODE
    return []


#  update controller memory based on measurements
def bangbangchem_update(t, x,  # time, cell state
                  ctrl_memo,  # controller memory
                  ref,  # currently tracked reference
                  par,  # system parameters
                  modules_name2pos,  # genetic module name to position decoder
                  controller_name2pos,  # controller name to position decoder
                  ctrled_var,  # name of the system's variable read and steered by the controller
                  meastimestep  # measurement time step
                  ):
    return jnp.array([
        # fluorescence measurement
        x[modules_name2pos[ctrled_var]]
    ])


# PROPORTIONAL-INTEGRAL CHEMICAL CONTROL -------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def pichem_initialise():
    # -------- SPECIFY CONTROLLER COMPONENTS FROM HERE...
    memos = ['ctrled_fp_level', 'integral']  # names of memory entries, updated with every measurement - fluorescence of the controlled gene, integral of the error
    dynvars = []  # names of dynamic variables, simulated using ODEs
    ctrled_var = 'p_ofp_mature' # name of the system's variabled read and steered by the controller
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in the memory vector or x (for dynvars)
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(memos)):
        name2pos[memos[i]] = i  # memory entries
    for i in range(0, len(dynvars)):
        name2pos[dynvars[i]] = 8 + i  # dynamic variables (name2pos will be updated once the cell's genetic modules are known)

    # default values of parameters
    default_par={}

    # default initial memory entries
    default_init_memo=[]
    for i in range(0, len(memos)):
        default_init_memo.append(0.0)

    # default initial conditions for dynamic variables
    default_init_conds = {}
    for i in range(0, len(dynvars)):
        default_init_conds[dynvars[i]] = 0.0

    # -------- DEFAULT VALUES OF CONTROLLER PARAMETERS/INITIAL CONDITIONS CAN BE SPECIFIED FROM HERE...
    default_par['Kp'] = 0.1  # proportional control gain
    default_par['Ki'] = 0.01  # integral control gain
    
    default_init_memo[name2pos['integral']] = 0.0  # initial integral value

    # maximum possible input value
    default_par['max_I'] = 1e3
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
def pichem_action(t,x, # simulation time, system state
               ctrl_memo, # controller memory
               ref, # currently tracked reference
               par, # system parameters
               modules_name2pos, # genetic module name to position decoder
               controller_name2pos, # controller name to position decoder
               ctrled_var # name of the system's variable read and steered by the controller
               ):
    # value of the controlled variable - from the last measurement
    x_ctrled=ctrl_memo[controller_name2pos['ctrled_fp_level']]

    # calculate input to the system
    u_calc=par['Kp']*(ref-x_ctrled)+par['Ki']*ctrl_memo[controller_name2pos['integral']]
    u = jnp.clip(u_calc, 0, par['max_I'])   # clip the control action to the allowed range

    # return input
    return u

# ode
def pichem_ode(t, x,  # time, cell state
              ctrl_memo,  # controller memory
              ref,  # currently tracked reference
              e, l,  # translation elongation rate, growth rate
              R,  # ribosome count in the cell, resource
              D, # resource competition
              par,  # system parameters
              modules_name2pos,  # genetic module name to position decoder
              controller_name2pos,  # controller name to position decoder
              ctrled_var  # name of the system's variable read and steered by the controller
              ):
    # RETURN THE ODE
    return []

#  update controller memory based on measurements
def pichem_update(t, x,  # time, cell state
                 ctrl_memo,  # controller memory
                 ref,  # currently tracked reference
                 par,  # system parameters
                 modules_name2pos,  # genetic module name to position decoder
                 controller_name2pos,  # controller name to position decoder
                 ctrled_var,  # name of the system's variable read and steered by the controller
                 meastimestep  # measurement time step
                 ):
    integral=ctrl_memo[controller_name2pos['integral']] + meastimestep*(ref-x[modules_name2pos[ctrled_var]])
    return jnp.array([
        # fluorescence measurement
        x[modules_name2pos[ctrled_var]],
        # update integral term
        integral
    ])