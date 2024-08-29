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
    # -------- SPECIFY CONTROLLER COMPONENTS FROM HERE...
    memos = ['ofp_level']  # names of memory entries, updated with every measurement
    dynvars = ['inducer_level']  # names of dynamic variables, simulated using ODEs
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

    return default_par, default_init_conds, default_init_memo, memos, dynvars, name2pos, controller_styles

# control action
def cci_action(t,x, # simulation time, system state
               ctrl_memo, # controller memory
               par, # system parameters
               modules_name2pos, # genetic module name to position decoder
               controller_name2pos # controller name to position decoder
               ):
    # constant inducer concentration
    u=x[controller_name2pos['inducer_level']]
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
    return [0.0]

# update controller memory based on measurements
def cci_update(t, x, # time, cell state
               ctrl_memo, # controller memory
               par, # system parameters
               modules_name2pos, # genetic module name to position decoder
               controller_name2pos # controller name to position decoder
               ):
    return jnp.array([
        # fluorescence measurement
        x[modules_name2pos['p_ofp']]
    ])