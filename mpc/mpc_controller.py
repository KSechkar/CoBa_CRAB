# MPC_CONTROLLER.PY - MPC controller implementation

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

    # get the transcription regulation function for the self-activating switch
    p_switch_term = p_s * par['I_switch2'] / par['K_switch2']
    F_s = par['baseline_switch2'] + (1 - par['baseline_switch2']) * (p_switch_term ** par['eta_switch2']) / (
                p_switch_term ** par['eta_switch2'] + 1)

    # get the transcription regulation function for the probe
    tai_conc = p_ta * u / (u + par['K_ta-i']) # concentration of the transcription activator protein
    F_b = par['baseline_tai-dna'] + (1 - par['baseline_tai-dna']) * (tai_conc ** par['eta_tai-dna']) / (
                tai_conc ** par['eta_tai-dna'] + par['K_tai-dna'] ** par['eta_tai-dna'])

    # get total resource competition
    Q = F_s*(par['q_s']+par['q_ofp']) + par['q_ta'] + F_b*par['q_b']
    # get total growth rate slowdown
    J = F_s*(par['j_s']+par['j_ofp']) + par['j_ta'] + F_b*par['j_b']

    # get the slowed-down growth rate
    l = par['l0'] / (1+J)

    dxdt=[
        # self-activating switch
        # switch protein
        par['v_s'] * (F_s*par['q_s']/(1+Q)) - par['deg_s'] * p_s - l * p_s,
        # immature output fluorescence protein
        par['v_ofp'] * (F_s*par['q_ofp']/(1+Q)) - par['deg_ofp'] * p_ofp - l * p_ofp - par['mu_ofp'] * p_ofp,
        # mature output fluorescence protein
        par['mu_ofp'] * p_ofp - par['deg_ofp'] * ofp_mature - l * ofp_mature,
        # probe
        # transcription activator protein
        par['v_ta'] * (par['q_ta']/(1+Q)) - par['deg_ta'] * p_ta - l * p_ta,
        # gene b mRNA
        par['v_b'] * (F_b*par['q_b']/(1+Q)) - par['deg_b'] * p_b - l * p_b
    ]

    return


def simulate_mpc_sys_model(ref,  # reference to be tracked
                           u0,  # original input (not subject to the current MPC iteration)
                           us,  # sequence of control inputs
                           meastimestep,  # measurement time step
                           horizon_steps  # prediction horizon in measurement time steps
                           ):

    return

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
    default_par['Kp'] = 0.1  # proportional control gain
    default_par['Ki'] = 0.01  # integral control gain

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
    u_calc = par['Kp'] * (ref - x_ctrled) + par['Ki'] * ctrl_memo[controller_name2pos['integral']]

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