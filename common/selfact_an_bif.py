# SELFACT_AN_BIF.PY: Analaytical bifurcation diagram retrieval for the self-activating switch

# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import jaxopt

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
from cell_model_case.cell_model import *
import cell_model_case.cell_genetic_modules as gms
import common.controllers as ctrls
import common.reference_switchers as refsws
import common.ode_solvers as odesols

# F_SWITCH FUNCTION VALUES ---------------------------------------------------------------------------------------------
# real value
def F_real_calc(p_s,   # switch protein level
                par # system parameters
                ):
    p_s_term = (p_s * par['I_s']) / par['K_s,s']

    F_s = par['F_s,0'] + (1 - par['F_s,0']) * (p_s_term ** par['eta_s,s']) / (
                p_s_term ** par['eta_s,s'] + 1)
    return F_s


# required value
def F_req_calc(p_s, # switch protein level
               Q_osynth, # burden of other (non-switch) synthetic genes
               par, # system parameters
               cellvars # steady-state cellular variables obtained by simulation
               ):
    return p_s * (1 + cellvars['chi_s']) * \
        (1 + Q_osynth) / cellvars['Q_sas_max'] / \
        (
                par['M'] * (1 - par['phi_q']) / par['n_s'] *
                (cellvars['Q_s_max'] / cellvars['Q_sas_max'])
                -
                p_s * (1 + cellvars['chi_s'])
        )

# F_SWITCH FUNCTION GRADIENTS ------------------------------------------------------------------------------------------
# real value
def dFreal_dps_calc(p_s, par):
    K_div_ac_frac = par['K_s,s'] / par['I_s']
    return par['eta_s,s'] * (1 - par['F_s,0']) * K_div_ac_frac ** par[
        'eta_s,s'] * p_s ** (par['eta_s,s'] - 1) / \
        (K_div_ac_frac ** par['eta_s,s'] + p_s ** par['eta_s,s']) ** 2


# required value
def dFreq_dps_calc(p_s,   # switch protein level
                        Q_osynth,   # burden of other (non-switch) synthetic genes
                        par,        # system parameters
                        cellvars    # steady-state cellular variables obtained by simulation
                        ):
    # Freq = a * p_s / (b - p_s) => dFreq/dps = a * b / (b - p_s) ** 2

    a_coeff = (1 + Q_osynth) / cellvars['Q_sas_max']

    b_coeff = par['M'] * (1 - par['phi_q']) / ((1 + cellvars['chi_s']) * par['n_s']) \
                * (cellvars['Q_s_max'] / cellvars['Q_sas_max'])

    return a_coeff * b_coeff / (b_coeff - p_s) ** 2


# FINDING THE BIFURCATIONS: AUXILIARIES FOR THE PARAMETRIC APPROACH ----------------------------------------------------
# difference of gradients at the fixed point for a given value of F_s
def gradiff_from_F(F,   # value of F_s
                   par,     # system parameters
                   cellvars     # steady-state cellular variables obtained by simulation
                   ):
    # reconstruct p_s and Q_osynth
    p_s = ps_from_F(F, par)
    Q_osynth = Q_osynth_from_F_and_ps(F, p_s, par, cellvars)

    # get the gradients
    dFreal_dps = dFreal_dps_calc(p_s, par)
    dFreq_dps = dFreq_dps_calc(p_s, Q_osynth, par, cellvars)
    return dFreal_dps - dFreq_dps


# difference of gradients at the fixed point for a given value of p_s
def gradiff_from_ps(p_s,  # value of p_s
                         par,    # system parameters
                         cellvars       # steady-state cellular variables obtained by simulation
                         ):
    # reconstruct F_s and Q_osynth
    F = F_real_calc(p_s, par)
    Q_osynth = Q_osynth_from_F_and_ps(F, p_s, par, cellvars)

    # get the gradients
    dFreal_dps = dFreal_dps_calc(p_s, par)
    dFreq_dps = dFreq_dps_calc(p_s, Q_osynth, par, cellvars)
    return dFreal_dps - dFreq_dps


# find p_s value for a given real value of F_s
def ps_from_F(F, par):
    return (par['K_s,s'] / par['I_s']) * \
        ((F - par['F_s,0']) / (1 - F)) ** (1 / par['eta_s,s'])


# find the value of Q_osynth yielding a fixed point for given F_s and p_s values
def Q_osynth_from_F_and_ps(F, p_s, par, cellvars):
    return F * cellvars['Q_sas_max'] / (p_s * (1 + cellvars['chi_s'])) * (
                par['M'] * (1 - par['phi_q']) / par['n_s'] * cellvars['Q_s_max'] / cellvars['Q_sas_max'] - p_s * (1 + cellvars['chi_s'])
            ) - 1


# just for convenience, can get a pair of Q_osynth and p_s values
def ps_and_Q_osynth_from_F(F, par, cellvars):
    p_s = ps_from_F(F, par)
    Q_osynth = Q_osynth_from_F_and_ps(F, p_s, par, cellvars)
    return jnp.array([p_s,Q_osynth])


# upper bound for Q_osynth to find the saddle point at the lower bifurcation (inflexion in real F_s values)
def ps_inflexion_in_Freal(par):
    return ((par['eta_s,s'] - 1) / (par['eta_s,s'] + 1)) ** (1 / par['eta_s,s']) * (
                par['K_s,s'] / par['I_s'])

# FINDING THE STEADY STATES: AUXILIARIES FOR THE PARAMETRIC APPROACH ---------------------------------------------------
# difference of Freal and Freq for a given p_s value
def diff_from_ps(ps, Q_osynth, par, cellvars):
    # get the F values
    F_real = F_real_calc(ps, par)
    F_req = F_req_calc(ps, Q_osynth, par, cellvars)
    return F_real - F_req


# CALCULATING OUTPUT FLUORESCENT PROTEIN LEVELS ------------------------------------------------------------------------
# calculate the total output fluorescent protein level for a given p_s value
def p_ofp_tot_from_p_s_and_Q_osynth(p_s, Q_osynth,
                                     par, cellvars):
    # get the F value
    F = F_real_calc(p_s, par)

    return 1 / (1 + cellvars['chi_ofp']) * \
        par['M'] * (1 - par['phi_q']) / par['n_ofp'] * \
        (F * cellvars['Q_ofp_max']) / (1 + Q_osynth + F*cellvars['Q_sas_max'])

# calculate the immature and mature output fluorescent protein levels for a given total value
def p_ofp_immat_mat(p_s, Q_osynth, par, cellvars):
    # get the F value
    F = F_real_calc(p_s, par)

    # get the ribosome abundance
    R = par['M'] * ((1 - par['phi_q'])/par['n_r']) * (cellvars['Q_r']/(1+Q_osynth+ F*cellvars['Q_sas_max']))

    # get the cell growth rate
    l = R*cellvars['e']/par['M']

    # get the total removal rate (with degradation by protease)
    removal_rate = l*(1+cellvars['chi_ofp'])

    # get the rate of p_ofp synthesis
    synth_rate = R * cellvars['e'] * (1 - par['phi_q']) / par['n_ofp'] * \
                 F * cellvars['Q_ofp_max'] / (1 + Q_osynth + F * cellvars['Q_sas_max'])

    # get the immature p_ofp level
    p_ofp = synth_rate/(removal_rate+par['mu_ofp'])
    # get the mature p_ofp level
    ofp_mature = p_ofp * par['mu_ofp'] / removal_rate

    # return the immature and mature protein levels
    return p_ofp, ofp_mature

# DETERMINING SEARCH INTERVAL BOUNDARIES -------------------------------------------------------------------------------
# find the maximum possible value of p_s
def find_p_s_sup(par, cellvars):
    return 1 / (1 + cellvars['chi_s']) * \
        par['M'] * (1 - par['phi_q']) / par['n_s'] * \
        cellvars['Q_s_max'] / (1+cellvars['Q_sas_max'])


# FIND STEADY-STATE CELLULAR VARIABLES BY SIMULATION -------------------------------------------------------------------
# calculate the NON-NORMALISED burden value for a gene
def xi_calc(func,F,c,a,k):
    return func*F*c*a/k


# calculate the degradation coefficient for a gene
def chi_calc(d, e, h, par, xi_prot, xi_r):
    return ((par['K_D'] + h) / par['K_D']) *\
        d * par['M']/e *\
        par['n_r']/par['n_prot'] * xi_prot/xi_r


# get the steady-state translation elongation rate and ribosomal gene transcription function for a given medium nutrient quality
def get_ss_F_and_e(nutr_qual):
    # initialise cell model
    model_auxil = ModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    model_par = model_auxil.default_params()  # get default parameter values
    init_conds = model_auxil.default_init_conds(model_par)  # get default initial conditions

    # add reference tracker switcher
    model_par_with_refswitch, ref_switcher = model_auxil.add_reference_switcher(model_par,
                                                                                        # cell model parameters
                                                                                        refsws.no_switching_initialise,
                                                                                        # function initialising the reference switcher
                                                                                        refsws.no_switching_switch
                                                                                        # function switching the references to be tracked
                                                                                        )

    # load synthetic genetic modules (self-activating switch + unexpressed const. gene) and the controller (constant irrelevant input)
    odeuus_complete, \
        module1_F_calc, module2_F_calc, \
        module1_specterms, module2_specterms, \
        controller_action, controller_update, \
        par, init_conds, controller_memo0, \
        synth_genes_total_and_each, synth_miscs_total_and_each, \
        controller_memos, controller_dynvars, controller_ctrledvar, \
        modules_name2pos, modules_styles, controller_name2pos, controller_styles, \
        module1_v_with_F_calc, module2_v_with_F_calc = model_auxil.add_modules_and_controller(
            # module 1
            gms.sas_initialise,  # function initialising the circuit
            gms.sas_ode,  # function defining the circuit ODEs
            gms.sas_F_calc, # function calculating the circuit genes' transcription regulation functions
            gms.sas_specterms,  # function calculating the circuit genes' effective mRNA levels
            # module 2
            gms.cicc_initialise,  # function initialising the circuit
            gms.cicc_ode,  # function defining the circuit ODEs
            gms.cicc_F_calc,    # function calculating the circuit genes' transcription regulation functions
            gms.cicc_specterms,  # function calculating the circuit genes' effective mRNA levels
            # controller
            ctrls.cci_initialise,  # function initialising the controller
            ctrls.cci_action,  # function calculating the controller action
            ctrls.cci_ode,  # function defining the controller ODEs
            ctrls.cci_update,  # function updating the controller based on measurements
            # cell model parameters and initial conditions
            model_par_with_refswitch, init_conds)

    # unpack the synthetic genes and miscellaneous species lists
    synth_genes = synth_genes_total_and_each[0]
    module1_genes = synth_genes_total_and_each[1]
    module2_genes = synth_genes_total_and_each[2]
    synth_miscs = synth_miscs_total_and_each[0]
    module1_miscs = synth_miscs_total_and_each[1]
    module2_miscs = synth_miscs_total_and_each[2]

    # LOAD THE PARAMETERS TO BE CONSIDERED
    init_conds.update({'sigma': nutr_qual})

    # SET ZERO SYNTHETIC GENE EXPRESSION TO RETRIEVE THE STEADY STATE
    par['c_s']=0.0
    par['c_ta']=0.0
    par['c_b']=0.0
    init_conds['inducer_level']=0.0

    # SET CONTROLLER PARAMETERS - irrelevant for the steady-state retrieval
    refs = jnp.array([0.0])
    control_delay = 0  # control action delay
    u0 = 0.0  # initial control action

    # get the jaxed synthetic gene parameters
    sgp4j=model_auxil.synth_gene_params_for_jax(par, synth_genes)

    # DETERMINISTIC SIMULATION
    # set simulation parameters
    tf = (0.0, 48.0)  # simulation time frame - assuming the steady state is reached in 48 hours

    # measurement time step
    meastimestep = 48.0  # hours

    # choose ODE solver
    ode_solver, us_size = odesols.create_diffrax_solver(odeuus_complete,
                                                        control_delay=0,
                                                        meastimestep=meastimestep,
                                                        rtol=1e-6, atol=1e-6,
                                                        solver_spec='Kvaerno3')

    # solve ODE
    ts_jnp, xs_jnp, \
        ctrl_memorecord_jnp, uexprecord_jnp, \
        refrecord_jnp = ode_sim(par,  # model parameters
                                ode_solver,  # ODE solver for the cell with the synthetic gene circuit
                                odeuus_complete,
                                # ODE function for the cell with the synthetic gene circuit and the controller (also gives calculated and experienced control actions)
                                controller_ctrledvar,  # name of the variable read and steered by the controller
                                controller_update, controller_action,
                                # function for updating the controller memory and calculating the control action
                                model_auxil.x0_from_init_conds(init_conds,
                                                                   par,
                                                                   synth_genes, synth_miscs, controller_dynvars,
                                                                   modules_name2pos,
                                                                   controller_name2pos),  # initial condition VECTOR
                                controller_memo0,  # initial controller memory record
                                u0,
                                # initial control action, applied before any measurement-informed actions reach the system
                                (len(synth_genes), len(module1_genes), len(module2_genes)),  # number of synthetic genes
                                (len(synth_miscs), len(module1_miscs), len(module2_miscs)),
                                # number of miscellaneous species
                                modules_name2pos, controller_name2pos,
                                # dictionaries mapping gene names to their positions in the state vector
                                sgp4j,
                                # synthetic gene parameters in jax.array form
                                tf, meastimestep,  # simulation time frame and measurement time step
                                control_delay,  # delay before control action reaches the system
                                us_size,  # size of the control action record needed
                                refs, ref_switcher,  # reference values and reference switcher
                                )

    # convert simulation results to numpy arrays
    ts = np.array(ts_jnp)
    xs = np.array(xs_jnp)
    ctrl_memorecord = np.array(ctrl_memorecord_jnp).T
    uexprecord = np.array(uexprecord_jnp)
    refrecord = np.array(refrecord_jnp)

    # LOOK AT THE STEADY STATE
    # get steady-state translation elongation rate and ribosomal gene transcription function
    es, _, F_rs, _, _, _, _ = model_auxil.get_e_l_Fr_nu_psi_T_D(ts, xs, par,
                                                                    synth_genes, synth_miscs,
                                                                    modules_name2pos,
                                                                    module1_specterms, module2_specterms)
    e = es[-1]
    F_r = F_rs[-1]

    return e, F_r


# get normalised maximum burden values and degradation coefficients for the synthetic genes, given the steady-state e and F_r
# CURRENTLY DOES NOT HANDLE NON-ZERO CHLORAMPHENICOL CONCENTRATIONS
def find_Qs_and_chis(
        e_ss,  # steady-state translation elongation rate
        F_r_ss,  # steady-state ribosomal gene transcription function
        par,  # system parameters
        synth_genes,  # list of synthetic genes
        modules_name2pos,  # dictionary mapping gene names to their positions in the state vector
    ):
    model_auxil = ModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes

    # get the jaxed synthetic gene parameters
    sgp4j = model_auxil.synth_gene_params_for_jax(par, synth_genes)

    # get the apparent ribosome-mRNA dissociation constants
    k_a = k_calc(e_ss, par['k+_a'], par['k-_a'], par['n_a'])  # metabolic genes
    k_r = k_calc(e_ss, par['k+_r'], par['k-_r'], par['n_r'])  # ribosomal genes
    k_het = np.array(k_calc(e_ss, sgp4j[0], sgp4j[1], sgp4j[2]))  # array for all heterologous genes

    # get the NON-NORMALISED native gene expression burden
    xi_a = xi_calc(1, 1, par['c_a'], par['a_a'], k_a)  # metabolic genes (F=1 since genes are constitutive)
    xi_r = xi_calc(1, F_r_ss, par['c_r'], par['a_r'], k_r)  # ribosomal genes
    xi_native = xi_a + xi_r  # total native gene expression burden

    # for each synthetic gene, get the NORMALISED, MAXIMUM POSSIBLE burden value and the degradation coefficient
    Qmaxs = {}  # dictionary for the normalised maximum burden values
    chis = {}  # dictionary for the degradation coefficients
    for gene in synth_genes:
        if((gene=='ofp') and ('s' in synth_genes)):  # in the self-activating switch, the ofp gene is co-expressed with the switch => switch gene c and a used
            xi_gene_max = xi_calc(1,  # the gene is functional
                                  1,    # for maximum burden, maximum transcription regulation function value is used, i.e. F=1
                                  par['c_s'], par['a_s'],
                                  k_het[modules_name2pos['k_ofp']])
        else:
            xi_gene_max = xi_calc(1,  # the gene is functional
                                  1,    # for maximum burden, maximum transcription regulation function value is used, i.e. F=1
                                  par['c_' + gene], par['a_' + gene],
                                  k_het[modules_name2pos['k_' + gene]])
        Qmaxs['Q_'+gene+'_max'] = xi_gene_max / xi_native  # record the normalised maximum burden value
        chis['chi_'+gene] = chi_calc(par['d_' + gene], e_ss,
                                         0, # assuming no chloramphenicol present
                                         par, xi_gene_max, xi_native)  # record the degradation coefficient

    # knowing normalised burden values q is also useful for native genes => retrieve
    Qs_native = {}
    Qs_native['Q_a']=xi_a/xi_native
    Qs_native['Q_r']=xi_r/xi_native

    # return
    return Qmaxs, Qs_native, chis

# FINDING THE BIFURCATIONS ---------------------------------------------------------------------------------------------
# find the bifurcation points for the self-activating switch
# for one point, the required F_s value touches the real F_s curve from below; for the other, from above
def find_bifurcations(par, cellvars):
    # intervals in which to look for the bifurcations
    p_s_sup = find_p_s_sup(par, cellvars)  # upper bound for p_s
    ps_inflexion = ps_inflexion_in_Freal(par)  # inflexion point in real F_s)
    F_min = par['F_s,0']  # lower bound of feasible region for F (corresponds to p_s=0)
    F_max = F_real_calc(p_s_sup, par)  # upper bound of feasible region for F
    F_inflexion = F_real_calc(ps_inflexion, par)  # inflexion point in real F_s

    # find the bifurcation with F_req touching F_real from below
    Freqbelow_bif_problem = jaxopt.Bisection(optimality_fun=gradiff_from_F,
                                             lower=F_min, upper=F_inflexion,
                                             maxiter=10000, tol=1e-18,
                                             check_bracket=False)  # required for vmapping and jitting
    F_bif_Freqbelow = Freqbelow_bif_problem.run(par=par, cellvars=cellvars).params  # F_s value at the bifurcation

    # find the bifurcation with F_req touching F_real from above
    Freqabove_bif_problem = jaxopt.Bisection(optimality_fun=gradiff_from_F,
                                             lower=F_inflexion, upper=F_max,
                                             maxiter=10000, tol=1e-18,
                                             check_bracket=False)  # required for vmapping and jitting
    F_bif_Freqabove = Freqabove_bif_problem.run(par=par, cellvars=cellvars).params  # F_s value at the bifurcation

    # get associated p_s, p_ofp_tot, Q_sas and Q_osynth values
    # touch-below bifurcation
    p_s_bif_Freqbelow = ps_from_F(F_bif_Freqbelow, par)  # p_s value at the bifurcation
    Q_osynth_bif_Freqbelow = Q_osynth_from_F_and_ps(F_bif_Freqbelow,
                                                         p_s_bif_Freqbelow,
                                                         par, cellvars)  # Q_osynth value enabling the bifurcation
    p_ofp_tot_bif_Freqbelow = p_ofp_tot_from_p_s_and_Q_osynth(p_s_bif_Freqbelow, Q_osynth_bif_Freqbelow, par, cellvars)  # p_ofp value at the bifurcation
    Q_sas_bif_Freqbelow = cellvars['Q_sas_max']*F_real_calc(p_s_bif_Freqbelow, par)  # Q_sas value at the bifurcation
    # touch-above bifurcation
    p_s_bif_Freqabove = ps_from_F(F_bif_Freqabove, par)  # p_s value at the bifurcation
    Q_osynth_bif_Freqabove = Q_osynth_from_F_and_ps(F_bif_Freqabove,
                                                         p_s_bif_Freqabove,
                                                         par, cellvars)  # Q_osynth value enabling the bifurcation
    p_ofp_tot_bif_Freqabove = p_ofp_tot_from_p_s_and_Q_osynth(p_s_bif_Freqabove, Q_osynth_bif_Freqabove, par, cellvars)  # p_ofp value at the bifurcation
    Q_sas_bif_Freqabove = cellvars['Q_sas_max']*F_real_calc(p_s_bif_Freqabove, par)  # Q_sas value at the bifurcation

    # return
    return (
        {'F': F_bif_Freqbelow, 'p_s': p_s_bif_Freqbelow, 'Q_osynth': Q_osynth_bif_Freqbelow,
            'p_ofp': p_ofp_tot_bif_Freqbelow, 'Q_sas': Q_sas_bif_Freqbelow},
        {'F': F_bif_Freqabove, 'p_s': p_s_bif_Freqabove, 'Q_osynth': Q_osynth_bif_Freqabove,
            'p_ofp': p_ofp_tot_bif_Freqabove, 'Q_sas': Q_sas_bif_Freqabove}
    )


# FINDING THE EQUILIIBRIA ----------------------------------------------------------------------------------------------
# find equilibria for the given Q_osynth values
def find_equilibria_for_Q_osynth_range(Q_osynth_range, # range of burdens from other genes to consider
                                       bifurcation_Freqbelow, bifurcation_Freqabove,    # bifurcation points
                                       par, cellvars):  # system parameters and steady-state cellular variables
    # get p_s and Q_osynth values at the bifurcation points
    p_s_bif_Freqbelow = bifurcation_Freqbelow['p_s']
    Q_osynth_bif_Freqbelow = bifurcation_Freqbelow['Q_osynth']
    p_s_bif_Freqabove = bifurcation_Freqabove['p_s']
    Q_osynth_bif_Freqabove = bifurcation_Freqabove['Q_osynth']

    # initialise lists to record the bifurcation curve points in
    bifurcation_curve_Q_osynth = []
    bifurcation_curve_p_s = []
    bifurcation_curve_Q_sas = []
    bifurcation_curve_p_ofp_tot = []
    bifurcation_curve_p_ofp = [] # immature ofp
    bifurcation_curve_ofp_mature = [] # mature ofp

    # get the highest-possible p_s value
    p_s_sup = find_p_s_sup(par, cellvars)

    # get high-expression equilibria for Q_osynth values where they are possible
    for Q_osynth in Q_osynth_range:
        # check if the high-expression equilibrium is possible
        if (Q_osynth < Q_osynth_bif_Freqabove):
            # at high-expression equilibiria, p_s is above the touch-above bifurcation value
            hieq_problem = jaxopt.Bisection(optimality_fun=diff_from_ps,
                                            lower=p_s_bif_Freqabove, upper=p_s_sup,
                                            maxiter=10000, tol=1e-18,
                                            check_bracket=False)  # required for vmapping and jitting
            p_s_hieq = hieq_problem.run(Q_osynth=Q_osynth, par=par, cellvars=cellvars).params

            # record the high-expression equilibrium
            bifurcation_curve_Q_osynth.append(Q_osynth)
            bifurcation_curve_p_s.append(p_s_hieq)
            bifurcation_curve_Q_sas.append(cellvars['Q_sas_max'] * F_real_calc(p_s_hieq, par))
            bifurcation_curve_p_ofp_tot.append(p_ofp_tot_from_p_s_and_Q_osynth(p_s_hieq, Q_osynth, par, cellvars))
            p_ofp, ofp_mature = p_ofp_immat_mat(p_s_hieq, Q_osynth, par, cellvars)
            bifurcation_curve_p_ofp.append(p_ofp)
            bifurcation_curve_ofp_mature.append(ofp_mature)

    # add the touch-above bifurcation point
    bifurcation_curve_Q_osynth.append(Q_osynth_bif_Freqabove)
    bifurcation_curve_p_s.append(p_s_bif_Freqabove)
    bifurcation_curve_Q_sas.append(cellvars['Q_sas_max'] * F_real_calc(p_s_bif_Freqabove, par))
    bifurcation_curve_p_ofp_tot.append(p_ofp_tot_from_p_s_and_Q_osynth(p_s_bif_Freqabove, Q_osynth_bif_Freqabove, par, cellvars))
    p_ofp, ofp_mature = p_ofp_immat_mat(p_s_bif_Freqabove, Q_osynth_bif_Freqabove, par, cellvars)
    bifurcation_curve_p_ofp.append(p_ofp)
    bifurcation_curve_ofp_mature.append(ofp_mature)

    # get the saddle point for Q_osynth values where they are possible
    for Q_osynth in np.flip(Q_osynth_range):  # range flipped for continuity of the curve
        # check if the saddle point is possible
        if (Q_osynth > Q_osynth_bif_Freqbelow and Q_osynth < Q_osynth_bif_Freqabove):
            # at saddle points, p_s is below the touch-below bifurcation value
            sp_problem = jaxopt.Bisection(optimality_fun=diff_from_ps,
                                          lower=p_s_bif_Freqbelow, upper=p_s_bif_Freqabove,
                                          maxiter=10000, tol=1e-18,
                                          check_bracket=False)
            p_s_sp = sp_problem.run(Q_osynth=Q_osynth, par=par, cellvars=cellvars).params
            bifurcation_curve_Q_osynth.append(Q_osynth)
            bifurcation_curve_p_s.append(p_s_sp)
            bifurcation_curve_Q_sas.append(cellvars['Q_sas_max'] * F_real_calc(p_s_sp, par))
            bifurcation_curve_p_ofp_tot.append(p_ofp_tot_from_p_s_and_Q_osynth(p_s_sp, Q_osynth, par, cellvars))
            p_ofp, ofp_mature = p_ofp_immat_mat(p_s_sp, Q_osynth, par, cellvars)
            bifurcation_curve_p_ofp.append(p_ofp)
            bifurcation_curve_ofp_mature.append(ofp_mature)

    # add the touch-above bifurcation point
    bifurcation_curve_Q_osynth.append(Q_osynth_bif_Freqbelow)
    bifurcation_curve_p_s.append(p_s_bif_Freqbelow)
    bifurcation_curve_Q_sas.append(cellvars['Q_sas_max'] * F_real_calc(p_s_bif_Freqbelow, par))
    bifurcation_curve_p_ofp_tot.append(p_ofp_tot_from_p_s_and_Q_osynth(p_s_bif_Freqbelow, Q_osynth_bif_Freqbelow, par, cellvars))
    p_ofp, ofp_mature = p_ofp_immat_mat(p_s_bif_Freqbelow, Q_osynth_bif_Freqbelow, par, cellvars)
    bifurcation_curve_p_ofp.append(p_ofp)
    bifurcation_curve_ofp_mature.append(ofp_mature)

    # get the low-expression equilibria for Q_osynth values where they are possible
    for Q_osynth in Q_osynth_range:  # range NOT flipped for continuity of the curve
        # check if the low-expression equilibrium is possible
        if (Q_osynth > Q_osynth_bif_Freqbelow):
            # at low-expression equilibiria, p_s is below the touch-below bifurcation value
            leeQ_problem = jaxopt.Bisection(optimality_fun=diff_from_ps,
                                            lower=0, upper=p_s_bif_Freqbelow,
                                            maxiter=10000, tol=1e-18,
                                            check_bracket=False)
            p_s_leeq = leeQ_problem.run(Q_osynth=Q_osynth, par=par, cellvars=cellvars).params
            # record the low-expression equilibrium
            bifurcation_curve_Q_osynth.append(Q_osynth)
            bifurcation_curve_p_s.append(p_s_leeq)
            bifurcation_curve_Q_sas.append(cellvars['Q_sas_max'] * F_real_calc(p_s_leeq, par))
            bifurcation_curve_p_ofp_tot.append(p_ofp_tot_from_p_s_and_Q_osynth(p_s_leeq, Q_osynth, par, cellvars))
            p_ofp, ofp_mature = p_ofp_immat_mat(p_s_leeq, Q_osynth, par, cellvars)
            bifurcation_curve_p_ofp.append(p_ofp)
            bifurcation_curve_ofp_mature.append(ofp_mature)

    # return
    return {'p_s': np.array(bifurcation_curve_p_s), 'Q_osynth': np.array(bifurcation_curve_Q_osynth),
            'p_ofp_total': np.array(bifurcation_curve_p_ofp_tot), 'Q_sas': np.array(bifurcation_curve_Q_sas),
            'p_ofp': np.array(bifurcation_curve_p_ofp), 'ofp_mature': np.array(bifurcation_curve_ofp_mature)}

# PARAMETER IDENTIFICATION (BASIC MODEL ONLY!!!) -----------------------------------------------------------------------
# fitting cost function for observed steady-state (Q_sas, Q_osynth) pairs
def id_cost(Q_sas_steady_states, Q_osynth_steady_states,
            # measured steady-state burdens exerted and experienced by the switch
            par, cellvars,  # general system parameters and steady-state cellular variables
            fitted_parvec  # vector of switch parameters being fitted
            ):
    return jnp.sum(jnp.square(Q_osynth_steady_states -
                              find_equilibria_for_Q_sas_range_id(Q_sas_steady_states, par,
                                                                 fitted_parvec[0], fitted_parvec[1], fitted_parvec[2],
                                                                 fitted_parvec[3], fitted_parvec[4], fitted_parvec[5],
                                                                 fitted_parvec[6])
                              ))


def ps_from_F_id(F,
                      K_s,switch,
                      I_switch,
                      F_s0,
                      eta_switch
                      ):
    return jnp.where(jnp.logical_and(F>=F_s0,eta_switch>=0.0),
                      (K_s,switch / I_switch) * ((F-F_s0) / (1 - F)) ** (1/eta_switch),
               jnp.zeros_like(F)
               )


def find_equilibria_for_Q_sas_range_id(Q_sas_range,  # range of burdens from other genes to consider
                                       par,
                                       Q_s_max,
                                       Q_ofp_max,
                                       n_switch,
                                       K_s,switch,
                                       I_switch,
                                       F_s0,
                                       eta_switch
                                       ):
    # get the maximum burden from the self-activating switch
    Q_sas_max = Q_s_max + Q_ofp_max

    # get the switch gene's transcription regulation function values corresponding to the Q_sas_range values
    F_s_range = jnp.minimum(Q_sas_range / Q_sas_max,jnp.ones_like(Q_sas_range))

    # get the corresponding switch protein levels
    p_s_range = ps_from_F_id(F_s_range,
                                       K_s,switch,
                                       I_switch,
                                       F_s0,
                                       eta_switch
                                       )

    # get the corresponding Q_osynth levels
    Q_osynth_range = -1 + jnp.multiply(jnp.multiply(F_s_range, Q_sas_max / p_s_range),
                                       par['M'] / n_switch * Q_s_max / Q_sas_max - p_s_range
                                       )

    # return
    return jnp.fmax(jnp.fmin(Q_osynth_range,jnp.ones_like(Q_osynth_range)),jnp.zeros_like(Q_osynth_range))


# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()