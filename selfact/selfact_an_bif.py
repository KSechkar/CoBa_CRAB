# SELFACT_AN_BIF.PY: Analaytical bifurcation diagram retrieval for the self-activating switch

# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import functools
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
import pickle
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, palettes as bkpalettes, transform as bktransform
from math import pi
import time

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
from sim_tools.cell_model import *
import sim_tools.cell_genetic_modules as gms
import sim_tools.controllers as ctrls
import sim_tools.reference_switchers as refsws
import sim_tools.ode_solvers as odesols

# F_SWITCH FUNCTION VALUES ---------------------------------------------------------------------------------------------
# real value
def F_real_calc(p_switch,   # switch protein level
                par # system parameters
                ):
    p_switch_term = (p_switch * par['I_switch']) / par['K_switch']

    F_switch = par['baseline_switch'] + (1 - par['baseline_switch']) * (p_switch_term ** par['eta_switch']) / (
                p_switch_term ** par['eta_switch'] + 1)
    return F_switch


# required value
def F_req_calc(p_switch, # switch protein level
               Q_osynth, # burden of other (non-switch) synthetic genes
               par, # system parameters
               cellvars # steady-state cellular variables obtained by simulation
               ):
    return p_switch * (1 + cellvars['chi_switch']) * \
        (1 + Q_osynth) / cellvars['Q_sas_max'] / \
        (
                par['M'] * (1 - par['phi_q']) / par['n_switch'] *
                (cellvars['Q_switch_max'] / cellvars['Q_sas_max'])
                -
                p_switch * (1 + cellvars['chi_switch'])
        )

# F_SWITCH FUNCTION GRADIENTS ------------------------------------------------------------------------------------------
# real value
def dFreal_dpswitch_calc(p_switch, par):
    K_div_ac_frac = par['K_switch'] / par['I_switch']
    return par['eta_switch'] * (1 - par['baseline_switch']) * K_div_ac_frac ** par[
        'eta_switch'] * p_switch ** (par['eta_switch'] - 1) / \
        (K_div_ac_frac ** par['eta_switch'] + p_switch ** par['eta_switch']) ** 2


# required value
def dFreq_dpswitch_calc(p_switch,   # switch protein level
                        Q_osynth,   # burden of other (non-switch) synthetic genes
                        par,        # system parameters
                        cellvars    # steady-state cellular variables obtained by simulation
                        ):
    # Freq = a * p_switch / (b - p_switch) => dFreq/dpswitch = a * b / (b - p_switch) ** 2

    a_coeff = (1 + Q_osynth) / cellvars['Q_sas_max']

    b_coeff = par['M'] * (1 - par['phi_q']) / ((1 + cellvars['chi_switch']) * par['n_switch']) \
                * (cellvars['Q_switch_max'] / cellvars['Q_sas_max'])

    return a_coeff * b_coeff / (b_coeff - p_switch) ** 2


# FINDING THE BIFURCATIONS: AUXILIARIES FOR THE PARAMETRIC APPROACH ----------------------------------------------------
# difference of gradients at the fixed point for a given value of F_switch
def gradiff_from_F(F,   # value of F_switch
                   par,     # system parameters
                   cellvars     # steady-state cellular variables obtained by simulation
                   ):
    # reconstruct p_switch and Q_osynth
    p_switch = pswitch_from_F(F, par)
    Q_osynth = Q_osynth_from_F_and_pswitch(F, p_switch, par, cellvars)

    # get the gradients
    dFreal_dpswitch = dFreal_dpswitch_calc(p_switch, par)
    dFreq_dpswitch = dFreq_dpswitch_calc(p_switch, Q_osynth, par, cellvars)
    return dFreal_dpswitch - dFreq_dpswitch


# difference of gradients at the fixed point for a given value of p_switch
def gradiff_from_pswitch(p_switch,  # value of p_switch
                         par,    # system parameters
                         cellvars       # steady-state cellular variables obtained by simulation
                         ):
    # reconstruct F_switch and Q_osynth
    F = F_real_calc(p_switch, par)
    Q_osynth = Q_osynth_from_F_and_pswitch(F, p_switch, par, cellvars)

    # get the gradients
    dFreal_dpswitch = dFreal_dpswitch_calc(p_switch, par)
    dFreq_dpswitch = dFreq_dpswitch_calc(p_switch, Q_osynth, par, cellvars)
    return dFreal_dpswitch - dFreq_dpswitch


# find p_switch value for a given real value of F_switch
def pswitch_from_F(F, par):
    return (par['K_switch'] / par['I_switch']) * \
        ((F - par['baseline_switch']) / (1 - F)) ** (1 / par['eta_switch'])


# find the value of Q_osynth yielding a fixed point for given F_switch and p_switch values
def Q_osynth_from_F_and_pswitch(F, p_switch, par, cellvars):
    return F * cellvars['Q_sas_max'] / (p_switch * (1 + cellvars['chi_switch'])) * (
                par['M'] * (1 - par['phi_q']) / par['n_switch'] * cellvars['Q_switch_max'] / cellvars['Q_sas_max'] - p_switch * (1 + cellvars['chi_switch'])
            ) - 1


# just for convenience, can get a pair of Q_osynth and p_switch values
def pswitch_and_Q_osynth_from_F(F, par, cellvars):
    p_switch = pswitch_from_F(F, par)
    Q_osynth = Q_osynth_from_F_and_pswitch(F, p_switch, par, cellvars)
    return jnp.array([p_switch,Q_osynth])


# upper bound for Q_osynth to find the saddle point at the lower bifurcation (inflexion in real F_switch values)
def pswitch_inflexion_in_Freal(par):
    return ((par['eta_switch'] - 1) / (par['eta_switch'] + 1)) ** (1 / par['eta_switch']) * (
                par['K_switch'] / par['I_switch'])

# FINDING THE STEADY STATES: AUXILIARIES FOR THE PARAMETRIC APPROACH ---------------------------------------------------
# difference of Freal and Freq for a given p_switch value
def diff_from_pswitch(pswitch, Q_osynth, par, cellvars):
    # get the F values
    F_real = F_real_calc(pswitch, par)
    F_req = F_req_calc(pswitch, Q_osynth, par, cellvars)
    return F_real - F_req


# CALCULATING OUTPUT FLUORESCENT PROTEIN LEVELS ------------------------------------------------------------------------
# calculate the total output fluorescent protein level for a given p_switch value
def p_ofp_tot_from_p_switch_and_Q_osynth(p_switch, Q_osynth,
                                     par, cellvars):
    # get the F value
    F = F_real_calc(p_switch, par)

    return 1 / (1 + cellvars['chi_ofp']) * \
        par['M'] * (1 - par['phi_q']) / par['n_ofp'] * \
        (F * cellvars['Q_ofp_max']) / (1 + Q_osynth + F*cellvars['Q_sas_max'])

# calculate the immature and mature output fluorescent protein levels for a given total value
def p_ofp_immat_mat(p_switch, Q_osynth, par, cellvars):
    # get the F value
    F = F_real_calc(p_switch, par)

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
# find the maximum possible value of p_switch
def find_p_switch_sup(par, cellvars):
    return 1 / (1 + cellvars['chi_switch']) * \
        par['M'] * (1 - par['phi_q']) / par['n_switch'] * \
        cellvars['Q_switch_max'] / (1+cellvars['Q_sas_max'])


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
    init_conds.update({'s': nutr_qual})

    # SET ZERO SYNTHETIC GENE EXPRESSION TO RETRIEVE THE STEADY STATE
    par['c_switch']=0.0
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
        if((gene=='ofp') and ('switch' in synth_genes)):  # in the self-activating switch, the ofp gene is co-expressed with the switch => switch gene c and a used
            xi_gene_max = xi_calc(1,  # the gene is functional
                                  1,    # for maximum burden, maximum transcription regulation function value is used, i.e. F=1
                                  par['c_switch'], par['a_switch'],
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
# for one point, the required F_switch value touches the real F_switch curve from below; for the other, from above
def find_bifurcations(par, cellvars):
    # intervals in which to look for the bifurcations
    p_switch_sup = find_p_switch_sup(par, cellvars)  # upper bound for p_switch
    pswitch_inflexion = pswitch_inflexion_in_Freal(par)  # inflexion point in real F_switch)
    F_min = par['baseline_switch']  # lower bound of feasible region for F (corresponds to p_switch=0)
    F_max = F_real_calc(p_switch_sup, par)  # upper bound of feasible region for F
    F_inflexion = F_real_calc(pswitch_inflexion, par)  # inflexion point in real F_switch

    # find the bifurcation with F_req touching F_real from below
    Freqbelow_bif_problem = jaxopt.Bisection(optimality_fun=gradiff_from_F,
                                             lower=F_min, upper=F_inflexion,
                                             maxiter=10000, tol=1e-18,
                                             check_bracket=False)  # required for vmapping and jitting
    F_bif_Freqbelow = Freqbelow_bif_problem.run(par=par, cellvars=cellvars).params  # F_switch value at the bifurcation

    # find the bifurcation with F_req touching F_real from above
    Freqabove_bif_problem = jaxopt.Bisection(optimality_fun=gradiff_from_F,
                                             lower=F_inflexion, upper=F_max,
                                             maxiter=10000, tol=1e-18,
                                             check_bracket=False)  # required for vmapping and jitting
    F_bif_Freqabove = Freqabove_bif_problem.run(par=par, cellvars=cellvars).params  # F_switch value at the bifurcation

    # get associated p_switch, p_ofp_tot, Q_sas and Q_osynth values
    # touch-below bifurcation
    p_switch_bif_Freqbelow = pswitch_from_F(F_bif_Freqbelow, par)  # p_switch value at the bifurcation
    Q_osynth_bif_Freqbelow = Q_osynth_from_F_and_pswitch(F_bif_Freqbelow,
                                                         p_switch_bif_Freqbelow,
                                                         par, cellvars)  # Q_osynth value enabling the bifurcation
    p_ofp_tot_bif_Freqbelow = p_ofp_tot_from_p_switch_and_Q_osynth(p_switch_bif_Freqbelow, Q_osynth_bif_Freqbelow, par, cellvars)  # p_ofp value at the bifurcation
    Q_sas_bif_Freqbelow = cellvars['Q_sas_max']*F_real_calc(p_switch_bif_Freqbelow, par)  # Q_sas value at the bifurcation
    # touch-above bifurcation
    p_switch_bif_Freqabove = pswitch_from_F(F_bif_Freqabove, par)  # p_switch value at the bifurcation
    Q_osynth_bif_Freqabove = Q_osynth_from_F_and_pswitch(F_bif_Freqabove,
                                                         p_switch_bif_Freqabove,
                                                         par, cellvars)  # Q_osynth value enabling the bifurcation
    p_ofp_tot_bif_Freqabove = p_ofp_tot_from_p_switch_and_Q_osynth(p_switch_bif_Freqabove, Q_osynth_bif_Freqabove, par, cellvars)  # p_ofp value at the bifurcation
    Q_sas_bif_Freqabove = cellvars['Q_sas_max']*F_real_calc(p_switch_bif_Freqabove, par)  # Q_sas value at the bifurcation

    # return
    return (
        {'F': F_bif_Freqbelow, 'p_switch': p_switch_bif_Freqbelow, 'Q_osynth': Q_osynth_bif_Freqbelow,
            'p_ofp': p_ofp_tot_bif_Freqbelow, 'Q_sas': Q_sas_bif_Freqbelow},
        {'F': F_bif_Freqabove, 'p_switch': p_switch_bif_Freqabove, 'Q_osynth': Q_osynth_bif_Freqabove,
            'p_ofp': p_ofp_tot_bif_Freqabove, 'Q_sas': Q_sas_bif_Freqabove}
    )


# FINDING THE EQUILIIBRIA ----------------------------------------------------------------------------------------------
# find equilibria for the given Q_osynth values
def find_equilibria_for_Q_osynth_range(Q_osynth_range, # range of burdens from other genes to consider
                                       bifurcation_Freqbelow, bifurcation_Freqabove,    # bifurcation points
                                       par, cellvars):  # system parameters and steady-state cellular variables
    # get p_switch and Q_osynth values at the bifurcation points
    p_switch_bif_Freqbelow = bifurcation_Freqbelow['p_switch']
    Q_osynth_bif_Freqbelow = bifurcation_Freqbelow['Q_osynth']
    p_switch_bif_Freqabove = bifurcation_Freqabove['p_switch']
    Q_osynth_bif_Freqabove = bifurcation_Freqabove['Q_osynth']

    # initialise lists to record the bifurcation curve points in
    bifurcation_curve_Q_osynth = []
    bifurcation_curve_p_switch = []
    bifurcation_curve_Q_sas = []
    bifurcation_curve_p_ofp_tot = []
    bifurcation_curve_p_ofp = [] # immature ofp
    bifurcation_curve_ofp_mature = [] # mature ofp

    # get the highest-possible p_switch value
    p_switch_sup = find_p_switch_sup(par, cellvars)

    # get high-expression equilibria for Q_osynth values where they are possible
    for Q_osynth in Q_osynth_range:
        # check if the high-expression equilibrium is possible
        if (Q_osynth < Q_osynth_bif_Freqabove):
            # at high-expression equilibiria, p_switch is above the touch-above bifurcation value
            hieq_problem = jaxopt.Bisection(optimality_fun=diff_from_pswitch,
                                            lower=p_switch_bif_Freqabove, upper=p_switch_sup,
                                            maxiter=10000, tol=1e-18,
                                            check_bracket=False)  # required for vmapping and jitting
            p_switch_hieq = hieq_problem.run(Q_osynth=Q_osynth, par=par, cellvars=cellvars).params

            # record the high-expression equilibrium
            bifurcation_curve_Q_osynth.append(Q_osynth)
            bifurcation_curve_p_switch.append(p_switch_hieq)
            bifurcation_curve_Q_sas.append(cellvars['Q_sas_max'] * F_real_calc(p_switch_hieq, par))
            bifurcation_curve_p_ofp_tot.append(p_ofp_tot_from_p_switch_and_Q_osynth(p_switch_hieq, Q_osynth, par, cellvars))
            p_ofp, ofp_mature = p_ofp_immat_mat(p_switch_hieq, Q_osynth, par, cellvars)
            bifurcation_curve_p_ofp.append(p_ofp)
            bifurcation_curve_ofp_mature.append(ofp_mature)

    # add the touch-above bifurcation point
    bifurcation_curve_Q_osynth.append(Q_osynth_bif_Freqabove)
    bifurcation_curve_p_switch.append(p_switch_bif_Freqabove)
    bifurcation_curve_Q_sas.append(cellvars['Q_sas_max'] * F_real_calc(p_switch_bif_Freqabove, par))
    bifurcation_curve_p_ofp_tot.append(p_ofp_tot_from_p_switch_and_Q_osynth(p_switch_bif_Freqabove, Q_osynth_bif_Freqabove, par, cellvars))
    p_ofp, ofp_mature = p_ofp_immat_mat(p_switch_bif_Freqabove, Q_osynth_bif_Freqabove, par, cellvars)
    bifurcation_curve_p_ofp.append(p_ofp)
    bifurcation_curve_ofp_mature.append(ofp_mature)

    # get the saddle point for Q_osynth values where they are possible
    for Q_osynth in np.flip(Q_osynth_range):  # range flipped for continuity of the curve
        # check if the saddle point is possible
        if (Q_osynth > Q_osynth_bif_Freqbelow and Q_osynth < Q_osynth_bif_Freqabove):
            # at saddle points, p_switch is below the touch-below bifurcation value
            sp_problem = jaxopt.Bisection(optimality_fun=diff_from_pswitch,
                                          lower=p_switch_bif_Freqbelow, upper=p_switch_bif_Freqabove,
                                          maxiter=10000, tol=1e-18,
                                          check_bracket=False)
            p_switch_sp = sp_problem.run(Q_osynth=Q_osynth, par=par, cellvars=cellvars).params
            bifurcation_curve_Q_osynth.append(Q_osynth)
            bifurcation_curve_p_switch.append(p_switch_sp)
            bifurcation_curve_Q_sas.append(cellvars['Q_sas_max'] * F_real_calc(p_switch_sp, par))
            bifurcation_curve_p_ofp_tot.append(p_ofp_tot_from_p_switch_and_Q_osynth(p_switch_sp, Q_osynth, par, cellvars))
            p_ofp, ofp_mature = p_ofp_immat_mat(p_switch_sp, Q_osynth, par, cellvars)
            bifurcation_curve_p_ofp.append(p_ofp)
            bifurcation_curve_ofp_mature.append(ofp_mature)

    # add the touch-above bifurcation point
    bifurcation_curve_Q_osynth.append(Q_osynth_bif_Freqbelow)
    bifurcation_curve_p_switch.append(p_switch_bif_Freqbelow)
    bifurcation_curve_Q_sas.append(cellvars['Q_sas_max'] * F_real_calc(p_switch_bif_Freqbelow, par))
    bifurcation_curve_p_ofp_tot.append(p_ofp_tot_from_p_switch_and_Q_osynth(p_switch_bif_Freqbelow, Q_osynth_bif_Freqbelow, par, cellvars))
    p_ofp, ofp_mature = p_ofp_immat_mat(p_switch_bif_Freqbelow, Q_osynth_bif_Freqbelow, par, cellvars)
    bifurcation_curve_p_ofp.append(p_ofp)
    bifurcation_curve_ofp_mature.append(ofp_mature)

    # get the low-expression equilibria for Q_osynth values where they are possible
    for Q_osynth in Q_osynth_range:  # range NOT flipped for continuity of the curve
        # check if the low-expression equilibrium is possible
        if (Q_osynth > Q_osynth_bif_Freqbelow):
            # at low-expression equilibiria, p_switch is below the touch-below bifurcation value
            leeQ_problem = jaxopt.Bisection(optimality_fun=diff_from_pswitch,
                                            lower=0, upper=p_switch_bif_Freqbelow,
                                            maxiter=10000, tol=1e-18,
                                            check_bracket=False)
            p_switch_leeq = leeQ_problem.run(Q_osynth=Q_osynth, par=par, cellvars=cellvars).params
            # record the low-expression equilibrium
            bifurcation_curve_Q_osynth.append(Q_osynth)
            bifurcation_curve_p_switch.append(p_switch_leeq)
            bifurcation_curve_Q_sas.append(cellvars['Q_sas_max'] * F_real_calc(p_switch_leeq, par))
            bifurcation_curve_p_ofp_tot.append(p_ofp_tot_from_p_switch_and_Q_osynth(p_switch_leeq, Q_osynth, par, cellvars))
            p_ofp, ofp_mature = p_ofp_immat_mat(p_switch_leeq, Q_osynth, par, cellvars)
            bifurcation_curve_p_ofp.append(p_ofp)
            bifurcation_curve_ofp_mature.append(ofp_mature)

    # return
    return {'p_switch': np.array(bifurcation_curve_p_switch), 'Q_osynth': np.array(bifurcation_curve_Q_osynth),
            'p_ofp_total': np.array(bifurcation_curve_p_ofp_tot), 'Q_sas': np.array(bifurcation_curve_Q_sas),
            'p_ofp': np.array(bifurcation_curve_p_ofp), 'ofp_mature': np.array(bifurcation_curve_ofp_mature)}


# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()