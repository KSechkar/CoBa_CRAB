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

import scipy

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
from sim_tools.cell_model import *
import sim_tools.genetic_modules as gms
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
               q_osynth, # burden of other (non-switch) synthetic genes
               par, # system parameters
               cellvars # steady-state cellular variables obtained by simulation
               ):
    return p_switch * (1 + cellvars['chi_switch']) * \
        (1 + q_osynth) / cellvars['q_sas_max'] / \
        (
                par['M'] * (1 - par['phi_q']) / par['n_switch'] *
                (cellvars['q_switch_max'] / cellvars['q_sas_max'])
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
                        q_osynth,   # burden of other (non-switch) synthetic genes
                        par,        # system parameters
                        cellvars    # steady-state cellular variables obtained by simulation
                        ):
    # Freq = a * p_switch / (b - p_switch) => dFreq/dpswitch = a * b / (b - p_switch) ** 2

    a_coeff = (1 + q_osynth) / cellvars['q_sas_max']

    b_coeff = par['M'] * (1 - par['phi_q']) / ((1 + cellvars['chi_switch']) * par['n_switch']) \
                * (cellvars['q_switch_max'] / cellvars['q_sas_max'])

    return a_coeff * b_coeff / (b_coeff - p_switch) ** 2


# FINDING THE BIFURCATIONS: PARAMETRIC APPROACH ------------------------------------------------------------------------
# difference of gradients at the fixed point for a given value of F_switch
def gradiff_from_F(F,   # value of F_switch
                   par,     # system parameters
                   cellvars     # steady-state cellular variables obtained by simulation
                   ):
    # reconstruct p_switch and q_osynth
    p_switch = pswitch_from_F(F, par)
    q_osynth = q_osynth_from_F_and_pswitch(F, p_switch, par, cellvars)

    # get the gradients
    dFreal_dpswitch = dFreal_dpswitch_calc(p_switch, par)
    dFreq_dpswitch = dFreq_dpswitch_calc(p_switch, q_osynth, par, cellvars)
    return dFreal_dpswitch - dFreq_dpswitch


# difference of gradients at the fixed point for a given value of p_switch
def gradiff_from_pswitch(p_switch,  # value of p_switch
                         par,    # system parameters
                         cellvars       # steady-state cellular variables obtained by simulation
                         ):
    # reconstruct F_switch and q_osynth
    F = F_real_calc(p_switch, par)
    q_osynth = q_osynth_from_F_and_pswitch(F, p_switch, par, cellvars)

    # get the gradients
    dFreal_dpswitch = dFreal_dpswitch_calc(p_switch, par)
    dFreq_dpswitch = dFreq_dpswitch_calc(p_switch, q_osynth, par, cellvars)
    return dFreal_dpswitch - dFreq_dpswitch


# find p_switch value for a given real value of F_switch
def pswitch_from_F(F, par):
    return (par['K_switch'] / par['I_switch']) * \
        ((F - par['baseline_switch']) / (1 - F)) ** (1 / par['eta_switch'])


# find the value of q_osynth yielding a fixed point for given F_switch and p_switch values
def q_osynth_from_F_and_pswitch(F, p_switch, par, cellvars):
    return F * cellvars['q_sas_max'] / (p_switch * (1 + cellvars['chi_switch'])) * (
                par['M'] * (1 - par['phi_q']) / par['n_switch'] * cellvars['q_switch_max'] / cellvars['q_sas_max'] - p_switch * (1 + cellvars['chi_switch'])
            ) - 1


# just for convenience, can get a pair of q_osynth and p_switch values
def pswitch_and_q_osynth_from_F(F, par, cellvars):
    p_switch = pswitch_from_F(F, par)
    q_osynth = q_osynth_from_F_and_pswitch(F, p_switch, par, cellvars)
    return jnp.array([p_switch,q_osynth])


# upper bound for q_osynth to find the saddle point at the lower bifurcation (inflexion in real F_switch values)
def pswitch_inflexion_in_Freal(par):
    return ((par['eta_switch'] - 1) / (par['eta_switch'] + 1)) ** (1 / par['eta_switch']) * (
                par['K_switch'] / par['I_switch'])

# FINDING THE STEADY STATES: PARAMETRIC APPROACH -----------------------------------------------------------------------
# difference of Freal and Freq for a given p_switch value
def diff_from_pswitch(pswitch, q_osynth, par, cellvars):
    # get the F values
    F_real = F_real_calc(pswitch, par)
    F_req = F_req_calc(pswitch, q_osynth, par, cellvars)
    return F_real - F_req

# square of the difference of Freal and Freq for a given p_switch value
def sqdiff_from_pswitch(pswitch, q_osynth, par, cellvars):
    return diff_from_pswitch(pswitch, q_osynth, par, cellvars)**2


# DETERMINING SEARCH INTERVAL BOUNDARIES -------------------------------------------------------------------------------
# find the maximum possible value of p_switch
def find_p_switch_sup(par, cellvars):
    return 1 / (1 + cellvars['chi_switch']) * \
        par['M'] * (1 - par['phi_q']) / par['n_switch'] * \
        cellvars['q_switch_max'] / (1+cellvars['q_sas_max'])


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
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    cellmodel_par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(cellmodel_par)  # get default initial conditions

    # add reference tracker switcher
    cellmodel_par_with_refswitch, ref_switcher = cellmodel_auxil.add_reference_switcher(cellmodel_par,
                                                                                        # cell model parameters
                                                                                        refsws.no_switching_initialise,
                                                                                        # function initialising the reference switcher
                                                                                        refsws.no_switching_switch
                                                                                        # function switching the references to be tracked
                                                                                        )

    # load synthetic genetic modules (self-activating switch + unexpressed const. gene) and the controller (constant irrelevant input)
    odeuus_complete, \
        module1_F_calc, module2_F_calc, controller_action, controller_update, \
        par, init_conds, controller_memo0, \
        synth_genes_total_and_each, synth_miscs_total_and_each, \
        controller_memos, controller_dynvars, controller_ctrledvar, \
        modules_name2pos, modules_styles, controller_name2pos, controller_styles, \
        module1_v_with_F_calc, module2_v_with_F_calc = cellmodel_auxil.add_modules_and_controller(
        # module 1
        gms.sas_initialise,  # function initialising the circuit
        gms.sas_ode,  # function defining the circuit ODEs
        gms.sas_F_calc,
        # function calculating the circuit genes' transcription regulation functions
        # module 2
        gms.cicc_initialise,  # function initialising the circuit
        gms.cicc_ode,  # function defining the circuit ODEs
        gms.cicc_F_calc,
        # function calculating the circuit genes' transcription regulation functions
        # controller
        ctrls.cci_initialise,  # function initialising the controller
        ctrls.cci_action,  # function calculating the controller action
        ctrls.cci_ode,  # function defining the controller ODEs
        ctrls.cci_update,  # function updating the controller based on measurements
        # cell model parameters and initial conditions
        cellmodel_par_with_refswitch, init_conds)

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
    sgp4j=cellmodel_auxil.synth_gene_params_for_jax(par, synth_genes)

    # DETERMINISTIC SIMULATION
    # set simulation parameters
    tf = (0.0, 50.0)  # simulation time frame - assuming the steady state is reached in 50 hours

    # measurement time step
    meastimestep = 50.0  # hours

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
                                cellmodel_auxil.x0_from_init_conds(init_conds,
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
    es, _, F_rs, _, _, _, _ = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D(ts, xs, par, synth_genes, synth_miscs, modules_name2pos)
    e = es[-1]
    F_r = F_rs[-1]

    return e, F_r


# get normalised maximum burden values and degradation coefficients for the synthetic genes, given the steady-state e and F_r
# CURRENTLY DOES NOT HANDLE NON-ZERO CHLORAMPHENICOL CONCENTRATIONS
def find_max_qs_and_chis(
        e_ss,  # steady-state translation elongation rate
        F_r_ss,  # steady-state ribosomal gene transcription function
        par,  # system parameters
        synth_genes,  # list of synthetic genes
        modules_name2pos,  # dictionary mapping gene names to their positions in the state vector
    ):
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes

    # get the jaxed synthetic gene parameters
    sgp4j = cellmodel_auxil.synth_gene_params_for_jax(par, synth_genes)

    # get the apparent ribosome-mRNA dissociation constants
    k_a = k_calc(e_ss, par['k+_a'], par['k-_a'], par['n_a'])  # metabolic genes
    k_r = k_calc(e_ss, par['k+_r'], par['k-_r'], par['n_r'])  # ribosomal genes
    k_het = np.array(k_calc(e_ss, sgp4j[0], sgp4j[1], sgp4j[2]))  # array for all heterologous genes

    # get the NON-NORMALISED native gene expression burden
    xi_a = xi_calc(1, 1, par['c_a'], par['a_a'], k_a)  # metabolic genes (F=1 since genes are constitutive)
    xi_r = xi_calc(1, F_r_ss, par['c_r'], par['a_r'], k_r)  # ribosomal genes
    xi_native = xi_a + xi_r  # total native gene expression burden

    # for each synthetic gene, get the NORMALISED, MAXIMUM POSSIBLE burden value and the degradation coefficient
    qmaxs = {}  # dictionary for the normalised maximum burden values
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
        qmaxs['q_'+gene+'_max'] = xi_gene_max / xi_native  # record the normalised maximum burden value
        chis['chi_'+gene] = chi_calc(par['d_' + gene], e_ss,
                                         0, # assuming no chloramphenicol present
                                         par, xi_gene_max, xi_native)  # record the degradation coefficient

    # return
    return qmaxs, chis

# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    # set up jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

    # initialise cell model
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    cellmodel_par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(cellmodel_par)  # get default initial conditions

    # add reference tracker switcher
    cellmodel_par_with_refswitch, ref_switcher = cellmodel_auxil.add_reference_switcher(cellmodel_par,
                                                                                        # cell model parameters
                                                                                        refsws.no_switching_initialise,
                                                                                        # function initialising the reference switcher
                                                                                        refsws.no_switching_switch
                                                                                        # function switching the references to be tracked
                                                                                        )

    # load synthetic genetic modules and the controller
    odeuus_complete, \
        module1_F_calc, module2_F_calc, controller_action, controller_update, \
        par, init_conds, controller_memo0, \
        synth_genes_total_and_each, synth_miscs_total_and_each, \
        controller_memos, controller_dynvars, controller_ctrledvar, \
        modules_name2pos, modules_styles, controller_name2pos, controller_styles, \
        module1_v_with_F_calc, module2_v_with_F_calc = cellmodel_auxil.add_modules_and_controller(
        # module 1
        gms.sas_initialise,  # function initialising the circuit
        gms.sas_ode,  # function defining the circuit ODEs
        gms.sas_F_calc,
        # function calculating the circuit genes' transcription regulation functions
        # module 2
        gms.cicc_initialise,  # function initialising the circuit
        gms.cicc_ode,  # function defining the circuit ODEs
        gms.cicc_F_calc,
        # function calculating the circuit genes' transcription regulation functions
        # controller
        ctrls.cci_initialise,  # function initialising the controller
        ctrls.cci_action,  # function calculating the controller action
        ctrls.cci_ode,  # function defining the controller ODEs
        ctrls.cci_update,  # function updating the controller based on measurements
        # cell model parameters and initial conditions
        cellmodel_par_with_refswitch, init_conds)

    # unpack the synthetic genes and miscellaneous species lists
    synth_genes = synth_genes_total_and_each[0]
    module1_genes = synth_genes_total_and_each[1]
    module2_genes = synth_genes_total_and_each[2]
    synth_miscs = synth_miscs_total_and_each[0]
    module1_miscs = synth_miscs_total_and_each[1]
    module2_miscs = synth_miscs_total_and_each[2]

    # find the steady-state cellular variables
    e_ss, F_r_ss = get_ss_F_and_e(init_conds['s'])
    qmaxs, chis = find_max_qs_and_chis(e_ss, F_r_ss, par, synth_genes, modules_name2pos)

    # SPECIFY SYNTHETIC GENE PARAMETERS
    # self-activating switch
    par['c_switch'] = 100.0
    par['a_switch'] = 1500.0
    par['K_switch'] = 250.0
    par['I_switch'] = 0.2
    par['k+_switch'] = par['k+_ofp']/100

    # chemically cybercontrolled probe
    par['c_ta'] = 100.0
    par['a_ta'] = 10.0
    par['c_b'] = 100.0
    par['a_b'] = 3000.0

    # GET MAXIMUM BURDENS AND DEGRADATION COEFFICIENTS
    e_ss, F_r_ss = get_ss_F_and_e(init_conds['s'])
    qmaxs, chis = find_max_qs_and_chis(e_ss, F_r_ss, par, synth_genes, modules_name2pos)
    cellvars={}
    cellvars.update(qmaxs)
    cellvars.update(chis)
    # add maxmium overall burden from the self-activating switch
    cellvars['q_sas_max']=cellvars['q_switch_max']+cellvars['q_ofp_max']

    print(qmaxs)
    print(chis)

    # FIND BIFURCATION THRESHOLDS
    q_osynth_range = np.linspace(0, 2*cellvars['q_sas_max'], 100)   # consider burdens from other synthetic genes for up to twice the own burden

    # find equilibria for every q_osynth value (to the nearest integer p_switch concentration)
    p_switch_sup=find_p_switch_sup(par, cellvars)  # upper bound for p_switch
    p_switch_range=np.arange(0.0,p_switch_sup+1,1.0)  # range of p_switch values to evaluate
    bifurcation_curve_q_osynth=[]
    bifurcation_curve_p_switch=[]
    for q_osynth in q_osynth_range:
        # get squared differences between real and required F values for every p_switch value
        sqdiff_for_pswitch=functools.partial(sqdiff_from_pswitch, q_osynth=q_osynth, par=par, cellvars=cellvars)
        sqdiffs=sqdiff_for_pswitch(p_switch_range)

        # roots are approximated by the points where the squared difference is lower than around it
        sqdiff_leq_than_prev=np.concatenate(([False],sqdiffs[1:]<=sqdiffs[:-1]))
        sqdiff_leq_than_next=np.concatenate((sqdiffs[:-1]<=sqdiffs[1:], [False]))
        sqdiff_close_to_zero=sqdiffs<1e-3
        approx_minima = np.where(np.logical_and(np.logical_and(sqdiff_leq_than_prev, sqdiff_leq_than_next),
                                                sqdiff_close_to_zero
                                                ))[0]

        # record the bifurcation points
        for minima in approx_minima:
            bifurcation_curve_q_osynth.append(q_osynth)
            bifurcation_curve_p_switch.append(p_switch_range[minima])

    # plot the bifurcation curve
    bkplot.output_file('bifurcation_curve.html')
    bif_fig = bkplot.figure(
        frame_width=480,
        frame_height=360,
        x_axis_label="q_p, probe burden",
        y_axis_label="p_switch, switch protein level",
        x_range=(min(q_osynth_range), max(q_osynth_range)),
        y_range=(0, p_switch_sup),
        title='Analytical bifurcation curve for self-activating switch',
        tools="box_zoom,pan,hover,reset"
    )
    # plot the controlled variable vs control action
    # bif_fig.line(x=np.array(bifurcation_curve_q_osynth),
    #              y=np.array(bifurcation_curve_p_switch),
    #              line_width=1.5, line_color='black',
    #              legend_label='true steady states')
    bif_fig.scatter(x=np.array(bifurcation_curve_q_osynth),
                    y=np.array(bifurcation_curve_p_switch),
                    marker='circle', size=5,
                    color='black', legend_label='true steady states')
    # legend formatting
    bif_fig.legend.label_text_font_size = "8pt"
    bif_fig.legend.location = "top_right"
    bif_fig.legend.click_policy = 'hide'

    # plot the real and required F_switch values at touch-below bifurcation
    # p_switch_range = np.linspace(0, p_switch_sup, 100)
    F_real_range = F_real_calc(p_switch_range, par)
    F_req_range = F_req_calc(p_switch_range, 0.4, par, cellvars)
    touch_fig = bkplot.figure(
        frame_width=480,
        frame_height=360,
        x_axis_label="p_switch, switch protein level",
        y_axis_label="F",
        x_range=(0, p_switch_sup),
        y_range=(0, 1),
        title='F vs p_switch',
        tools="box_zoom,pan,hover,reset"
    )
    # plot the controlled variable vs control action
    touch_fig.line(x=p_switch_range,
                   y=F_real_range,
                   line_width=1.5, line_color='black',
                   legend_label='F_real')
    touch_fig.line(x=p_switch_range,
                     y=F_req_range,
                     line_width=1.5, line_color='red',
                     legend_label='F_req')
    # legend formatting
    touch_fig.legend.label_text_font_size = "8pt"
    touch_fig.legend.location = "top_right"
    touch_fig.legend.click_policy = 'hide'


    # show plot
    bkplot.save(bklayouts.column(bif_fig, touch_fig))


    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


