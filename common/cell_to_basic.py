'''
CELL_TO_BASIC.PY: Simulate the cell model (for a single synthetic gene with given parameters)
to find the parameters of the basic model yielding the same output
'''

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax.numpy as jnp

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
from cell_model_case.cell_model import ModelAuxiliary as CellModelAuxiliary, ode_sim as cell_ode_sim  # cell model import
import cell_model_case.cell_genetic_modules as cell_gms
import common.controllers as ctrls
import common.reference_switchers as refsws
import common.ode_solvers as odesols

# PICK OUT A GIVEN GENE'S PARAMETERS RELEVANT TO SETTING THE BASIC MODEL PARAMETERS ------------------------------------
def pick_params_for_cell2basic(
        gene_name,  # name of the gene to be characterised
        par
    ):
    picked_params = {}

    picked_params['c_' + gene_name] = par['c_' + gene_name]  # copy no. (nM)
    picked_params['a_' + gene_name] = par['a_' + gene_name]  # promoter strength (unitless)
    picked_params['b_' + gene_name] = par['b_' + gene_name]  # mRNA decay rate (/h)
    picked_params['k+_' + gene_name] = par['k+_' + gene_name]  # ribosome binding rate (/h/nM)
    picked_params['k-_' + gene_name] = par['k-_' + gene_name]  # ribosome unbinding rate (/h)
    picked_params['n_' + gene_name] = par['n_' + gene_name]  # protein length (aa)

    # gene functionality is a legacy parameter, it being 1.0 just means the gene is here
    if not(('func_'+gene_name) in par.keys()):
        picked_params['func_'+gene_name] = 1.0

    return picked_params


# CELL MODEL -> BASIC PARAMETER CONVERTER ------------------------------------------------------------------------------
# find the basic model parameters for native genes
# (run the function for finding synthetic gene parameters but with zero copy number)
def cell2basic(
        cellmodel_params,  # parameters of the cell model
        nutr_qual  # culture medium's nutrient quality
    ):
    # get a dictionary of default synthetic gene parameters
    synth_gene_params, _, _, _, _, _ = cell_gms.constfp_initialise()
    # set the copy number to zero
    for key in synth_gene_params.keys():
        if(key[0:2]=='c_'):
            synth_gene_params[key] = 0.0

        # GET DEFAULT CELL MODEL PARAMETERS AND NUTRIENT QUALITY IF NOT PROVIDED
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes

    # PREPARE FOR SIMULATION
    # initialise cell model
    cellmodel_par = cellmodel_auxil.default_params()  # get default parameter values
    cellmodel_init_conds = cellmodel_auxil.default_init_conds(cellmodel_par)  # get default initial conditions

    # add reference tracker switcher
    model_par_with_refswitch, ref_switcher = cellmodel_auxil.add_reference_switcher(cellmodel_par,
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
        par, cellmodel_init_conds, controller_memo0, \
        synth_genes_total_and_each, synth_miscs_total_and_each, \
        controller_memos, controller_dynvars, controller_ctrledvar, \
        modules_name2pos, modules_styles, controller_name2pos, controller_styles, \
        module1_v_with_F_calc, module2_v_with_F_calc = cellmodel_auxil.add_modules_and_controller(
        # module 1
        cell_gms.constfp_initialise,  # function initialising the circuit
        cell_gms.constfp_ode,  # function defining the circuit ODEs
        cell_gms.constfp_F_calc,  # function calculating the circuit genes' transcription regulation functions
        cell_gms.constfp_specterms,  # function calculating the circuit genes' effective mRNA levels
        # module 2
        cell_gms.nocircuit_initialise,  # function initialising the circuit
        cell_gms.nocircuit_ode,  # function defining the circuit ODEs
        cell_gms.nocircuit_F_calc,  # function calculating the circuit genes' transcription regulation functions
        cell_gms.nocircuit_specterms,  # function calculating the circuit genes' effective mRNA levels
        # controller
        ctrls.cci_initialise,  # function initialising the controller
        ctrls.cci_action,  # function calculating the controller action
        ctrls.cci_ode,  # function defining the controller ODEs
        ctrls.cci_update,  # function updating the controller based on measurements
        # cell model parameters and initial conditions
        model_par_with_refswitch, cellmodel_init_conds)

    # unpack the synthetic genes and miscellaneous species lists
    synth_genes = synth_genes_total_and_each[0]
    module1_genes = synth_genes_total_and_each[1]
    module2_genes = synth_genes_total_and_each[2]
    synth_miscs = synth_miscs_total_and_each[0]
    module1_miscs = synth_miscs_total_and_each[1]
    module2_miscs = synth_miscs_total_and_each[2]

    # LOAD THE PARAMETERS TO BE CONSIDERED
    # copy the relevant cell model parameters
    for key in cellmodel_params.keys():
        par[key] = cellmodel_params[key]
    # copy the relevant gene parameters
    cellmodel_init_conds.update({'sigma': nutr_qual})

    # set synthetic gene degradation and maturation rates to zero as irrelevant for the RC factor
    par['d_ofp'] = 0.0
    par['m_ofp'] = 0.0

    # SET CONTROLLER PARAMETERS - irrelevant for the steady-state retrieval
    refs = jnp.array([0.0])
    control_delay = 0  # control action delay
    u0 = 0.0  # initial control action

    # get the jaxed synthetic gene parameters
    sgp4j = cellmodel_auxil.synth_gene_params_for_jax(par, synth_genes)

    # DETERMINISTIC SIMULATION
    # set simulation parameters
    tf = (0.0, 48.0)  # simulation time frame - assuming the steady state is reached in 48 hours

    # measurement time step
    meastimestep = 48.0  # hours - only care about the steady state at the end

    # choose ODE solver
    ode_solver, us_size = odesols.create_euler_solver(odeuus_complete,
                                                      control_delay=control_delay,
                                                      meastimestep=meastimestep,
                                                      euler_timestep=1e-5)

    # solve ODE
    ts_jnp, xs_jnp, \
        ctrl_memorecord_jnp, uexprecord_jnp, \
        refrecord_jnp = cell_ode_sim(par,  # model parameters
                                     ode_solver,  # ODE solver for the cell with the synthetic gene circuit
                                     odeuus_complete,
                                     # ODE function for the cell with the synthetic gene circuit and the controller (also gives calculated and experienced control actions)
                                     controller_ctrledvar,  # name of the variable read and steered by the controller
                                     controller_update, controller_action,
                                     # function for updating the controller memory and calculating the control action
                                     cellmodel_auxil.x0_from_init_conds(cellmodel_init_conds,
                                                                        par,
                                                                        synth_genes, synth_miscs, controller_dynvars,
                                                                        modules_name2pos,
                                                                        controller_name2pos),
                                     # initial condition VECTOR
                                     controller_memo0,  # initial controller memory record
                                     u0,
                                     # initial control action, applied before any measurement-informed actions reach the system
                                     (len(synth_genes), len(module1_genes), len(module2_genes)),
                                     # number of synthetic genes
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

    # LOOK AT THE STEADY STATE
    # get steady-state translation elongation rate and resource competition denominator
    es, _, _, _, _, _, Ds = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D(ts, xs, par,
                                                                  synth_genes, synth_miscs,
                                                                  modules_name2pos,
                                                                  module1_specterms, module2_specterms)
    e = es[-1]
    D = Ds[-1]
    # assuming that other native proteins are the same length as metabolic and housekeeping proteins in the cell model, i.e. n_o=n_a=n_q=300 amino acids
    n_o = par['n_a']
    # get the steady-state mass fractions of ribosomal proteins
    phi_r = par['n_r'] * xs[-1, 3] / par['M']
    # get the steady-state mass fraction of the synthetic protein
    phi_ofp = par['n_ofp'] * (xs[-1, modules_name2pos['p_ofp']] + xs[-1, modules_name2pos['ofp_mature']]) / par['M']
    # get the steady-state mass fraction of other native proteins
    phi_o = 1 - phi_r - phi_ofp

    # GET THE BASIC MODEL PARAMETERS FROM THE STEADY STATE
    # get the basic host cell model parameters
    basic_host_params = {'M': par['M'],
                         'n_r': par['n_r'], 'n_o': n_o,
                         'e': e,
                         'q_r': float(phi_r * (D - 1)), 'q_o': float(phi_o * (D - 1))}

    # return the dictionary of host cell model parameters
    return basic_host_params


# MAIN FUNCTION  -------------------------------------------------------------------------------------------------------
def main():
    circ_par = {}

    # set the parameters for the synthetic genes
    # dummy constitituive reporter just to ensure correct simulation
    circ_par['c_ofp'] = 0.0

    # initialise cell model
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    cellmodel_par = cellmodel_auxil.default_params()  # get default parameter values
    cell_init_conds = cellmodel_auxil.default_init_conds(cellmodel_par)  # get default initial conditions

    # add reference tracker switcher
    model_par_with_refswitch, ref_switcher = cellmodel_auxil.add_reference_switcher(cellmodel_par,
                                                                                # cell model parameters
                                                                                refsws.timed_switching_initialise,
                                                                                # function initialising the reference switcher
                                                                                refsws.timed_switching_switch
                                                                                # function switching the references to be tracked
                                                                                )

    # load synthetic genetic modules and the controller
    odeuus_complete, \
        module1_F_calc, module2_F_calc, \
        module1_specterms, module2_specterms, \
        controller_action, controller_update, \
        par, init_conds, controller_memo0, \
        synth_genes_total_and_each, synth_miscs_total_and_each, \
        controller_memos, controller_dynvars, controller_ctrledvar, \
        modules_name2pos, modules_styles, controller_name2pos, controller_styles, \
        module1_v_with_F_calc, module2_v_with_F_calc = cellmodel_auxil.add_modules_and_controller(
        # module 1
        cell_gms.constfp_initialise,  # function initialising the circuit
        cell_gms.constfp_ode,  # function defining the circuit ODEs
        cell_gms.constfp_F_calc,  # function calculating the circuit genes' transcription regulation functions
        cell_gms.constfp_specterms,
        # function calculating the circuit genes effective mRNA levels (due to possible co-expression from the same operons)
        # module 2
        cell_gms.nocircuit_initialise,  # function initialising the circuit
        cell_gms.nocircuit_ode,  # function defining the circuit ODEs
        cell_gms.nocircuit_F_calc,  # function calculating the circuit genes' transcription regulation functions
        cell_gms.nocircuit_specterms,
        # function calculating the circuit genes effective mRNA levels (due to possible co-expression from the same operons)
        # controller
        ctrls.cci_initialise,  # function initialising the controller
        ctrls.cci_action,  # function calculating the controller action
        ctrls.cci_ode,  # function defining the controller ODEs
        ctrls.cci_update,  # function updating the controller based on measurements
        # cell model parameters and initial conditions
        model_par_with_refswitch, cell_init_conds)

    # unpack the synthetic genes and miscellaneous species lists
    synth_genes = synth_genes_total_and_each[0]
    module1_genes = synth_genes_total_and_each[1]
    module2_genes = synth_genes_total_and_each[2]
    synth_miscs = synth_miscs_total_and_each[0]
    module1_miscs = synth_miscs_total_and_each[1]
    module2_miscs = synth_miscs_total_and_each[2]

    par.update(circ_par)

    print(cell2basic(par, cell_init_conds['sigma']))

    return

# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()