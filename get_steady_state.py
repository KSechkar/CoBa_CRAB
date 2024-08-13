'''
GET_STEADY_STATE.PY - get the system's steady state and steady-state vaalues of important variables
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import functools
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
import pickle
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, palettes as bkpalettes, transform as bktransform
from math import pi
import time

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
import synthetic_circuits as circuits
from cell_model import *

# SIMULATE THE SYSTEM TO GET ITS STEADY STATE [vmappable]---------------------------------------------------------------
def get_steady_state(par,    # dictionary with model parameters
                        ode_with_circuit,  # ODE function for the cell with the synthetic gene circuit
                        x0,  # initial condition vector
                        num_circuit_genes, num_circuit_miscs, circuit_name2pos, sgp4j, # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder, relevant synthetic gene parameters in jax.array form
                        t_to_steady, rtol=1e-6, atol=1e-6    # simulation parameters: time until steady state, when to save the system's state, relative and absolute tolerances
                        ):
    # define the ODE term
    vector_field = lambda t, y, args: ode_with_circuit(t, y, args)
    term = ODETerm(vector_field)

    # define arguments of the ODE term
    args = (
        par,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        num_circuit_genes, num_circuit_miscs,  # number of genes and miscellaneous species in the circuit
        sgp4j  # relevant synthetic gene parameters in jax.array form
    )

    # define the solver
    solver = Kvaerno3()

    # define the timestep controller
    stepsize_controller = PIDController(rtol=rtol, atol=atol)

    # solvew the ODE
    sol = diffeqsolve(term, solver,
                      args=args,
                      t0=0, t1=t_to_steady, dt0=0.1, y0=x0,
                      max_steps=None,
                      stepsize_controller=stepsize_controller)

    return sol

# GET STEADY STATE VALUES OF VARIABLES FOR ANALYTICAL DERIVATIONS ------------------------------------------------------
# auxiliary: calculate the burden value for a gene
def xi_calc(func,F,c,a,k):
    return func*F*c*a/k

# auxiliary: protease action adjustment factor for a gene
def P_calc(d_gene,
           e, h, par, xi_prot, xi_r):
    return ((par['K_D'] + h) / par['K_D']) *\
        d_gene * par['M']/e *\
        par['n_r']/par['n_prot'] * xi_prot/xi_r

# get the steady state values
def values_for_analytical(par,  # dictionary with model parameters
           ode_with_circuit,  # ODE function for the cell with synthetic circuit
           init_conds,  # initial condition DICTIONARY for the cell with synthetic circuit
           circuit_genes, circuit_miscs, circuit_name2pos,  # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
           circuit_F_calc,  # function for calculating the synthetic genes' transcription regulation functions
           t_to_steady=50, rtol=1e-6, atol=1e-6     # simulation parameters: time until steady state, relative and absolute tolerances
           ):
    # auxiliary tools for simulating the model and plotting simulation outcomes
    cellmodel_auxil = CellModelAuxiliary()

    # GET E, F_R AND H VALUES ------------------------------------------------------------------------------------------
    # parameters for simulation to get e and F_r values - not taking into account synthetic genes save for cat
    par4eFr = par.copy()
    for gene in circuit_genes:
        if(gene != 'cat'):
            par4eFr['c_'+gene] = 0
    # also not taking into account integrase action
    par4eFr['k_sxf'] = 0 # make the strand exchange rate equal to 0

    # get the jaxed synthetic gene parameters
    sgp4j = cellmodel_auxil.synth_gene_params_for_jax(par4eFr, circuit_genes)

    # simulate the cell model to find the steady state
    sol=get_steady_state(par4eFr, ode_with_circuit,
                            cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),
                            len(circuit_genes), len(circuit_miscs), circuit_name2pos, sgp4j, t_to_steady, rtol, atol)
    ts = np.array(sol.ts)
    xs = np.array(sol.ys)

    # get the e and F_r values - now with ORIGINAL parameters
    es,_,F_rs,_,_,_,_,_ = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnodeg(ts,xs,par,circuit_genes,circuit_miscs,circuit_name2pos)
    e=es[-1]
    F_r=F_rs[-1]

    # get the internal chloramphenicol concentration
    h=xs[-1,7]

    # get the apparent ribosome-mRNA dissociation constants for native, cat and punisher genes
    k_a=k_calc(e,par['k+_a'],par['k-_a'],par['n_a'])  # metabolic genes
    k_r=k_calc(e,par['k+_r'],par['k-_r'],par['n_r'])  # ribosomal genes
    k_het=np.array(k_calc(e,sgp4j[0],sgp4j[1],sgp4j[2]))  # array for all heterologous genes
    k_cat = k_het[circuit_name2pos['k_cat']]  # chloramphenicol resistance gene
    k_switch = k_het[circuit_name2pos['k_switch']]  # self-activating switch gene
    k_int = k_het[circuit_name2pos['k_int']]  # integrase gene
    k_prot = k_het[circuit_name2pos['k_prot']]  # protease gene

    # GETTING THE BURDEN VALUES: INVOLVES ANOTHER SIMULATION TO FIND STEADY-STATE F VALUES FOR SYNTHETIC GENES ---------
    # parameters for simulation to get e and F_r values - not taking into account integrase action
    par4xi = par.copy()
    par4xi['k_sxf'] = 0  # make the strand exchange rate equal to 0

    # get the jaxed synthetic gene parameters
    sgp4j = cellmodel_auxil.synth_gene_params_for_jax(par4xi, circuit_genes)

    # Simulate the cell model t find the steady state
    vector_field = lambda t, y, args: ode_with_circuit(t, y, args)
    term = ODETerm(vector_field)
    args = (
        par4xi,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        len(circuit_genes), len(circuit_miscs),  # number of genes and miscellaneous species in the circuit
        sgp4j  # relevant synthetic gene parameters in jax.array form
    )
    sol = get_steady_state(par4xi, ode_with_circuit,
                            cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),
                            len(circuit_genes), len(circuit_miscs), circuit_name2pos, sgp4j, t_to_steady, rtol, atol)
    ts = np.array(sol.ts)
    xs = np.array(sol.ys)

    # get the transcription regulation function values
    Fs = np.array(circuit_F_calc(ts[-1],xs[-1,:],par,circuit_name2pos))

    # get the burden (xi) values for native genes, chloramphenicol resistance gene and punisher genes
    xi_a = xi_calc(1,1,par['c_a'],par['a_a'],k_a)  # metabolic genes (F=1 since genes are constitutive)
    xi_r = xi_calc(1,F_r,par['c_r'],par['a_r'],k_r)  # ribosomal genes
    xi_cat = xi_calc(par['func_cat'],1,par['c_cat'],par['a_cat'],k_cat)  # chloramphenicol resistance gene (F=1 since the gene is constitutive)
    xi_switch_max = xi_calc(par['func_switch'],1,par['c_switch'],par['a_switch'],k_switch)  # MAXIMUM  for the self-activating switch gene (F=1 for MAXIMUM POSSIBLE EXPRESSION)
    xi_int_max = xi_calc(par['func_int'],1,par['c_switch'],par['a_switch'],k_int)  # MAXIMUM for the integrase gene (F=1 for MAXIMUM POSSIBLE EXPRESSION). IMPORTANT: INTEGRASE EXPRESSED FROM SAME OPERON AS SWITCH
    xi_prot = xi_calc(par['func_prot'],1,par['c_prot'],par['a_prot'],k_prot)  # MAXIMUM for the protease gene (F=1 for MAXIMUM POSSIBLE EXPRESSION)

    # get the total burden (xi) values for all remaining synthetic genes
    xi_other_genes = 0
    for gene in circuit_genes:
        if(gene not in ['cat','switch','int','prot']):
            xi_other_genes += xi_calc(par['func_'+gene],Fs[circuit_name2pos['F_'+gene]],par['c_'+gene],par['a_'+gene],k_het[circuit_name2pos['k_'+gene]])

    # package all burden values into a dictionary
    xis={'a':xi_a, 'r':xi_r, 'cat':xi_cat, 'switch (max)':xi_switch_max, 'int (max)':xi_int_max, 'prot':xi_prot, 'other':xi_other_genes}

    # protein degradation adjustment factor
    P_switch = P_calc(par['d_switch'],e,h,par,xi_prot,xi_r)
    P_int = P_calc(par['d_int'],e,h,par,xi_prot,xi_r)

    # package all adjustment factors into a dictionary
    Ps={'switch':P_switch, 'int':P_int}

    # return
    return e, F_r, h, xis, Ps

# get the steady state values - for the case when the switch prot. and integrase expressed from separate operons
def values_for_analytical_sep(par,  # dictionary with model parameters
           ode_with_circuit,  # ODE function for the cell with synthetic circuit
           init_conds,  # initial condition DICTIONARY for the cell with synthetic circuit
           circuit_genes, circuit_miscs, circuit_name2pos,  # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
           circuit_F_calc,  # function for calculating the synthetic genes' transcription regulation functions
           t_to_steady=50, rtol=1e-6, atol=1e-6     # simulation parameters: time until steady state, relative and absolute tolerances
           ):
    # auxiliary tools for simulating the model and plotting simulation outcomes
    cellmodel_auxil = CellModelAuxiliary()

    # GET E, F_R AND H VALUES ------------------------------------------------------------------------------------------
    # parameters for simulation to get e and F_r values - not taking into account synthetic genes save for cat
    par4eFr = par.copy()
    for gene in circuit_genes:
        if(gene != 'cat'):
            par4eFr['c_'+gene] = 0
    # also not taking into account integrase action
    par4eFr['k_sxf'] = 0 # make the strand exchange rate equal to 0

    # get the jaxed synthetic gene parameters
    sgp4j = cellmodel_auxil.synth_gene_params_for_jax(par4eFr, circuit_genes)

    # simulate the cell model to find the steady state
    sol=get_steady_state(par4eFr, ode_with_circuit,
                            cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),
                            len(circuit_genes), len(circuit_miscs), circuit_name2pos, sgp4j, t_to_steady, rtol, atol)
    ts = np.array(sol.ts)
    xs = np.array(sol.ys)

    # get the e and F_r values - now with ORIGINAL parameters
    es,_,F_rs,_,_,_,_,_ = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnodeg(ts,xs,par,circuit_genes,circuit_miscs,circuit_name2pos)
    e=es[-1]
    F_r=F_rs[-1]

    # get the internal chloramphenicol concentration
    h=xs[-1,7]

    # get the apparent ribosome-mRNA dissociation constants for native, cat and punisher genes
    k_a=k_calc(e,par['k+_a'],par['k-_a'],par['n_a'])  # metabolic genes
    k_r=k_calc(e,par['k+_r'],par['k-_r'],par['n_r'])  # ribosomal genes
    k_het=np.array(k_calc(e,sgp4j[0],sgp4j[1],sgp4j[2]))  # array for all heterologous genes
    k_cat = k_het[circuit_name2pos['k_cat']]  # chloramphenicol resistance gene
    k_switch = k_het[circuit_name2pos['k_switch']]  # self-activating switch gene
    k_int = k_het[circuit_name2pos['k_int']]  # integrase gene
    k_prot = k_het[circuit_name2pos['k_prot']]  # protease gene

    # GETTING THE BURDEN VALUES: INVOLVES ANOTHER SIMULATION TO FIND STEADY-STATE F VALUES FOR SYNTHETIC GENES ---------
    # parameters for simulation to get e and F_r values - not taking into account integrase action
    par4xi = par.copy()
    par4xi['k_sxf'] = 0  # make the strand exchange rate equal to 0

    # get the jaxed synthetic gene parameters
    sgp4j = cellmodel_auxil.synth_gene_params_for_jax(par4xi, circuit_genes)

    # Simulate the cell model t find the steady state
    vector_field = lambda t, y, args: ode_with_circuit(t, y, args)
    term = ODETerm(vector_field)
    args = (
        par4xi,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        len(circuit_genes), len(circuit_miscs),  # number of genes and miscellaneous species in the circuit
        sgp4j  # relevant synthetic gene parameters in jax.array form
    )
    sol = get_steady_state(par4xi, ode_with_circuit,
                            cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),
                            len(circuit_genes), len(circuit_miscs), circuit_name2pos, sgp4j, t_to_steady, rtol, atol)
    ts = np.array(sol.ts)
    xs = np.array(sol.ys)

    # get the transcription regulation function values
    Fs = np.array(circuit_F_calc(ts[-1],xs[-1,:],par,circuit_name2pos))

    # get the burden (xi) values for native genes, chloramphenicol resistance gene and punisher genes
    xi_a = xi_calc(1,1,par['c_a'],par['a_a'],k_a)  # metabolic genes (F=1 since genes are constitutive)
    xi_r = xi_calc(1,F_r,par['c_r'],par['a_r'],k_r)  # ribosomal genes
    xi_cat = xi_calc(par['func_cat'],1,par['c_cat'],par['a_cat'],k_cat)  # chloramphenicol resistance gene (F=1 since the gene is constitutive)
    xi_switch_max = xi_calc(par['func_switch'],1,par['c_switch'],par['a_switch'],k_switch)  # MAXIMUM  for the self-activating switch gene (F=1 for MAXIMUM POSSIBLE EXPRESSION)
    xi_int_max = xi_calc(par['func_int'],1,par['c_switch'],par['a_switch'],k_int)  # MAXIMUM for the integrase gene (F=1 for MAXIMUM POSSIBLE EXPRESSION). IMPORTANT: INTEGRASE EXPRESSED FROM SAME OPERON AS SWITCH
    xi_prot = xi_calc(par['func_prot'],1,par['c_prot'],par['a_prot'],k_prot)  # MAXIMUM for the protease gene (F=1 for MAXIMUM POSSIBLE EXPRESSION)

    # get the total burden (xi) values for all remaining synthetic genes
    xi_other_genes = 0
    for gene in circuit_genes:
        if(gene not in ['cat','switch','int','prot']):
            xi_other_genes += xi_calc(par['func_'+gene],Fs[circuit_name2pos['F_'+gene]],par['c_'+gene],par['a_'+gene],k_het[circuit_name2pos['k_'+gene]])

    # package all burden values into a dictionary
    xis={'a':xi_a, 'r':xi_r, 'cat':xi_cat, 'switch (max)':xi_switch_max, 'int (max)':xi_int_max, 'prot':xi_prot, 'other':xi_other_genes}

    # protein degradation adjustment factor
    P_switch = P_calc(par['d_switch'],e,h,par,xi_prot,xi_r)
    P_int = P_calc(par['d_int'],e,h,par,xi_prot,xi_r)

    # package all adjustment factors into a dictionary
    Ps={'switch':P_switch, 'int':P_int}

    # return
    return e, F_r, h, xis, Ps