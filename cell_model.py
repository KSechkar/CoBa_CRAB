'''
CELL_MODEL.PY: Python/Jax implementation of the coarse-grained resource-aware E.coli model
Class to enable resource-aware simulation of synthetic gene expression in the cell

Version where THE PROTEASE IS A SYNTHETIC PROTEIN WHOSE CONC. AFFECTS DEGRADATION RATES
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import functools
from diffrax import diffeqsolve, Kvaerno3, Heun, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts

import time

# CIRCUIT AND EXTERNAL INPUT IMPORTS -----------------------------------------------------------------------------------
import genetic_modules as gms
import controllers as ctrls


# CELL MODEL FUNCTIONS -------------------------------------------------------------------------------------------------
# Definitions of functions appearing in the cell model ODEs
# apparent mRNA-ribosome dissociation constant
def k_calc(e, kplus, kminus, n):
    return (kminus + e / n) / kplus


# translation elongation rate
def e_calc(par, tc):
    return par['e_max'] * tc / (tc + par['K_e'])


# growth rate
def l_calc(par, e, B, prodeflux):
    return (e * B - prodeflux) / par['M']


# tRNA charging rate
def nu_calc(par, tu, s):
    return par['nu_max'] * s * (tu / (tu + par['K_nu']))


# tRNA synthesis rate
def psi_calc(par, T):
    return par['psi_max'] * T / (T + par['tau'])


# ribosomal gene transcription regulation function
def Fr_calc(par, T):
    return T / (T + par['tau'])


# CELL MODEL AUXILIARIES -----------------------------------------------------------------------------------------------
# Auxiliries for the cell model - set up default parameters and initial conditions, plot simulation outcomes
class CellModelAuxiliary:
    # INITIALISE
    def __init__(self):
        # plotting colours
        self.gene_colours = {'a': "#EDB120", 'r': "#7E2F8E", 'q': '#C0C0C0',
                             # colours for metabolic, riboozyme, housekeeping genes
                             'het': "#0072BD",
                             'h': "#77AC30"}  # colours for heterologous genes and intracellular chloramphenicol level
        self.tRNA_colours = {'tc': "#000000", 'tu': "#ABABAB"}  # colours for charged and uncharged tRNAs
        return

    # PROCESS SYNTHETIC CIRCUIT MODULE
    # add synthetic circuit to the cell model
    def add_modules_and_controller(self,
                                   # module 1
                                   module1_initialiser,  # function initialising the circuit
                                   module1_ode,  # function defining the circuit ODEs
                                   module1_F_calc, # function calculating the circuit genes' transcription regulation functions
                                   # module 2
                                   module2_initialiser,  # function initialising the circuit
                                   module2_ode,  # function defining the circuit ODEs
                                   module2_F_calc, # function calculating the circuit genes' transcription regulation functions
                                   # controller
                                   controller_initialiser,  # function initialising the controller
                                   controller_action,  # function calculating the control action
                                   controller_ode,  # function defining the controller ODEs
                                   controller_update,  # function updating the controller based on measurements
                                   # cell model parameters and initial conditions
                                   cellmodel_par, cellmodel_init_conds,
                                   # host cell model parameters and initial conditions
                                   # optional support for hybrid simulations
                                   module1_v=None, module2_v=None  # optional stochastic components for the circuit
                                   ):
        # call initialisers
        module1_par, module1_init_conds, module1_genes, module1_miscs, module1_name2pos, module1_styles = module1_initialiser()
        module2_par, module2_init_conds, module2_genes, module2_miscs, module2_name2pos, module2_styles = module2_initialiser()
        controller_par, controller_init_conds, controller_init_memory, controller_name2pos = controller_initialiser()

        # update parameter, initial condition
        cellmodel_par.update(module1_par)
        cellmodel_par.update(module2_par)
        cellmodel_par.update(controller_par)
        cellmodel_init_conds.update(module1_init_conds)
        cellmodel_init_conds.update(module2_init_conds)
        cellmodel_init_conds.update(controller_init_conds)
        
        # parameter merge special case: presence of CAT or protease genes in one module means it's present overall
        cellmodel_par['cat_gene_present'] = module1_par['cat_gene_present'] or module2_par['cat_gene_present']
        cellmodel_par['prot_gene_present'] = module1_par['prot_gene_present'] or module2_par['prot_gene_present']

        # merge style dictionaries for the two modules
        modules_styles = module1_styles.copy()
        modules_styles.update(module2_styles)
        
        # merge name-to-position dictionaries for the two modules: requires shifting the position of the second module's species
        modules_name2pos = module1_name2pos.copy()
        x_entries_for_module_1 = len(module1_genes) * 2 + len(module1_miscs)  # number of entries for the first module in the state vector
        for key in module2_name2pos.keys():
            if((key[0:2]=='m_') or (key[0:2]== 'p')): # for mRNA and protein species, shift position by the number of species in x for the first module - IF the gene is present
                if((cellmodel_par['cat_gene_present'] or key[2:] != 'cat') or (cellmodel_par['prot_gene_present'] or key[2:] != 'prot')):
                    modules_name2pos[key] = module2_name2pos[key] + x_entries_for_module_1
            elif(key[0:2] == 'k_'): # for synthetic mRNA-ribosome dissociation constants, shift position by the number of GENES in the first module
                modules_name2pos[key] = module2_name2pos[key] + len(module1_genes)
            elif(key[0:2] == 'F_'): # for transcription regulation functions, no shift necessary
                modules_name2pos[key] = module2_name2pos[key]
            else: # the rest is miscellaneous species, shift position by the number of species in x for the first module
                modules_name2pos[key] = module2_name2pos[key] + x_entries_for_module_1

        # merge gene and miscellaneous species lists
        synth_genes = module1_genes + module2_genes
        synth_miscs = module1_miscs + module2_miscs

        # join the circuit ODEs with the transcription regulation functions
        module1_ode_with_F_calc = lambda t, x, u, e, l, R, k_het, D, p_prot, par, name2pos: module1_ode(module1_F_calc,
                                                                                                        t, x, u,
                                                                                                        e, l, R,
                                                                                                        k_het, D, p_prot,
                                                                                                        par,
                                                                                                        name2pos)
        module2_ode_with_F_calc = lambda t, x, u, e, l, R, k_het, D, p_prot, par, name2pos: module2_ode(module2_F_calc,
                                                                                                        t, x, u,
                                                                                                        e, l, R,
                                                                                                        k_het, D, p_prot,
                                                                                                        par,
                                                                                                        name2pos)

        # IF stochastic component specified, predefine F_calc for it as well
        if ((module1_v != None) and (module2_v != None)):
            module1_v_with_F_calc = lambda t, x, u, e, l, R, k_het, D, p_prot, mRNA_count_scales, par, name2pos: module1_v(
                module1_F_calc,
                t, x, e, l, R, k_het, D, p_prot,
                mRNA_count_scales,
                par, name2pos)
            module2_v_with_F_calc = lambda t, x, u, e, l, R, k_het, D, p_prot, mRNA_count_scales, par, name2pos: module2_v(
                module2_F_calc,
                t, x, e, l, R, k_het, D, p_prot,
                mRNA_count_scales,
                par, name2pos)
        else:
            module1_v_with_F_calc = None
            module2_v_with_F_calc = None

        # add the geetic module and controller ODEs (as well as control action calculator) to that of the host cell model
        cellmodel_ode = lambda t, x, ctrl_memo, args: ode(t, x, ctrl_memo,
                                                          module1_ode_with_F_calc, module2_ode_with_F_calc,
                                                          controller_ode, controller_action,
                                                          args)

        # return updated ode and parameter, initial conditions, circuit gene (and miscellaneous specie) names
        # name - position in state vector decoder and colours for plotting the circuit's time evolution
        return (cellmodel_ode,
                module1_F_calc, module2_F_calc, controller_action, controller_update,
                cellmodel_par, cellmodel_init_conds,
                synth_genes, synth_miscs,
                modules_name2pos, modules_styles, controller_name2pos,
                module1_v_with_F_calc, module2_v_with_F_calc)

    # package synthetic gene parameters into jax arrays for calculating k values
    def synth_gene_params_for_jax(self, par,  # system parameters
                                  synth_genes  # circuit gene names
                                  ):
        # initialise parameter arrays
        kplus_het = np.zeros(len(synth_genes))
        kminus_het = np.zeros(len(synth_genes))
        n_het = np.zeros(len(synth_genes))
        d_het = np.zeros(len(synth_genes))

        # fill parameter arrays
        for i in range(0, len(synth_genes)):
            kplus_het[i] = par['k+_' + synth_genes[i]]
            kminus_het[i] = par['k-_' + synth_genes[i]]
            n_het[i] = par['n_' + synth_genes[i]]
            d_het[i] = par['d_' + synth_genes[i]]

        # return as a tuple of arrays
        return (jnp.array(kplus_het), jnp.array(kminus_het), jnp.array(n_het), jnp.array(d_het))

    # SET DEFAULTS
    # set default parameters
    def default_params(self):
        '''
        Chloramphenicol-related parameters taken from:
        Gutiérrez Mena J et al. 2022 Dynamic cybergenetic control of bacterial co-culture composition via optogenetic feedback
        All other parameters taken from:
        Sechkar A et al. 2024 A coarse-grained bacterial cell model for resource-aware analysis and design of synthetic gene circuits
        '''

        params = {}  # initialise

        # GENERAL PARAMETERS
        params['M'] = 1.19e9  # cell mass (aa)
        params['phi_q'] = 0.59  # constant housekeeping protein mass fraction

        # GENE EXPRESSION parAMETERS
        # metabolic/aminoacylating genes
        params['c_a'] = 1.0  # copy no. (nM) - convention
        params['b_a'] = 6.0  # mRNA decay rate (/h)
        params['k+_a'] = 60.0  # ribosome binding rate (/h/nM)
        params['k-_a'] = 60.0  # ribosome unbinding rate (/h)
        params['n_a'] = 300.0  # protein length (aa)

        # ribosomal gene
        params['c_r'] = 1.0  # copy no. (nM) - convention
        params['b_r'] = 6.0  # mRNA decay rate (/h)
        params['k+_r'] = 60.0  # ribosome binding rate (/h/nM)
        params['k-_r'] = 60.0  # ribosome unbinding rate (/h)
        params['n_r'] = 7459.0  # protein length (aa)

        # ACTIVATION & RATE FUNCTION PARAMETERS
        params['e_max'] = 7.2e4  # max translation elongation rate (aa/h)
        params['psi_max'] = 4.32e5  # max tRNA syntheis rate per untig rowth rate (nM)
        params['tau'] = 1.0  # ppGpp sensitivity (ribosome transc. and tRNA synth. Hill const)

        # CHLORAMPHENICOL-RELATED PARAMETERS
        params['h_ext'] = 0.0  # chloramphenicol concentration in the culture medium (nM)
        params['diff_h'] = 90.0 * 60  # chloramphenicol diffusion coefficient through the cell membrane (1/h)
        params['K_D'] = 1300.0  # chloramphenicol-ribosome dissociation constant (nM)
        params['K_C'] = (1 / 3) / 60  # dissociation constant for chloramphenicol removal by the cat (chloramphenicol resistance) protein, if present (nM*h)
        params[
            'cat_gene_present'] = 0  # 1 if cat gene is present, 0 otherwise (will be automatically set to 1 if your circuit has a gene titled 'cat' and you haven't messed where you shouldn't)

        # PARAMETERS FITTED TO DATA IN SECHKAR ET AL., 2024
        params['a_a'] = 394464.6979  # metabolic gene transcription rate (/h)
        params['a_r'] = 1.0318 * params['a_a']  # ribosomal gene transcription rate (/h)
        params['nu_max'] = 4.0469e3  # max tRNA amioacylation rate (/h)
        params['K_nu'] = 1.2397e3  # tRNA charging rate Michaelis-Menten constant (nM)
        params['K_e'] = 1.2397e3  # translation elongation rate Michaelis-Menten constant (nM)
        return params

    # set default initial conditions
    def default_init_conds(self, par):
        init_conds = {}  # initialise

        # mRNA concentrations - non-zero to avoid being stuck at lambda=0
        init_conds['m_a'] = 1000.0  # metabolic
        init_conds['m_r'] = 0.01  # ribosomal

        # protein concentrations - start with 50/50 a/R allocation as a convention
        init_conds['p_a'] = par['M'] * (1 - par['phi_q']) / (2 * par['n_a'])  # metabolic *
        init_conds['R'] = par['M'] * (1 - par['phi_q']) / (2 * par['n_r'])  # ribosomal *

        # tRNA concentrations - 3E-5 abundance units in Chure and Cremer 2022 are equivalent to 80 uM = 80000 nM
        init_conds['tc'] = 80000.0  # charged tRNAs
        init_conds['tu'] = 80000.0  # free tRNAs

        # nutrient quality s and chloramphenicol concentration h
        init_conds['s'] = 0.5
        init_conds['h'] = 0.0  # no translation inhibition assumed by default
        return init_conds

    # PREPARE FOR SIMULATIONS
    # set default initial condition vector
    def x0_from_init_conds(self, init_conds,
                           par,
                           synth_genes, synth_miscs,
                           modules_name2pos, controller_name2pos):
        # NATIVE GENES
        x0 = [
            # mRNAs;
            init_conds['m_a'],  # metabolic gene transcripts
            init_conds['m_r'],  # ribosomal gene transcripts

            # proteins
            init_conds['p_a'],  # metabolic proteins
            init_conds['R'],  # non-inactivated ribosomes

            # tRNAs
            init_conds['tc'],  # charged
            init_conds['tu'],  # uncharged

            # culture medium's nutrient quality and chloramphenicol concentration
            init_conds['s'],  # nutrient quality
            init_conds['h'],  # chloramphenicol levels IN THE CELL
        ]
        # GENETIC MODULES
        x0 += [0]*(2*len(synth_genes)+len(synth_miscs))  # initialise synthetic circuit species to zero
        for gene in synth_genes:  # mRNAs and proteins
            # set initial condition only if gene present
            if (par['cat_gene_present'] or gene != 'cat') or (par['prot_gene_present'] or gene != 'prot'):
                x0[modules_name2pos['m_' + gene]]=init_conds['m_' + gene]
                x0[modules_name2pos['p_' + gene]]=init_conds['p_' + gene]
        for misc in synth_miscs:    # miscellanous species
            x0[modules_name2pos[misc]]=init_conds[misc]

        # CONTROLLER
        x0 += [0]*len(controller_name2pos)
        for key in controller_name2pos.keys():
            x0[controller_name2pos[key]] = init_conds[key]

        return jnp.array(x0)

    # PLOT RESULTS, CALCULATE CELLULAR VARIABLES
    # plot protein composition of the cell by mass over time
    # def plot_protein_masses(self, ts, xs,
    #                         par, synth_genes,  # model parameters, list of circuit genes
    #                         dimensions=(320, 180), tspan=None):
    #     # set default time span if unspecified
    #     if (tspan == None):
    #         tspan = (ts[0], ts[-1])
    # 
    #     # create figure
    #     mass_figure = bkplot.figure(
    #         frame_width=dimensions[0],
    #         frame_height=dimensions[1],
    #         x_axis_label="t, hours",
    #         y_axis_label="Protein mass, aa",
    #         x_range=tspan,
    #         title='Protein masses',
    #         tools="box_zoom,pan,hover,reset"
    #     )
    # 
    #     flip_t = np.flip(ts)  # flipped time axis for patch plotting
    # 
    #     # plot heterologous protein mass - if there are any heterologous proteins to begin with
    #     if (len(synth_genes) != 0):
    #         bottom_line = np.zeros(xs.shape[0])
    #         top_line = bottom_line + np.sum(xs[:, 8 + len(synth_genes):8 + len(synth_genes) * 2] * np.array(
    #             self.synth_gene_params_for_jax(par, synth_genes)[2], ndmin=2), axis=1)
    #         mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
    #                           line_width=0.5, line_color='black', fill_color=self.gene_colours['het'],
    #                           legend_label='het')
    #     else:
    #         top_line = np.zeros(xs.shape[0])
    # 
    #     # plot mass of inactivated ribosomes
    #     if ((xs[:, 7] != 0).any()):
    #         bottom_line = top_line
    #         top_line = bottom_line + xs[:, 3] * par['n_r'] * (xs[:, 7] / (par['K_D'] + xs[:, 7]))
    #         mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
    #                           line_width=0.5, line_color='black', fill_color=self.gene_colours['h'], legend_label='R:h')
    # 
    #     # plot mass of active ribosomes - only if there are any to begin with
    #     bottom_line = top_line
    #     top_line = bottom_line + xs[:, 3] * par['n_r'] * (par['K_D'] / (par['K_D'] + xs[:, 7]))
    #     mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
    #                       line_width=0.5, line_color='black', fill_color=self.gene_colours['r'],
    #                       legend_label='R (free)')
    # 
    #     # plot metabolic protein mass
    #     bottom_line = top_line
    #     top_line = bottom_line + xs[:, 2] * par['n_a']
    #     mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
    #                       line_width=0.5, line_color='black', fill_color=self.gene_colours['a'], legend_label='p_a')
    # 
    #     # plot housekeeping protein mass
    #     bottom_line = top_line
    #     top_line = bottom_line / (1 - par['phi_q'])
    #     mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
    #                       line_width=0.5, line_color='black', fill_color=self.gene_colours['q'], legend_label='p_q')
    # 
    #     # add legend
    #     mass_figure.legend.label_text_font_size = "8pt"
    #     mass_figure.legend.location = "top_right"
    # 
    #     return mass_figure
    # 
    # # plot mRNA, protein and tRNA concentrations over time
    # def plot_native_concentrations(self, ts, xs,
    #                                par, synth_genes,  # model parameters, list of circuit genes
    #                                dimensions=(320, 180), tspan=None):
    #     # set default time span if unspecified
    #     if (tspan == None):
    #         tspan = (ts[0], ts[-1])
    # 
    #     # Create a ColumnDataSource object for the plot
    #     source = bkmodels.ColumnDataSource(data={
    #         't': ts,
    #         'm_a': xs[:, 0],  # metabolic mRNA
    #         'm_r': xs[:, 1],  # ribosomal mRNA
    #         'p_a': xs[:, 2],  # metabolic protein
    #         'R': xs[:, 3],  # ribosomal protein
    #         'tc': xs[:, 4],  # charged tRNA
    #         'tu': xs[:, 5],  # uncharged tRNA
    #         's': xs[:, 6],  # nutrient quality
    #         'h': xs[:, 7],  # chloramphenicol concentration
    #         'm_het': np.sum(xs[:, 8:8 + len(synth_genes)], axis=1),  # heterologous mRNA
    #         'p_het': np.sum(xs[:, 8 + len(synth_genes):8 + len(synth_genes) * 2], axis=1),  # heterologous protein
    #     })
    # 
    #     # PLOT mRNA CONCENTRATIONS
    #     mRNA_figure = bkplot.figure(
    #         frame_width=dimensions[0],
    #         frame_height=dimensions[1],
    #         x_axis_label="t, hours",
    #         y_axis_label="mRNA conc., nM",
    #         x_range=tspan,
    #         title='mRNA concentrations',
    #         tools="box_zoom,pan,hover,reset"
    #     )
    #     mRNA_figure.line(x='t', y='m_a', source=source, line_width=1.5, line_color=self.gene_colours['a'],
    #                      legend_label='m_a')  # plot metabolic mRNA concentrations
    #     mRNA_figure.line(x='t', y='m_r', source=source, line_width=1.5, line_color=self.gene_colours['r'],
    #                      legend_label='m_r')  # plot ribosomal mRNA concentrations
    #     mRNA_figure.line(x='t', y='m_het', source=source, line_width=1.5, line_color=self.gene_colours['het'],
    #                      legend_label='m_het')  # plot heterologous mRNA concentrations
    #     mRNA_figure.legend.label_text_font_size = "8pt"
    #     mRNA_figure.legend.location = "top_right"
    #     mRNA_figure.legend.click_policy = 'hide'
    # 
    #     # PLOT protein CONCENTRATIONS
    #     protein_figure = bkplot.figure(
    #         frame_width=dimensions[0],
    #         frame_height=dimensions[1],
    #         x_axis_label="t, hours",
    #         y_axis_label="Protein conc., nM",
    #         x_range=tspan,
    #         title='Protein concentrations',
    #         tools="box_zoom,pan,hover,reset"
    #     )
    #     protein_figure.line(x='t', y='p_a', source=source, line_width=1.5, line_color=self.gene_colours['a'],
    #                         legend_label='p_a')  # plot metabolic protein concentrations
    #     protein_figure.line(x='t', y='R', source=source, line_width=1.5, line_color=self.gene_colours['r'],
    #                         legend_label='R')  # plot ribosomal protein concentrations
    #     protein_figure.line(x='t', y='p_het', source=source, line_width=1.5, line_color=self.gene_colours['het'],
    #                         legend_label='p_het')  # plot heterologous protein concentrations
    #     protein_figure.legend.label_text_font_size = "8pt"
    #     protein_figure.legend.location = "top_right"
    #     protein_figure.legend.click_policy = 'hide'
    # 
    #     # PLOT tRNA CONCENTRATIONS
    #     tRNA_figure = bkplot.figure(
    #         frame_width=dimensions[0],
    #         frame_height=dimensions[1],
    #         x_axis_label="t, hours",
    #         y_axis_label="tRNA conc., nM",
    #         x_range=tspan,
    #         title='tRNA concentrations',
    #         tools="box_zoom,pan,hover,reset"
    #     )
    #     tRNA_figure.line(x='t', y='tc', source=source, line_width=1.5, line_color=self.tRNA_colours['tc'],
    #                      legend_label='tc')  # plot charged tRNA concentrations
    #     tRNA_figure.line(x='t', y='tu', source=source, line_width=1.5, line_color=self.tRNA_colours['tu'],
    #                      legend_label='tu')  # plot uncharged tRNA concentrations
    #     tRNA_figure.legend.label_text_font_size = "8pt"
    #     tRNA_figure.legend.location = "top_right"
    #     tRNA_figure.legend.click_policy = 'hide'        
    # 
    #     # PLOT INTRACELLULAR CHLORAMPHENICOL CONCENTRATION
    #     h_figure = bkplot.figure(
    #         frame_width=dimensions[0],
    #         frame_height=dimensions[1],
    #         x_axis_label="t, hours",
    #         y_axis_label="h, nM",
    #         x_range=tspan,
    #         title='Intracellular chloramphenicol concentration',
    #         tools="box_zoom,pan,hover,reset"
    #     )
    #     h_figure.line(x='t', y='h', source=source, line_width=1.5, line_color=self.gene_colours['h'],
    #                   legend_label='h')  # plot intracellular chloramphenicol concentration
    # 
    #     return mRNA_figure, protein_figure, tRNA_figure, h_figure
    # 
    # # plot concentrations for the synthetic circuits
    # def plot_circuit_concentrations(self, ts, xs,
    #                                 par, synth_genes, synth_miscs, modules_name2pos,
    #                                 # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
    #                                 circuit_styles,  # colours for the circuit plots
    #                                 dimensions=(320, 180), tspan=None):
    #     # if no circuitry at all, return no plots
    #     if (len(synth_genes) + len(synth_miscs) == 0):
    #         return None, None, None
    # 
    #     # set default time span if unspecified
    #     if (tspan == None):
    #         tspan = (ts[0], ts[-1])
    # 
    #     # Create a ColumnDataSource object for the plot
    #     data_for_column = {'t': ts}  # initialise with time axis
    #     # record synthetic mRNA and protein concentrations
    #     for i in range(0, len(synth_genes)):
    #         data_for_column['m_' + synth_genes[i]] = xs[:, 8 + i]
    #         data_for_column['p_' + synth_genes[i]] = xs[:, 8 + len(synth_genes) + i]
    #     # record miscellaneous species' concentrations
    #     for i in range(0, len(synth_miscs)):
    #         data_for_column[synth_miscs[i]] = xs[:, 8 + len(synth_genes) * 2 + i]
    #     source = bkmodels.ColumnDataSource(data=data_for_column)
    # 
    #     # PLOT mRNA and PROTEIN CONCENTRATIONS (IF ANY)
    #     if (len(synth_genes) > 0):
    #         # mRNAs
    #         mRNA_figure = bkplot.figure(
    #             frame_width=dimensions[0],
    #             frame_height=dimensions[1],
    #             x_axis_label="t, hours",
    #             y_axis_label="mRNA conc., nM",
    #             x_range=tspan,
    #             title='mRNA concentrations',
    #             tools="box_zoom,pan,hover,reset"
    #         )
    #         for gene in synth_genes:
    #             mRNA_figure.line(x='t', y='m_' + gene, source=source, line_width=1.5,
    #                              line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
    #                              legend_label='m_' + gene)
    #         mRNA_figure.legend.label_text_font_size = "8pt"
    #         mRNA_figure.legend.location = "top_right"
    #         mRNA_figure.legend.click_policy = 'hide'
    # 
    #         # proteins
    #         protein_figure = bkplot.figure(
    #             frame_width=dimensions[0],
    #             frame_height=dimensions[1],
    #             x_axis_label="t, hours",
    #             y_axis_label="Protein conc., nM",
    #             x_range=tspan,
    #             title='Protein concentrations',
    #             tools="box_zoom,pan,hover,reset"
    #         )
    #         for gene in synth_genes:
    #             protein_figure.line(x='t', y='p_' + gene, source=source, line_width=1.5,
    #                                 line_color=circuit_styles['colours'][gene],
    #                                 line_dash=circuit_styles['dashes'][gene],
    #                                 legend_label='p_' + gene)
    #         protein_figure.legend.label_text_font_size = "8pt"
    #         protein_figure.legend.location = "top_right"
    #         protein_figure.legend.click_policy = 'hide'
    #     else:
    #         mRNA_figure = None
    #         protein_figure = None
    # 
    #     # PLOT MISCELLANEOUS SPECIES' CONCENTRATIONS (IF ANY)
    #     if (len(synth_miscs) > 0):
    #         misc_figure = bkplot.figure(
    #             frame_width=dimensions[0],
    #             frame_height=dimensions[1],
    #             x_axis_label="t, hours",
    #             y_axis_label="Conc., nM",
    #             x_range=tspan,
    #             title='Miscellaneous species concentrations',
    #             tools="box_zoom,pan,hover,reset"
    #         )
    #         for misc in synth_miscs:
    #             misc_figure.line(x='t', y=misc, source=source, line_width=1.5,
    #                              line_color=circuit_styles['colours'][misc], line_dash=circuit_styles['dashes'][misc],
    #                              legend_label=misc)
    #         misc_figure.legend.label_text_font_size = "8pt"
    #         misc_figure.legend.location = "top_right"
    #         misc_figure.legend.click_policy = 'hide'
    #     else:
    #         misc_figure = None
    # 
    #     return mRNA_figure, protein_figure, misc_figure
    # 
    # # plot transcription regulation function values for the circuit's genes
    # def plot_circuit_regulation(self, ts, xs,
    #                             circuit_F_calc,
    #                             # function calculating the transcription regulation functions for the circuit
    #                             par, synth_genes, synth_miscs, modules_name2pos,
    #                             # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
    #                             circuit_styles,  # colours for the circuit plots
    #                             dimensions=(320, 180), tspan=None):
    #     # if no circuitry, return no plots
    #     if (len(synth_genes) == 0):
    #         return None
    # 
    #     # set default time span if unspecified
    #     if (tspan == None):
    #         tspan = (ts[0], ts[-1])
    # 
    #     # find values of gene transcription regulation functions
    #     Fs = np.zeros((len(ts), len(synth_genes)))  # initialise
    #     for i in range(0, len(ts)):
    #         Fs[i, :] = np.array(circuit_F_calc(ts[i], xs[i, :], par, modules_name2pos)[:])
    # 
    #     # Create ColumnDataSource object for the plot
    #     data_for_column = {'t': ts}  # initialise with time axis
    #     for i in range(0, len(synth_genes)):
    #         data_for_column['F_' + str(synth_genes[i])] = Fs[:, i]
    # 
    #     # PLOT
    #     F_figure = bkplot.figure(
    #         frame_width=dimensions[0],
    #         frame_height=dimensions[1],
    #         x_axis_label="t, hours",
    #         y_axis_label="Transc. reg. funcs. F",
    #         x_range=tspan,
    #         y_range=(0, 1.05),
    #         title='Gene transcription regulation',
    #         tools="box_zoom,pan,hover,reset"
    #     )
    #     for gene in synth_genes:
    #         F_figure.line(x='t', y='F_' + gene, source=data_for_column, line_width=1.5,
    #                       line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
    #                       legend_label='F_' + gene)
    #     F_figure.legend.label_text_font_size = "8pt"
    #     F_figure.legend.location = "top_right"
    #     F_figure.legend.click_policy = 'hide'
    # 
    #     return F_figure

    # plot physiological variables: growth rate, translation elongation rate, ribosomal gene transcription regulation function, ppGpp concentration, tRNA charging rate, RC denominator
    def plot_phys_variables(self, ts, xs,
                            par, synth_genes, synth_miscs, modules_name2pos,
                            # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                            dimensions=(320, 180), tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get cell variables' values over time
        e, l, F_r, nu, _, T, D, D_nodeg = self.get_e_l_Fr_nu_psi_T_D_Dnodeg(ts, xs, par, synth_genes, synth_miscs,
                                                                            modules_name2pos)

        # Create a ColumnDataSource object for the plot
        data_for_column = {'t': np.array(ts), 'e': np.array(e), 'l': np.array(l), 'F_r': np.array(F_r),
                           'ppGpp': np.array(1 / T), 'nu': np.array(nu), 'D': np.array(D), 'D_nodeg': np.array(D_nodeg)}
        source = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT GROWTH RATE
        l_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Growth rate, 1/h",
            x_range=tspan,
            y_range=(0, 2),
            title='Growth rate',
            tools="box_zoom,pan,hover,reset"
        )
        l_figure.line(x='t', y='l', source=source, line_width=1.5, line_color='blue', legend_label='l')
        l_figure.legend.label_text_font_size = "8pt"
        l_figure.legend.location = "top_right"
        l_figure.legend.click_policy = 'hide'

        # PLOT TRANSLATION ELONGATION RATE
        e_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Translation elongation rate, 1/h",
            x_range=tspan,
            y_range=(0, par['e_max']),
            title='Translation elongation rate',
            tools="box_zoom,pan,hover,reset"
        )
        e_figure.line(x='t', y='e', source=source, line_width=1.5, line_color='blue', legend_label='e')
        e_figure.legend.label_text_font_size = "8pt"
        e_figure.legend.location = "top_right"
        e_figure.legend.click_policy = 'hide'

        # PLOT RIBOSOMAL GENE TRANSCRIPTION REGULATION FUNCTION
        Fr_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Ribosomal gene transcription regulation function",
            x_range=tspan,
            y_range=(0, 1),
            title='Ribosomal gene transcription regulation function',
            tools="box_zoom,pan,hover,reset"
        )
        Fr_figure.line(x='t', y='F_r', source=source, line_width=1.5, line_color='blue', legend_label='F_r')
        Fr_figure.legend.label_text_font_size = "8pt"
        Fr_figure.legend.location = "top_right"
        Fr_figure.legend.click_policy = 'hide'

        # PLOT ppGpp CONCENTRATION
        ppGpp_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Rel. ppGpp conc. = 1/T",
            x_range=tspan,
            title='ppGpp concentration',
            tools="box_zoom,pan,hover,reset"
        )
        ppGpp_figure.line(x='t', y='ppGpp', source=source, line_width=1.5, line_color='blue', legend_label='ppGpp')
        ppGpp_figure.legend.label_text_font_size = "8pt"
        ppGpp_figure.legend.location = "top_right"
        ppGpp_figure.legend.click_policy = 'hide'

        # PLOT tRNA CHARGING RATE
        nu_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="tRNA charging rate, aa/h",
            x_range=tspan,
            title='tRNA charging rate',
            tools="box_zoom,pan,hover,reset"
        )
        nu_figure.line(x='t', y='nu', source=source, line_width=1.5, line_color='blue', legend_label='nu')
        nu_figure.legend.label_text_font_size = "8pt"
        nu_figure.legend.location = "top_right"
        nu_figure.legend.click_policy = 'hide'

        # PLOT RC DENOMINATOR
        D_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="RC denominator D",
            x_range=tspan,
            title='Resource Competition denominator',
            tools="box_zoom,pan,hover,reset"
        )
        D_figure.line(x='t', y='D', source=source, line_width=1.5, line_color='blue', legend_label='D')
        # D_figure.line(x='t', y='D_nodeg', source=source, line_width=1.5, line_color='red', legend_label='D (no deg.)')
        D_figure.legend.label_text_font_size = "8pt"
        D_figure.legend.location = "top_right"
        D_figure.legend.click_policy = 'hide'

        return l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure

    # find values of different cellular variables
    def get_e_l_Fr_nu_psi_T_D_Dnodeg(self, t, x,
                                     par, synth_genes, synth_miscs, modules_name2pos):
        # give the state vector entries meaningful names
        m_a = x[:, 0]  # metabolic gene mRNA
        m_r = x[:, 1]  # ribosomal gene mRNA
        p_a = x[:, 2]  # metabolic proteins
        R = x[:, 3]  # non-inactivated ribosomes
        tc = x[:, 4]  # charged tRNAs
        tu = x[:, 5]  # uncharged tRNAs
        s = x[:, 6]  # nutrient quality (constant)
        h = x[:, 7]  # chloramphenicol concentration (constant)
        x_het = x[:, 8:8 + 2 * len(synth_genes)]  # heterologous protein concentrations
        misc = x[:, 8 + 2 * len(synth_genes):8 + 2 * len(synth_genes) + len(synth_miscs)]  # miscellaneous species

        # vector of Synthetic Gene Parameters 4 JAX
        sgp4j = self.synth_gene_params_for_jax(par, synth_genes)
        kplus_het, kminus_het, n_het, d_het = sgp4j

        # FIND SPECIAL SYNTHETIC PROTEIN CONCENTRATIONS - IF PRESENT
        # chloramphenicol acetyltransferase (antibiotic reistance)
        p_cat = jax.lax.select(par['cat_gene_present'] == 1, x[:,modules_name2pos['p_cat']], jnp.zeros_like(x[:,0]))
        # synthetic protease (synthetic protein degradation)
        p_prot = jax.lax.select(par['prot_gene_present'] == 1, x[:,modules_name2pos['p_prot']], jnp.zeros_like(x[:,0]))

        # CALCULATE PHYSIOLOGICAL VARIABLES
        # translation elongation rate
        e = e_calc(par, tc)

        # ribosome dissociation constants
        k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'])
        k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'])
        k_het = k_calc((jnp.atleast_2d(jnp.array(e) * jnp.ones((len(synth_genes), 1)))).T,
                       jnp.atleast_2d(kplus_het) * jnp.ones((len(e), 1)),
                       jnp.atleast_2d(kminus_het) * jnp.ones((len(e), 1)),
                       jnp.atleast_2d(n_het) * jnp.ones((len(e), 1)))  # heterologous genes

        # ratio of charged to uncharged tRNAs
        T = tc / tu

        # corection to ribosome availability due to chloramphenicol action
        H = (par['K_D'] + h) / par['K_D']

        # heterologous mRNA levels scaled by RBS strength
        m_het_div_k_het = jnp.sum(x[:, 8:8 + len(synth_genes)] / k_het, axis=1)  # heterologous protein synthesis flux

        # heterologous protein degradation flux
        prodeflux = jnp.multiply(
            p_prot,
            jnp.sum(d_het * n_het * x[:, 8 + len(synth_genes):8 + len(synth_genes) * 2],axis=1)
        )
        # heterologous protein degradation flux
        prodeflux_times_H_div_eR = prodeflux * H / (e * R)

        # resource competition denominator
        m_notq_div_k_notq = m_a / k_a + m_r / k_r + m_het_div_k_het
        mq_div_kq = (par['phi_q'] * (1 - prodeflux_times_H_div_eR) * m_notq_div_k_notq - par[
            'phi_q'] * prodeflux_times_H_div_eR) / \
                    (1 - par['phi_q'] * (1 - prodeflux_times_H_div_eR))
        D = H * (1 + mq_div_kq + m_notq_div_k_notq)
        B = R * (1 / H - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

        nu = nu_calc(par, tu, s)  # tRNA charging rate

        l = l_calc(par, e, B, prodeflux)  # growth rate

        psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

        F_r = Fr_calc(par, T)  # ribosomal gene transcription regulation function

        # RC denominator, as it would be without active protein degradation by the protease
        D_nodeg = H * (1 + (1/(1-par['phi_q'])) * m_notq_div_k_notq)
        return e, l, F_r, nu, jnp.multiply(psi, l), T, D, D_nodeg

    # PLOT RESULTS FOR SEVERAL TRAJECTORIES AT ONCE (SAME TIME AXIS)
    # plot mRNA, protein and tRNA concentrations over time
    def plot_native_concentrations_multiple(self, ts, xss,
                                            par, synth_genes,  # model parameters, list of circuit genes
                                            dimensions=(320, 180), tspan=None,
                                            simtraj_alpha=0.1):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss)):
            # Create a ColumnDataSource object for the plot
            sources[i] = bkmodels.ColumnDataSource(data={
                't': ts,
                'm_a': xss[i, :, 0],  # metabolic mRNA
                'm_r': xss[i, :, 1],  # ribosomal mRNA
                'p_a': xss[i, :, 2],  # metabolic protein
                'R': xss[i, :, 3],  # ribosomal protein
                'tc': xss[i, :, 4],  # charged tRNA
                'tu': xss[i, :, 5],  # uncharged tRNA
                's': xss[i, :, 6],  # nutrient quality
                'h': xss[i, :, 7],  # chloramphenicol concentration
                'm_het': np.sum(xss[i, :, 8:8 + len(synth_genes)], axis=1),  # heterologous mRNA
                'p_het': np.sum(xss[i, :, 8 + len(synth_genes):8 + len(synth_genes) * 2], axis=1),  # heterologous protein
            })
            
        # Create a ColumnDataSource object for plotting the average trajectory
        source_avg = bkmodels.ColumnDataSource(data={
            't': ts,
            'm_a': np.mean(xss[:, :, 0], axis=0),  # metabolic mRNA
            'm_r': np.mean(xss[:, :, 1], axis=0),  # ribosomal mRNA
            'p_a': np.mean(xss[:, :, 2], axis=0),  # metabolic protein
            'R': np.mean(xss[:, :, 3], axis=0),  # ribosomal protein
            'tc': np.mean(xss[:, :, 4], axis=0),  # charged tRNA
            'tu': np.mean(xss[:, :, 5], axis=0),  # uncharged tRNA
            's': np.mean(xss[:, :, 6], axis=0),  # nutrient quality
            'h': np.mean(xss[:, :, 7], axis=0),  # chloramphenicol concentration
            'm_het': np.sum(np.mean(xss[:, :, 8:8 + len(synth_genes)], axis=0), axis=1),  # heterologous mRNA
            'p_het': np.sum(np.mean(xss[:, :, 8 + len(synth_genes):8 + len(synth_genes) * 2], axis=0), axis=1),  # heterologous protein
        })

        # PLOT mRNA CONCENTRATIONS
        mRNA_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="mRNA conc., nM",
            x_range=tspan,
            title='mRNA concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            mRNA_figure.line(x='t', y='m_a', source=sources[i], line_width=1.5, line_color=self.gene_colours['a'],
                             legend_label='m_a', line_alpha=simtraj_alpha)  # plot metabolic mRNA concentrations
            mRNA_figure.line(x='t', y='m_r', source=sources[i], line_width=1.5, line_color=self.gene_colours['r'],
                                legend_label='m_r', line_alpha=simtraj_alpha)  # plot ribosomal mRNA concentrations
            mRNA_figure.line(x='t', y='m_het', source=sources[i], line_width=1.5, line_color=self.gene_colours['het'],
                                legend_label='m_het', line_alpha=simtraj_alpha)  # plot heterologous mRNA concentrations
        # plot average trajectory
        mRNA_figure.line(x='t', y='m_a', source=source_avg, line_width=1.5, line_color=self.gene_colours['a'],
                            legend_label='m_a')  # plot metabolic mRNA concentrations
        mRNA_figure.line(x='t', y='m_r', source=source_avg, line_width=1.5, line_color=self.gene_colours['r'],
                            legend_label='m_r')  # plot ribosomal mRNA concentrations
        mRNA_figure.line(x='t', y='m_het', source=source_avg, line_width=1.5, line_color=self.gene_colours['het'],
                            legend_label='m_het')  # plot heterologous mRNA concentrations
        # add and format the legend
        mRNA_figure.legend.label_text_font_size = "8pt"
        mRNA_figure.legend.location = "top_right"
        mRNA_figure.legend.click_policy = 'hide'
        
        # PLOT protein CONCENTRATIONS
        protein_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Protein conc., nM",
            x_range=tspan,
            title='Protein concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            protein_figure.line(x='t', y='p_a', source=sources[i], line_width=1.5, line_color=self.gene_colours['a'],
                                legend_label='p_a', line_alpha=simtraj_alpha)
            protein_figure.line(x='t', y='R', source=sources[i], line_width=1.5, line_color=self.gene_colours['r'],
                                legend_label='R', line_alpha=simtraj_alpha)
            protein_figure.line(x='t', y='p_het', source=sources[i], line_width=1.5, line_color=self.gene_colours['het'],
                                legend_label='p_het', line_alpha=simtraj_alpha)
        # plot average trajectory
        protein_figure.line(x='t', y='p_a', source=source_avg, line_width=1.5, line_color=self.gene_colours['a'],
                            legend_label='p_a')
        protein_figure.line(x='t', y='R', source=source_avg, line_width=1.5, line_color=self.gene_colours['r'],
                            legend_label='R')
        protein_figure.line(x='t', y='p_het', source=source_avg, line_width=1.5, line_color=self.gene_colours['het'],
                            legend_label='p_het')
        # add and format the legend
        protein_figure.legend.label_text_font_size = "8pt"
        protein_figure.legend.location = "top_right"
        protein_figure.legend.click_policy = 'hide'
        
        # PLOT tRNA CONCENTRATIONS
        tRNA_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="tRNA conc., nM",
            x_range=tspan,
            title='tRNA concentrations',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            tRNA_figure.line(x='t', y='tc', source=sources[i], line_width=1.5, line_color=self.tRNA_colours['tc'],
                                legend_label='tc', line_alpha=simtraj_alpha)
            tRNA_figure.line(x='t', y='tu', source=sources[i], line_width=1.5, line_color=self.tRNA_colours['tu'],
                                legend_label='tu', line_alpha=simtraj_alpha)
        # plot average trajectory
        tRNA_figure.line(x='t', y='tc', source=source_avg, line_width=1.5, line_color=self.tRNA_colours['tc'],
                            legend_label='tc')
        tRNA_figure.line(x='t', y='tu', source=source_avg, line_width=1.5, line_color=self.tRNA_colours['tu'],
                            legend_label='tu')
        # add and format the legend
        tRNA_figure.legend.label_text_font_size = "8pt"
        tRNA_figure.legend.location = "top_right"
        tRNA_figure.legend.click_policy = 'hide'
        
        # PLOT INTRACELLULAR CHLORAMPHENICOL CONCENTRATION
        h_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="h, nM",
            x_range=tspan,
            title='Intracellular chloramphenicol concentration',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            h_figure.line(x='t', y='h', source=sources[i], line_width=1.5, line_color=self.gene_colours['h'],
                            legend_label='h', line_alpha=simtraj_alpha)
        # plot average trajectory
        h_figure.line(x='t', y='h', source=source_avg, line_width=1.5, line_color=self.gene_colours['h'],
                        legend_label='h')
        # add and format the legend
        h_figure.legend.label_text_font_size = "8pt"
        h_figure.legend.location = "top_right"
        h_figure.legend.click_policy = 'hide'
        
        return mRNA_figure, protein_figure, tRNA_figure, h_figure


    # plot concentrations for the synthetic circuits
    def plot_circuit_concentrations_multiple(self, ts, xss,
                                             par, synth_genes, synth_miscs, modules_name2pos,
                                             # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                             circuit_styles,  # colours for the circuit plots
                                             dimensions=(320, 180), tspan=None,
                                             simtraj_alpha=0.1):
        # if no circuitry at all, return no plots
        if (len(synth_genes) + len(synth_miscs) == 0):
            return None, None, None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss)):
            # Create a ColumnDataSource object for the plot
            data_for_column = {'t': ts}
            # record synthetic mRNA and protein concentrations
            for j in range(0, len(synth_genes)):
                data_for_column['m_' + synth_genes[j]] = xss[i, :, 8 + j]
                data_for_column['p_' + synth_genes[j]] = xss[i, :, 8 + len(synth_genes) + j]
            # record miscellaneous species' concentrations
            for j in range(0, len(synth_miscs)):
                data_for_column[synth_miscs[j]] = xss[i, :, 8 + len(synth_genes) * 2 + j]
            sources[i] = bkmodels.ColumnDataSource(data=data_for_column)

        # Create a ColumnDataSource object for plotting the average trajectory
        data_for_column = {'t': ts}
        # record synthetic mRNA and protein concentrations
        for j in range(0, len(synth_genes)):
            data_for_column['m_' + synth_genes[j]] = np.mean(xss[:, :, 8 + j], axis=0)
            data_for_column['p_' + synth_genes[j]] = np.mean(xss[:, :, 8 + len(synth_genes) + j], axis=0)
        # record miscellaneous species' concentrations
        for j in range(0, len(synth_miscs)):
            data_for_column[synth_miscs[j]] = np.mean(xss[:, :, 8 + len(synth_genes) * 2 + j], axis=0)
        source_avg = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT mRNA and PROTEIN CONCENTRATIONS (IF ANY)
        if (len(synth_genes) > 0):
            # mRNAs
            mRNA_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="mRNA conc., nM",
                x_range=tspan,
                title='mRNA concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            # plot simulated trajectories
            for i in range(0, len(xss)):
                for gene in synth_genes:
                    mRNA_figure.line(x='t', y='m_' + gene, source=sources[i], line_width=1.5,
                                     line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                                     legend_label='m_' + gene, line_alpha=simtraj_alpha)
            # plot average trajectory
            for gene in synth_genes:
                mRNA_figure.line(x='t', y='m_' + gene, source=source_avg, line_width=1.5,
                                 line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                                 legend_label='m_' + gene)
            # add and format the legend
            mRNA_figure.legend.label_text_font_size = "8pt"
            mRNA_figure.legend.location = 'top_left'
            mRNA_figure.legend.click_policy = 'hide'

            # proteins
            protein_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="Protein conc., nM",
                x_range=tspan,
                title='Protein concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            # plot simulated trajectories
            for i in range(0, len(xss)):
                for gene in synth_genes:
                    protein_figure.line(x='t', y='p_' + gene, source=sources[i], line_width=1.5,
                                        line_color=circuit_styles['colours'][gene],
                                        line_dash=circuit_styles['dashes'][gene],
                                        legend_label='p_' + gene, line_alpha=simtraj_alpha)
            # plot average trajectory
            for gene in synth_genes:
                protein_figure.line(x='t', y='p_' + gene, source=source_avg, line_width=1.5,
                                    line_color=circuit_styles['colours'][gene],
                                    line_dash=circuit_styles['dashes'][gene],
                                    legend_label='p_' + gene)
            # add and format the legend
            protein_figure.legend.label_text_font_size = "8pt"
            protein_figure.legend.location = 'top_left'
            protein_figure.legend.click_policy = 'hide'
        else:
            mRNA_figure = None
            protein_figure = None

        # PLOT MISCELLANEOUS SPECIES' CONCENTRATIONS (IF ANY)
        if (len(synth_miscs) > 0):
            misc_figure = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="Conc., nM",
                x_range=tspan,
                title='Miscellaneous species concentrations',
                tools="box_zoom,pan,hover,reset"
            )
            # plot simulated trajectories
            for i in range(0, len(xss)):
                for misc in synth_miscs:
                    misc_figure.line(x='t', y=misc, source=sources[i], line_width=1.5,
                                     line_color=circuit_styles['colours'][misc], line_dash=circuit_styles['dashes'][misc],
                                     legend_label=misc, line_alpha=simtraj_alpha)
            # plot average trajectory
            for misc in synth_miscs:
                misc_figure.line(x='t', y=misc, source=source_avg, line_width=1.5,
                                 line_color=circuit_styles['colours'][misc], line_dash=circuit_styles['dashes'][misc],
                                 legend_label=misc)
            # add and format the legend
            misc_figure.legend.label_text_font_size = "8pt"
            misc_figure.legend.location = 'top_left'
            misc_figure.legend.click_policy = 'hide'
        else:
            misc_figure = None

        return mRNA_figure, protein_figure, misc_figure


    # plot transcription regulation function values for the circuit's genes
    def plot_circuit_regulation_multiple(self, ts, xss,
                                         par, circuit_F_calc,
                                            # function calculating the transcription regulation functions for the circuit
                                            synth_genes, synth_miscs, modules_name2pos,
                                            # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                            circuit_styles,  # colours for the circuit plots
                                            dimensions=(320, 180), tspan=None,
                                            simtraj_alpha=0.1):
        # if no circuitry at all, return no plots
        if (len(synth_genes) + len(synth_miscs) == 0):
            return None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # find values of gene transcription regulation functions and create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0, len(xss)):
            Fs = np.zeros((len(ts), len(synth_genes)))  # initialise
            for k in range(0, len(ts)):
                Fs[k, :] = np.array(circuit_F_calc(ts[k], xss[i, k, :], par, modules_name2pos)[:])

            # Create a ColumnDataSource object for the plot
            data_for_column = {'t': ts}
            for j in range(0, len(synth_genes)):
                data_for_column['F_' + synth_genes[j]] = Fs[:, j]
            sources[i] = bkmodels.ColumnDataSource(data=data_for_column)

        # Create a ColumnDataSource object for plotting the average trajectory
        data_for_column = {'t': ts}
        for j in range(0, len(synth_genes)):
            data_for_column['F_' + synth_genes[j]] = np.zeros_like(ts)
        # add gene transcription regulation functions for different trajectories together
        for i in range(0, len(xss)):
            for j in range(0, len(synth_genes)):
                data_for_column['F_' + synth_genes[j]] += np.array(sources[i].data['F_' + synth_genes[j]])
        # divide by the number of trajectories to get the average
        for j in range(0, len(synth_genes)):
            data_for_column['F_' + synth_genes[j]] /= len(xss)
        source_avg = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT TRANSCRIPTION REGULATION FUNCTIONS
        F_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Transc. reg. funcs. F",
            x_range=tspan,
            y_range=(0, 1.05),
            title='Gene transcription regulation',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            for gene in synth_genes:
                F_figure.line(x='t', y='F_' + gene, source=sources[i], line_width=1.5,
                              line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                              legend_label='F_' + gene, line_alpha=simtraj_alpha)
        # plot average trajectory
        for gene in synth_genes:
            F_figure.line(x='t', y='F_' + gene, source=source_avg, line_width=1.5,
                          line_color=circuit_styles['colours'][gene], line_dash=circuit_styles['dashes'][gene],
                          legend_label='F_' + gene)
        # add and format the legend
        F_figure.legend.label_text_font_size = "8pt"
        F_figure.legend.location = 'top_left'
        F_figure.legend.click_policy = 'hide'

        return F_figure

    # plot physiological variables: growth rate, translation elongation rate, ribosomal gene transcription regulation function, ppGpp concentration, tRNA charging rate, RC denominator
    def plot_phys_variables_multiple(self, ts, xss,
                            par, synth_genes, synth_miscs, modules_name2pos,
                            # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                            dimensions=(320, 180), tspan=None,
                            simtraj_alpha=0.1):
        # if no circuitry at all, return no plots
        if (len(synth_genes) + len(synth_miscs) == 0):
            return None, None, None, None, None, None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get cell variables' values over time and create ColumnDataSource objects for the plot
        sources = {}
        for i in range(0,len(xss)):
            e, l, F_r, nu, psi, T, D, D_nodeg = self.get_e_l_Fr_nu_psi_T_D_Dnodeg(ts, xss[i, :, :],
                                                                                    par, synth_genes, synth_miscs, modules_name2pos)
            # Create a ColumnDataSource object for the plot
            sources[i] = bkmodels.ColumnDataSource(data={
                't': np.array(ts),
                'e': np.array(e),
                'l': np.array(l),
                'F_r': np.array(F_r),
                'nu': np.array(nu),
                'psi': np.array(psi),
                '1/T': np.array(1/T),
                'D': np.array(D),
                'D_nodeg': np.array(D_nodeg)
            })

        # Create a ColumnDataSource object for plotting the average trajectory
        data_for_column = {'t': ts,
                           'e': np.zeros_like(ts),
                           'l': np.zeros_like(ts),
                           'F_r': np.zeros_like(ts),
                           'nu': np.zeros_like(ts),
                           'psi': np.zeros_like(ts),
                           '1/T': np.zeros_like(ts),
                           'D': np.zeros_like(ts),
                           'D_nodeg': np.zeros_like(ts)}
        # add physiological variables for different trajectories together
        for i in range(0, len(xss)):
            data_for_column['e'] += np.array(sources[i].data['e'])
            data_for_column['l'] += np.array(sources[i].data['l'])
            data_for_column['F_r'] += np.array(sources[i].data['F_r'])
            data_for_column['nu'] += np.array(sources[i].data['nu'])
            data_for_column['psi'] += np.array(sources[i].data['psi'])
            data_for_column['1/T'] += np.array(sources[i].data['1/T'])
            data_for_column['D'] += np.array(sources[i].data['D'])
            data_for_column['D_nodeg'] += np.array(sources[i].data['D_nodeg'])
        # divide by the number of trajectories to get the average
        data_for_column['e'] /= len(xss)
        data_for_column['l'] /= len(xss)
        data_for_column['F_r'] /= len(xss)
        data_for_column['nu'] /= len(xss)
        data_for_column['psi'] /= len(xss)
        data_for_column['1/T'] /= len(xss)
        data_for_column['D'] /= len(xss)
        data_for_column['D_nodeg'] /= len(xss)
        source_avg = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT GROWTH RATE
        l_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Growth rate, 1/h",
            x_range=tspan,
            title='Growth rate',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            l_figure.line(x='t', y='l', source=sources[i], line_width=1.5, line_color='blue', legend_label='l', line_alpha=simtraj_alpha)
        # plot average trajectory
        l_figure.line(x='t', y='l', source=source_avg, line_width=1.5, line_color='blue', legend_label='l')
        # add and format the legend
        l_figure.legend.label_text_font_size = "8pt"
        l_figure.legend.location = 'top_left'
        l_figure.legend.click_policy = 'hide'

        # PLOT TRANSLATION ELONGATION RATE
        e_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Translation elongation rate, aa/s",
            x_range=tspan,
            title='Translation elongation rate',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            e_figure.line(x='t', y='e', source=sources[i], line_width=1.5, line_color='blue', legend_label='e', line_alpha=simtraj_alpha)
        # plot average trajectory
        e_figure.line(x='t', y='e', source=source_avg, line_width=1.5, line_color='blue', legend_label='e')
        # add and format the legend
        e_figure.legend.label_text_font_size = "8pt"
        e_figure.legend.location = 'top_left'
        e_figure.legend.click_policy = 'hide'

        # PLOT RIBOSOMAL GENE TRANSCRIPTION REGULATION FUNCTION
        F_r_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Ribosomal gene transc. reg. func. F_r",
            x_range=tspan,
            y_range=(0, 1.05),
            title='Ribosomal gene transcription regulation function',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            F_r_figure.line(x='t', y='F_r', source=sources[i], line_width=1.5, line_color='blue', legend_label='F_r', line_alpha=simtraj_alpha)
        # plot average trajectory
        F_r_figure.line(x='t', y='F_r', source=source_avg, line_width=1.5, line_color='blue', legend_label='F_r')
        # add and format the legend
        F_r_figure.legend.label_text_font_size = "8pt"
        F_r_figure.legend.location = 'top_left'
        F_r_figure.legend.click_policy = 'hide'

        # PLOT ppGpp CONCENTRATION
        ppGpp_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="[ppGpp], nM",
            x_range=tspan,
            title='ppGpp concentration',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            ppGpp_figure.line(x='t', y='1/T', source=sources[i], line_width=1.5, line_color='blue', legend_label='1/T', line_alpha=simtraj_alpha)
        # plot average trajectory
        ppGpp_figure.line(x='t', y='1/T', source=source_avg, line_width=1.5, line_color='blue', legend_label='1/T')
        # add and format the legend
        ppGpp_figure.legend.label_text_font_size = "8pt"
        ppGpp_figure.legend.location = 'top_left'
        ppGpp_figure.legend.click_policy = 'hide'

        # PLOT tRNA CHARGING RATE
        nu_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="tRNA charging rate, 1/s",
            x_range=tspan,
            title='tRNA charging rate',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            nu_figure.line(x='t', y='nu', source=sources[i], line_width=1.5, line_color='blue', legend_label='nu', line_alpha=simtraj_alpha)
        # plot average trajectory
        nu_figure.line(x='t', y='nu', source=source_avg, line_width=1.5, line_color='blue', legend_label='nu')

        # PLOT RC DENOMINATOR
        D_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="RC denominator",
            x_range=tspan,
            title='RC denominator',
            tools="box_zoom,pan,hover,reset"
        )
        # plot simulated trajectories
        for i in range(0, len(xss)):
            D_figure.line(x='t', y='D', source=sources[i], line_width=1.5, line_color='blue', legend_label='D', line_alpha=simtraj_alpha)
            #D_figure.line(x='t', y='D_nodeg', source=sources[i], line_width=1.5, line_color='red', legend_label='D_nodeg', line_alpha=simtraj_alpha)
        # plot average trajectory
        D_figure.line(x='t', y='D', source=source_avg, line_width=1.5, line_color='blue', legend_label='D')
        # D_figure.line(x='t', y='D_nodeg', source=source_avg, line_width=1.5, line_color='red', legend_label='D_nodeg')
        # add and format the legend
        D_figure.legend.label_text_font_size = "8pt"
        D_figure.legend.location = 'top_left'
        D_figure.legend.click_policy = 'hide'

        return l_figure, e_figure, F_r_figure, ppGpp_figure, nu_figure, D_figure


# DETERMINISTIC SIMULATION ---------------------------------------------------------------------------------------------
def ode_sim_loop(par,  # dictionary with model parameters
            ode_solver,  # ODE function for the cell with the synthetic gene circuit
            controller_update,  # function for updating the controller memory
            x0,  # initial condition VECTOR
            ctrl_memo0,  # initial controller memory
            num_synth_genes, num_synth_miscs, # number of synthetic genes and miscellaneous species in the circuit
            synth_genes, synth_miscs, # dictionary with circuit gene and miscellaneous specie name
            modules_name2pos, controller_name2pos, # variable name to position in the state vector decoders
            sgp4j, # some synthetic gene parameters in jax.array form - for efficient simulation
            tf,  # simulation time frame
            meastimestep   # output measurement time window
            ):
    # define the arguments for finding the next state vector
    args = (par,
            modules_name2pos, controller_name2pos,
            num_synth_genes, num_synth_miscs,
            sgp4j)

    # time points at which we save the solution
    ts = jnp.arange(tf[0], tf[1] + meastimestep / 2, meastimestep)

    # make the retrieval of next simulator state a lambda-function for jax.lax.scanning
    scan_step = lambda sim_state, t: ode_sim_step(sim_state, t,
                                                  meastimestep,
                                                  args,
                                                  ode_solver, controller_update)

    # define the jac.lax.scan function
    ode_scan = lambda sim_state_rec0, ts: jax.lax.scan(scan_step, sim_state_rec0, ts)
    ode_scan_jit = jax.jit(ode_scan)

    # initalise the simulator state: (t, x, sim_step_cntr, record_step_cntr, key, tf, xs)
    sim_state0 = {'t': tf[0], 'x': x0, 'ctrl_memo': ctrl_memo0,  # time, state vector, controller memory
                  'tf': tf  # overall simulation time frame
                  }

    # run the simulation
    _, sim_outcome = ode_scan_jit(sim_state0, ts)

    # extract the simulation outcomes
    t= sim_outcome[0]
    xs = sim_outcome[1]
    ctrl_memos = sim_outcome[2]

    # return the simulation outcomes
    return t, xs, ctrl_memos


def ode_sim_step(sim_state, t,
                 meastimestep,
                 args,
                 ode_solver, controller_update):
    # get next measurement time
    next_t = sim_state['t'] + meastimestep

    # simulate the ODE until the next measurement
    next_x = ode_solver(tf=(t, next_t),
                        x0=sim_state['x'],
                        ctrl_memo=sim_state['ctrl_memo'],
                        args=args)

    # update the controller memory
    next_ctrl_memo = controller_update(t, next_x, sim_state['ctrl_memo'], args)

    # update the overall simulation state
    next_sim_state = {'t': t+meastimestep,
                      'x': next_x,
                      'ctrl_memo': next_ctrl_memo,
                      'tf': sim_state['tf']}

    return next_sim_state, (t,sim_state['x'],sim_state['ctrl_memo'])


# ode
def ode(t, x, # simulation time and state vector
        ctrl_memo, # controller memory
        module1_ode, module2_ode, # ODEs for the genetic modules
        controller_ode, controller_action, # ODE and control action calculation for the controller
        args):
    # unpack the args
    par = args[0]  # model parameters
    modules_name2pos = args[1]  # gene name - position in circuit vector decoder
    controller_name2pos = args[2]  # controller name - position in circuit vector decoder
    num_synth_genes = args[3]; num_synth_miscs = args[4]  # number of genes and miscellaneous species in the circuit
    kplus_het, kminus_het, n_het, d_het = args[
        5]  # unpack jax-arrayed synthetic gene parameters for calculating k values

    # give the state vector entries meaningful names
    m_a = x[0]  # metabolic gene mRNA
    m_r = x[1]  # ribosomal gene mRNA
    p_a = x[2]  # metabolic proteins
    R = x[3]  # non-inactivated ribosomes
    tc = x[4]  # charged tRNAs
    tu = x[5]  # uncharged tRNAs
    s = x[6]  # nutrient quality (constant)
    h = x[7]  # INTERNAL chloramphenicol concentration (varies)
    # synthetic circuit genes and miscellaneous species can be accessed directly from x with modules_name2pos

    # FIND SPECIAL SYNTHETIC PROTEIN CONCENTRATIONS - IF PRESENT
    # chloramphenicol acetyltransferase (antibiotic reistance)
    p_cat = jax.lax.select(par['cat_gene_present'] == 1, x[modules_name2pos['p_cat']], 0.0)
    # synthetic protease (synthetic protein degradation)
    p_prot = jax.lax.select(par['prot_gene_present'] == 1, x[modules_name2pos['p_prot']], 0.0)

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'])
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'])
    k_het = k_calc(e, kplus_het, kminus_het, n_het)

    T = tc / tu  # ratio of charged to uncharged tRNAs

    H = (par['K_D'] + h) / par['K_D']  # corection to ribosome availability due to chloramphenicol action


    m_het_div_k_het = jnp.sum(x[8:8 + num_synth_genes] / k_het)

    # heterologous mRNA levels scaled by RBS strength
    m_het_div_k_het = jnp.sum(x[8:8 + num_synth_genes] / k_het)

    # heterologous protein degradation flux
    prodeflux = jnp.sum(
        # (degradation rate times protease level times protein concnetration) times number of AAs per protein
        d_het * p_prot * x[8 + num_synth_genes:8 + num_synth_genes * 2] * n_het
    )
    prodeflux_times_H_div_eR = prodeflux * H / (e * R)  # degradation flux scaled by overall protein synthesis rate

    # resource competition denominator
    m_notq_div_k_notq = m_a / k_a + m_r / k_r + m_het_div_k_het
    mq_div_kq = (par['phi_q'] * (1 - prodeflux_times_H_div_eR) * m_notq_div_k_notq - par[
        'phi_q'] * prodeflux_times_H_div_eR) / \
                (1 - par['phi_q'] * (1 - prodeflux_times_H_div_eR))
    D = H * (1 + mq_div_kq + m_notq_div_k_notq)
    B = R * (1 / H - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    nu = nu_calc(par, tu, s)  # tRNA charging rate

    l = l_calc(par, e, B, prodeflux)  # growth rate

    psi = psi_calc(par, T)  # tRNA synthesis rate - AMENDED

    # CONTROL ACTION CALCULATION
    u = controller_action(t,x,ctrl_memo,par,modules_name2pos,controller_name2pos)

    # return dx/dt for the host cell
    dxdt = jnp.array([
                         # mRNAs
                         l * par['c_a'] * par['a_a'] - (par['b_a'] + l) * m_a,
                         l * Fr_calc(par, T) * par['c_r'] * par['a_r'] - (par['b_r'] + l) * m_r,
                         # metabolic protein p_a
                         (e / par['n_a']) * (m_a / k_a / D) * R - l * p_a,
                         # ribosomes
                         (e / par['n_r']) * (m_r / k_r / D) * R - l * R,
                         # tRNAs
                         nu * p_a - l * tc - e * B,
                         l * psi - l * tu - nu * p_a + e * B,
                         # nutrient quality assumed constant
                         0,
                         # chloramphenicol concentration
                         par['diff_h'] * (par['h_ext'] - h) - h * p_cat / par['K_C'] - l * h
                     ] +
                     module1_ode(t, x, u, e, l, R, k_het, D, p_prot, par, modules_name2pos)
                     +
                     module2_ode(t, x, u, e, l, R, k_het, D, p_prot,
                                 par, modules_name2pos)
                     +
                     controller_ode(t, x, e, l, R, k_het, D, p_prot,
                                    par, modules_name2pos, controller_name2pos)
                     )
    return dxdt


# ODE SOLVER INITIALISERS ----------------------------------------------------------------------------------------------
def diffrax_solver(ode_with_circuit,
                   rtol, atol,  # relative and absolute integration tolerances
                   solver=Kvaerno3()  # ODE solver
                   ):

    # define the time points at which we save the solution
    stepsize_controller = PIDController(rtol=rtol, atol=atol)

    # define ODE solver
    ode_solver = lambda tf, x0, ctrl_memo, args: diffeqsolve(
                    ODETerm(lambda t, x, args: ode_with_circuit(t, x, ctrl_memo, args)),
                    solver,
                    args=args,
                    t0=tf[0], t1=tf[1], dt0=0.1, y0=x0,
                    max_steps=None,
                    stepsize_controller=stepsize_controller).ys[-1]
    return ode_solver

# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    # set up jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

    # initialise cell model
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    cellmodel_par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(cellmodel_par)  # get default initial conditions

    # load synthetic genetic modules and the controller
    ode, \
    module1_F_calc, module2_F_calc, controller_action, controller_update, \
    par, init_conds, \
    synth_genes, synth_miscs, \
    modules_name2pos, modules_styles, controller_name2pos, \
    module1_v_with_F_calc, module2_v_with_F_calc = cellmodel_auxil.add_modules_and_controller(
        # module 1
        gms.constfp_initialise,  # function initialising the circuit
        gms.constfp_ode,  # function defining the circuit ODEs
        gms.constfp_F_calc,
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
        cellmodel_par, init_conds)

    # DETERMINISTIC SIMULATION
    # set simulation parameters
    tf = (0, 30)  # simulation time frame

    # measurement time step
    meastimestep = 0.1  # hours

    # choose ODE solver
    ode_solver = diffrax_solver(ode,
                                rtol=1e-6, atol=1e-6,
                                solver=Kvaerno3())

    # solve ODE
    timer= time.time()
    t, xs, ctrl_memos = ode_sim_loop(par,
                                     ode_solver, controller_update,
                                     cellmodel_auxil.x0_from_init_conds(init_conds,
                                                                        par,
                                                                        synth_genes, synth_miscs,
                                                                        modules_name2pos, controller_name2pos),
                                     jnp.array([]),  # empty controller memory
                                     len(synth_genes), len(synth_miscs),
                                     # number of synthetic genes and miscellaneous species
                                     synth_genes, synth_miscs,  # lists of synthetic genes and miscellaneous species
                                     modules_name2pos, controller_name2pos, # dictionaries mapping gene names to their positions in the state vector
                                     cellmodel_auxil.synth_gene_params_for_jax(par,synth_genes),  # synthetic gene parameters in jax.array form
                                     tf, meastimestep,  # simulation time frame and measurement time step
                                     )
    print('Simulation time: ', time.time()-timer, ' s')

    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
