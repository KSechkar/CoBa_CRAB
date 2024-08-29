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
        controller_par, controller_init_conds, controller_init_memory, \
            controller_memos, controller_dynvars, controller_name2pos, controller_styles = controller_initialiser()

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
        modules_styles['colours'].update(module2_styles['colours'])
        modules_styles['dashes'].update(module2_styles['dashes'])
        
        # merge name-to-position dictionaries for the two modules: requires rearranging the variables
        # new order in x: module 1 mRNAs - module 2 mRNAs - module 1 proteins - module 2 proteins - module 1 misc - module 2 misc
        # new order in k (mRNA-ribosome dissociation constants): module 1 - module 2
        # order in F the same
        modules_name2pos = module1_name2pos.copy()
        modules_name2pos.update(module2_name2pos)
        for key in modules_name2pos.keys():
            # module 1 mRNAs kept as they are
            if((key[0:2]=='m_') and (key in module1_name2pos)):
                continue
            # module 2 mRNAs shifted by the number of module 1 mRNAs
            elif((key[0:2]=='m_') and (key in module2_name2pos)):
                if ((cellmodel_par['cat_gene_present'] or key[2:] != 'cat') or (cellmodel_par['prot_gene_present'] or key[2:] != 'prot')):
                    modules_name2pos[key] = modules_name2pos[key] + len(module1_genes)
            # module 1 proteins shifted by the number of module 2 mRNAs
            elif((key[0:2]=='p_') and (key in module1_name2pos)):
                if ((cellmodel_par['cat_gene_present'] or key[2:] != 'cat') and (cellmodel_par['prot_gene_present'] or key[2:] != 'prot')):
                    modules_name2pos[key] = modules_name2pos[key] + len(module2_genes)
            # module 2 proteins shifted by the number of module 1 mRNAs and proteins
            elif((key[0:2]=='p_') and (key in module2_name2pos)):
                if ((cellmodel_par['cat_gene_present'] or key[2:] != 'cat') and (cellmodel_par['prot_gene_present'] or key[2:] != 'prot')):
                    modules_name2pos[key] = modules_name2pos[key] + 2 * len(module1_genes)
            # module 1 k values kept as they are
            elif((key[0:2]=='k_') and (key in module1_name2pos)):
                continue
            # module 2 k values shifted by the number of module 1 mRNAs
            elif((key[0:2]=='k_') and (key in module2_name2pos)):
                modules_name2pos[key] = modules_name2pos[key] + len(module1_genes)
            # module 1 and 2 F values kept as they are
            elif((key[0:2]=='F_')):
                continue
            else: # miscellaneous species
                #  module 1 misc shifted by the number of module 2 mRNAs and proteins
                if(key in module1_name2pos):
                    modules_name2pos[key] = module1_name2pos[key] + 2 * len(module2_genes)
                #  module 1 misc shifted by the number of module 1 mRNAs, proteins and miscs
                else:
                    modules_name2pos[key] = module2_name2pos[key] + 2 * len(module1_genes) +len(module1_miscs)

        # update controller name-to-position dictionary now that genetic modules have been added
        for key in controller_name2pos.keys():
            if(key in controller_name2pos):
                controller_name2pos[key] = controller_name2pos[key] + 2 * len(module1_genes) + 2 * len(module2_genes) + len(module1_miscs) + len(module2_miscs)

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
        cellmodel_ode = lambda t, x, args: ode(t, x,
                                         # ctrl_memo,
                                         module1_ode_with_F_calc, module2_ode_with_F_calc,
                                         controller_ode, controller_action,
                                         args)

        # return updated ode and parameter, initial conditions, circuit gene (and miscellaneous specie) names
        # name - position in state vector decoder and colours for plotting the circuit's time evolution
        return (cellmodel_ode,
                module1_F_calc, module2_F_calc, controller_action, controller_update,
                cellmodel_par, cellmodel_init_conds, controller_init_memory,
                (synth_genes, module1_genes, module2_genes),
                (synth_miscs, module1_miscs, module2_miscs),
                controller_memos, controller_dynvars,
                modules_name2pos, modules_styles,
                controller_name2pos, controller_styles,
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
                           synth_genes, synth_miscs, controller_dynvars,
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
        x0 += [0]*len(controller_dynvars)
        for dynvar in controller_dynvars:
            x0[controller_name2pos[dynvar]] = init_conds[dynvar]

        return jnp.array(x0)

    # PLOT RESULTS, CALCULATE CELLULAR VARIABLES
    # plot protein composition of the cell by mass over time
    def plot_protein_masses(self, ts, xs,
                            par, # model parameters
                            synth_genes, synth_miscs, # list of circuit genes and miscellaneous species
                            modules_name2pos, # dictionary mapping gene names to their positions in the state vector
                            dimensions=(320, 180), tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # create figure
        mass_figure = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Protein mass, aa",
            x_range=tspan,
            title='Protein masses',
            tools="box_zoom,pan,hover,reset"
        )

        flip_t = np.flip(ts)  # flipped time axis for patch plotting

        # plot heterologous protein mass - if there are any heterologous proteins to begin with
        if (len(synth_genes) != 0):
            bottom_line = np.zeros(xs.shape[0])
            top_line = np.zeros(xs.shape[0])
            # include heterologous protein mass
            for gene in synth_genes:
                top_line += xs[:, modules_name2pos['p_' + gene]] * par['n_' + gene]

            # some miscellaneous species may be heterologous proteins having undergone some changes
            for misc in synth_miscs:
                if (misc[0:2] == 'p_'):
                    top_line += xs[:, modules_name2pos[misc]] * par['n_' + misc[2:]]

            mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                              line_width=0.5, line_color='black', fill_color=self.gene_colours['het'],
                              legend_label='het')
        else:
            top_line = np.zeros(xs.shape[0])

        # plot mass of inactivated ribosomes
        if ((xs[:, 7] != 0).any()):
            bottom_line = top_line
            top_line = bottom_line + xs[:, 3] * par['n_r'] * (xs[:, 7] / (par['K_D'] + xs[:, 7]))
            mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                              line_width=0.5, line_color='black', fill_color=self.gene_colours['h'], legend_label='R:h')

        # plot mass of active ribosomes - only if there are any to begin with
        bottom_line = top_line
        top_line = bottom_line + xs[:, 3] * par['n_r'] * (par['K_D'] / (par['K_D'] + xs[:, 7]))
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['r'],
                          legend_label='R (free)')

        # plot metabolic protein mass
        bottom_line = top_line
        top_line = bottom_line + xs[:, 2] * par['n_a']
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['a'], legend_label='p_a')

        # plot housekeeping protein mass
        bottom_line = top_line
        top_line = bottom_line / (1 - par['phi_q'])
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['q'], legend_label='p_q')

        # add legend
        mass_figure.legend.label_text_font_size = "8pt"
        mass_figure.legend.location = "top_right"

        return mass_figure

    # plot mRNA, protein and tRNA concentrations over time
    def plot_native_concentrations(self, ts, xs,
                                   par, # model parameters
                                   synth_genes, synth_miscs,  # list of circuit genes and miscellaneous species
                                      modules_name2pos,  # dictionary mapping gene names to their positions in the state vector
                                   dimensions=(320, 180), tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get total concentrations of synthetic mRNAs and proteins
        m_het = np.zeros(xs.shape[0])
        p_het = np.zeros(xs.shape[0])
        for gene in synth_genes:
            m_het += xs[:, modules_name2pos['m_' + gene]]
            p_het += xs[:, modules_name2pos['p_' + gene]]
        for misc in synth_miscs: # some miscellaneous species may be heterologous mRNAs or proteins having undergone some changes
            if (misc[0:2] == 'm_'):
                m_het += xs[:, modules_name2pos[misc]]
            elif (misc[0:2] == 'p_'):
                p_het += xs[:, modules_name2pos[misc]]

        # Create a ColumnDataSource object for the plot
        source = bkmodels.ColumnDataSource(data={
            't': ts,
            'm_a': xs[:, 0],  # metabolic mRNA
            'm_r': xs[:, 1],  # ribosomal mRNA
            'p_a': xs[:, 2],  # metabolic protein
            'R': xs[:, 3],  # ribosomal protein
            'tc': xs[:, 4],  # charged tRNA
            'tu': xs[:, 5],  # uncharged tRNA
            's': xs[:, 6],  # nutrient quality
            'h': xs[:, 7],  # chloramphenicol concentration
            'm_het': m_het,  # heterologous mRNA
            'p_het': p_het,  # heterologous protein
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
        mRNA_figure.line(x='t', y='m_a', source=source, line_width=1.5, line_color=self.gene_colours['a'],
                         legend_label='m_a')  # plot metabolic mRNA concentrations
        mRNA_figure.line(x='t', y='m_r', source=source, line_width=1.5, line_color=self.gene_colours['r'],
                         legend_label='m_r')  # plot ribosomal mRNA concentrations
        mRNA_figure.line(x='t', y='m_het', source=source, line_width=1.5, line_color=self.gene_colours['het'],
                         legend_label='m_het')  # plot heterologous mRNA concentrations
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
        protein_figure.line(x='t', y='p_a', source=source, line_width=1.5, line_color=self.gene_colours['a'],
                            legend_label='p_a')  # plot metabolic protein concentrations
        protein_figure.line(x='t', y='R', source=source, line_width=1.5, line_color=self.gene_colours['r'],
                            legend_label='R')  # plot ribosomal protein concentrations
        protein_figure.line(x='t', y='p_het', source=source, line_width=1.5, line_color=self.gene_colours['het'],
                            legend_label='p_het')  # plot heterologous protein concentrations
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
        tRNA_figure.line(x='t', y='tc', source=source, line_width=1.5, line_color=self.tRNA_colours['tc'],
                         legend_label='tc')  # plot charged tRNA concentrations
        tRNA_figure.line(x='t', y='tu', source=source, line_width=1.5, line_color=self.tRNA_colours['tu'],
                         legend_label='tu')  # plot uncharged tRNA concentrations
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
        h_figure.line(x='t', y='h', source=source, line_width=1.5, line_color=self.gene_colours['h'],
                      legend_label='h')  # plot intracellular chloramphenicol concentration

        return mRNA_figure, protein_figure, tRNA_figure, h_figure

    # plot concentrations for the synthetic circuits
    def plot_circuit_concentrations(self, ts, xs,
                                    par, synth_genes, synth_miscs, modules_name2pos,
                                    # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                                    modules_styles,  # colours for the circuit plots
                                    dimensions=(320, 180), tspan=None):
        # if no circuitry at all, return no plots
        if (len(synth_genes) + len(synth_miscs) == 0):
            return None, None, None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # Create a ColumnDataSource object for the plot
        data_for_column = {'t': ts}  # initialise with time axis
        # record synthetic mRNA and protein concentrations
        for i in range(0, len(synth_genes)):
            data_for_column['m_' + synth_genes[i]] = xs[:, 8 + i]
            data_for_column['p_' + synth_genes[i]] = xs[:, 8 + len(synth_genes) + i]
        # record miscellaneous species' concentrations
        for i in range(0, len(synth_miscs)):
            data_for_column[synth_miscs[i]] = xs[:, 8 + len(synth_genes) * 2 + i]
        source = bkmodels.ColumnDataSource(data=data_for_column)

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
            for gene in synth_genes:
                mRNA_figure.line(x='t', y='m_' + gene, source=source, line_width=1.5,
                                 line_color=modules_styles['colours'][gene], line_dash=modules_styles['dashes'][gene],
                                 legend_label='m_' + gene)
            mRNA_figure.legend.label_text_font_size = "8pt"
            mRNA_figure.legend.location = "top_right"
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
            for gene in synth_genes:
                protein_figure.line(x='t', y='p_' + gene, source=source, line_width=1.5,
                                    line_color=modules_styles['colours'][gene],
                                    line_dash=modules_styles['dashes'][gene],
                                    legend_label='p_' + gene)
            protein_figure.legend.label_text_font_size = "8pt"
            protein_figure.legend.location = "top_right"
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
            for misc in synth_miscs:
                misc_figure.line(x='t', y=misc, source=source, line_width=1.5,
                                 line_color=modules_styles['colours'][misc], line_dash=modules_styles['dashes'][misc],
                                 legend_label=misc)
            misc_figure.legend.label_text_font_size = "8pt"
            misc_figure.legend.location = "top_right"
            misc_figure.legend.click_policy = 'hide'
        else:
            misc_figure = None

        return mRNA_figure, protein_figure, misc_figure

    # plot transcription regulation function values for the circuit's genes
    def plot_circuit_regulation(self, ts, xs, ctrl_memorecord, # time points and state vectors
                                module1_F_calc, module2_F_calc, # transcription regulation functions for both modules
                                controller_action, # control action calculator
                                par, # model parameters
                                synth_genes_total_and_each,     # list of synthetic genes - total and for each module
                                synth_miscs_total_and_each,     # list of synthetic miscellaneous species - total and for each module
                                modules_name2pos,   # dictionary mapping gene names to their positions in the state vector
                                controller_name2pos, # dictionary mapping controller species to their positions in the state vector
                                modules_styles,  # colours for the circuit plots
                                dimensions=(320, 180), tspan=None):
        # unpack synthetic gene and miscellaneous specie lists
        synth_genes = synth_genes_total_and_each[0]
        module1_genes = synth_genes_total_and_each[1]
        module2_genes = synth_genes_total_and_each[2]
        synth_miscs = synth_miscs_total_and_each[0]
        module1_miscs = synth_miscs_total_and_each[1]
        module2_miscs = synth_miscs_total_and_each[2]

        # get control action values
        us=self.get_u(ts,xs,ctrl_memorecord,
              controller_action,
              par, modules_name2pos, controller_name2pos)

        # if no circuitry, return no plots
        if (len(synth_genes) == 0):
            return None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # find values of gene transcription regulation functions
        Fs1 = np.zeros((len(ts), len(module1_genes)))  # initialise
        Fs2 = np.zeros((len(ts), len(module2_genes)))  # initialise
        for i in range(0, len(ts)):
            Fs1[i, :] = np.array(module1_F_calc(ts[i], xs[i, :], us[i], par, modules_name2pos)[:])
            Fs2[i, :] = np.array(module2_F_calc(ts[i], xs[i, :], us[i], par, modules_name2pos)[:])

        # Create ColumnDataSource object for the plot
        data_for_column = {'t': ts}  # initialise with time axis
        for i in range(0, len(module1_genes)):
            data_for_column['F_' + module1_genes[i]] = Fs1[:, i]
        for i in range(0, len(module2_genes)):
            data_for_column['F_' + module2_genes[i]] = Fs2[:, i]

        # PLOT
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
        for gene in synth_genes:
            F_figure.line(x='t', y='F_' + gene, source=data_for_column, line_width=1.5,
                          line_color=modules_styles['colours'][gene], line_dash=modules_styles['dashes'][gene],
                          legend_label='F_' + gene)
        F_figure.legend.label_text_font_size = "8pt"
        F_figure.legend.location = "top_right"
        F_figure.legend.click_policy = 'hide'

        return F_figure

    # plot controller states and actions over time
    def plot_controller(self, ts, xs, ctrl_memorecord, # time points and state vectors
                        memos, dynvars, # controller memo and dynamic variable names
                        controller_action, controller_update, # control action calculator and controller state update
                        par, # model parameters
                        modules_name2pos,   # dictionary mapping gene names to their positions in the state vector
                        controller_name2pos, # dictionary mapping controller species to their positions in the state vector
                        controller_styles, # colours for the controller plots
                        dimensions=(320, 180), tspan=None):

        # get control action values
        us=self.get_u(ts,xs,ctrl_memorecord,
                controller_action,
                par, modules_name2pos, controller_name2pos)

        # set default time span if unspecified
        if(tspan==None):
            tspan=(ts[0], ts[-1])

        # memory entry figure
        if (len(memos) == 0):
            memo_fig = None
        else:
            # create a ColumnDataSource object for the plot
            data_for_column = {'t': ts}
            for i in range(0, len(memos)):
                data_for_column[memos[i]] = ctrl_memorecord[:, i]
            source = bkmodels.ColumnDataSource(data=data_for_column)

            # PLOT
            memo_fig = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="Memory values",
                x_range=tspan,
                title='Controller memory',
                tools="box_zoom,pan,hover,reset"
            )
            for i in range(0, len(memos)):
                memo_fig.line(x='t', y=memos[i], source=source, line_width=1.5,
                              line_color=controller_styles['colours'][memos[i]],
                              line_dash=controller_styles['dashes'][memos[i]],
                              legend_label=memos[i])
            # legend formatting
            memo_fig.legend.label_text_font_size = "8pt"
            memo_fig.legend.location = "top_right"
            memo_fig.legend.click_policy = 'hide'

        # dynamic variables figure
        if(len(dynvars)==0):
            dynvar_fig=None
        else:
            # Create a ColumnDataSource object for the plot
            data_for_column = {'t': ts}
            for i in range(0, len(dynvars)):
                data_for_column[dynvars[i]] = xs[:, controller_name2pos[dynvars[i]]]
            source = bkmodels.ColumnDataSource(data=data_for_column)

            # PLOT
            dynvar_fig = bkplot.figure(
                frame_width=dimensions[0],
                frame_height=dimensions[1],
                x_axis_label="t, hours",
                y_axis_label="Dynamic variable values",
                x_range=tspan,
                title='Controller dynamic variables',
                tools="box_zoom,pan,hover,reset"
            )
            for i in range(0, len(dynvars)):
                dynvar_fig.line(x='t', y=dynvars[i], source=source, line_width=1.5,
                              line_color=controller_styles['colours'][dynvars[i]],
                              line_dash=controller_styles['dashes'][dynvars[i]],
                              legend_label=dynvars[i])
            # legend formatting
            dynvar_fig.legend.label_text_font_size = "8pt"
            dynvar_fig.legend.location = "top_right"
            dynvar_fig.legend.click_policy = 'hide'

        # controller action figure
        u_fig = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Control action",
            x_range=tspan,
            title='Controller action',
            tools="box_zoom,pan,hover,reset"
        )
        u_fig.line(x=ts, y=us, line_width=1.5, line_color='blue', legend_label='u') # plot control action
        # legend formatting
        u_fig.legend.label_text_font_size = "8pt"
        u_fig.legend.location = "top_right"
        u_fig.legend.click_policy = 'hide'

        return memo_fig, dynvar_fig, u_fig


    # plot physiological variables: growth rate, translation elongation rate, ribosomal gene transcription regulation function, ppGpp concentration, tRNA charging rate, RC denominator
    def plot_phys_variables(self, ts, xs,
                            par, synth_genes, synth_miscs, modules_name2pos,
                            # model parameters, list of circuit genes and miscellaneous species, and dictionary mapping gene names to their positions in the state vector
                            dimensions=(320, 180), tspan=None):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get cell variables' values over time
        e, l, F_r, nu, _, T, D = self.get_e_l_Fr_nu_psi_T_D(ts, xs, par, synth_genes, synth_miscs,
                                                                            modules_name2pos)

        # Create a ColumnDataSource object for the plot
        data_for_column = {'t': np.array(ts), 'e': np.array(e), 'l': np.array(l), 'F_r': np.array(F_r),
                           'ppGpp': np.array(1 / T), 'nu': np.array(nu), 'D': np.array(D)}
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
        D_figure.legend.label_text_font_size = "8pt"
        D_figure.legend.location = "top_right"
        D_figure.legend.click_policy = 'hide'

        return l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure

    # find values of different cellular variables
    def get_e_l_Fr_nu_psi_T_D(self, t, x,
                                     par,
                                     synth_genes,
                                     synth_miscs,
                                     modules_name2pos):
        # give the state vector entries meaningful names
        m_a = x[:, 0]  # metabolic gene mRNA
        m_r = x[:, 1]  # ribosomal gene mRNA
        p_a = x[:, 2]  # metabolic proteins
        R = x[:, 3]  # non-inactivated ribosomes
        tc = x[:, 4]  # charged tRNAs
        tu = x[:, 5]  # uncharged tRNAs
        s = x[:, 6]  # nutrient quality (constant)
        h = x[:, 7]  # chloramphenicol concentration (constant)
        x_het = x[:, 8:8 + 2 * len(synth_genes)]  # heterologous mRNA and protein concentrations
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

        return e, l, F_r, nu, jnp.multiply(psi, l), T, D

    # find control inputs
    def get_u(self, t, x,
              ctrl_memo, 
              controller_action,
              par, 
              modules_name2pos, controller_name2pos):
        u = np.zeros(len(t))
        for i in range(0, len(t)):
            u[i] = controller_action(t[i], x[i, :], ctrl_memo[i], par, modules_name2pos, controller_name2pos)
        return u


# DETERMINISTIC SIMULATION ---------------------------------------------------------------------------------------------
# ode
def ode(t, x, # simulation time and state vector
        #ctrl_memo, # controller memory
        module1_ode, module2_ode, # ODEs for the genetic modules
        controller_ode, controller_action, # ODE and control action calculation for the controller
        args):
    # unpack the args
    par = args[0]  # model parameters
    modules_name2pos = args[1]  # gene name - position in circuit vector decoder
    controller_name2pos = args[2]  # controller name - position in circuit vector decoder
    num_synth_genes, num_synth_genes1, num_synth_genes2 = args[3] # number of synthetic genes: total and for each module
    num_synth_miscs, num_synth_miscs1, num_synth_miscs2 = args[4]  # number of miscellaneous species: total and for each module
    kplus_het, kminus_het, n_het, d_het = args[
        5]  # unpack jax-arrayed synthetic gene parameters for calculating k values

    # get the controller memory
    ctrl_memo = args[6]

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

    # GENETIC MODULE ODE CALCULATION
    module1_ode_value = module1_ode(t, x, u, e, l, R, k_het, D, p_prot, par, modules_name2pos)
    module2_ode_value = module2_ode(t, x, u, e, l, R, k_het, D, p_prot, par, modules_name2pos)

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
                     ]
                    +
                    # add synthetic mRNA ODEs
                    module1_ode_value[0:num_synth_genes1] + module2_ode_value[0:num_synth_genes2]
                    +
                    # add synthetic protein ODEs
                    module1_ode_value[num_synth_genes1:num_synth_genes1*2] + module2_ode_value[num_synth_genes2:num_synth_genes2*2]
                    +
                    # add miscellaneous species ODEs
                    module1_ode_value[num_synth_genes1*2:] + module2_ode_value[num_synth_genes2*2:]
                    +
                    # add controller ODEs
                    controller_ode(t, x, e, l, R, k_het, D, p_prot, par, modules_name2pos, controller_name2pos)
                     )
    return dxdt

# ode simulation loop
def ode_sim_loop(par,  # dictionary with model parameters
            ode_solver,  # ODE function for the cell with the synthetic gene circuit
            controller_update,  # function for updating the controller memory
            controller_action,  # function for calculating the control action
            x0,  # initial condition VECTOR
            ctrl_memo0,  # initial controller memory
            num_synth_genes, num_synth_miscs, # number of synthetic genes and miscellaneous species in the circuit
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
                                                  ode_solver,
                                                  controller_update)

    # define the jac.lax.scan function
    ode_scan = lambda sim_state_rec0, ts: jax.lax.scan(scan_step, sim_state_rec0, ts)
    ode_scan_jit = jax.jit(ode_scan)

    # initalise the simulator state: (t, x, sim_step_cntr, record_step_cntr, key, tf, xs)
    sim_state0 = {'t': tf[0], 'x': x0,  # time, state vector
                  'ctrl_memo': jnp.array(ctrl_memo0),  # controller memory (jnp array format)
                  'tf': tf  # overall simulation time frame
                  }

    # run the simulation
    _, sim_outcome = ode_scan_jit(sim_state0, ts)

    # extract the simulation outcomes
    t= sim_outcome[0]
    xs = sim_outcome[1]
    ctrl_memorecord = sim_outcome[2]

    # return the simulation outcomes
    return t, xs, ctrl_memorecord

# one step of the ode simulation loop (from one measurement to the next)
def ode_sim_step(sim_state, t,
                 meastimestep,
                 args,
                 ode_solver,
                 controller_update):

    # get next measurement time
    next_t = sim_state['t'] + meastimestep

    # simulate the ODE until the next measurement
    next_x = ode_solver(tf=(t, next_t),
                        x0=sim_state['x'],
                        args=args+(sim_state['ctrl_memo'],))

    # update the controller memory
    # par= args[0]
    # modules_name2pos = args[1]
    # controller_name2pos = args[2]
    next_ctrl_memo = controller_update(t, next_x, sim_state['ctrl_memo'], args[0], args[1], args[2])

    # update the overall simulation state
    next_sim_state = {'t': t+meastimestep,
                      'x': next_x,
                      'ctrl_memo': next_ctrl_memo,
                      'tf': sim_state['tf']}

    return next_sim_state, (t,sim_state['x'],sim_state['ctrl_memo'])


# ODE SOLVER INITIALISERS ----------------------------------------------------------------------------------------------
def diffrax_solver(ode_complete,
                   rtol, atol,  # relative and absolute integration tolerances
                   solver=Kvaerno3()  # ODE solver
                   ):
    # define ODE term
    vector_field = lambda t, y, args: ode_complete(t, y, args)
    term = ODETerm(vector_field)

    # define the time points at which we save the solution
    stepsize_controller = PIDController(rtol=rtol, atol=atol)

    # define ODE solver
    ode_solver = lambda tf, x0, args: diffeqsolve(
                    term,
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
    par, init_conds, controller_memo0, \
    synth_genes_total_and_each, synth_miscs_total_and_each, \
    controller_memos, controller_dynvars, \
    modules_name2pos, modules_styles, controller_name2pos, controller_styles, \
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

    # unpack the synthetic genes and miscellaneous species lists
    synth_genes= synth_genes_total_and_each[0]
    module1_genes = synth_genes_total_and_each[1]
    module2_genes = synth_genes_total_and_each[2]
    synth_miscs= synth_miscs_total_and_each[0]
    module1_miscs = synth_miscs_total_and_each[1]
    module2_miscs = synth_miscs_total_and_each[2]

    # SET PARAMETERS
    # set the parameters for the synthetic genes
    par['c_ofp']=100
    par['a_ofp']=1000
    par['c_b']=100
    par['a_b']=1000

    # DETERMINISTIC SIMULATION
    # set simulation parameters
    tf = (0, 20)  # simulation time frame

    # measurement time step
    meastimestep = 0.1  # hours

    # choose ODE solver
    ode_solver = diffrax_solver(ode,
                                rtol=1e-6, atol=1e-6,
                                solver=Kvaerno3())

    # solve ODE
    timer= time.time()
    ts_jnp, xs_jnp, ctrl_memorecord_jnp = ode_sim_loop(par,
                                                  ode_solver,
                                                  controller_update, controller_action,
                                                  cellmodel_auxil.x0_from_init_conds(init_conds,
                                                                                     par,
                                                                                     synth_genes, synth_miscs, controller_dynvars,
                                                                                     modules_name2pos,
                                                                                     controller_name2pos),
                                                  controller_memo0,  # initial controller memory record
                                                  (len(synth_genes), len(module1_genes), len(module2_genes)), # number of synthetic genes
                                                  (len(synth_miscs), len(module1_miscs), len(module2_miscs)), # number of miscellaneous species
                                                  modules_name2pos, controller_name2pos, # dictionaries mapping gene names to their positions in the state vector
                                                  cellmodel_auxil.synth_gene_params_for_jax(par, synth_genes), # synthetic gene parameters in jax.array form
                                                  tf, meastimestep,  # simulation time frame and measurement time step
                                                  )

    # convert simulation results to numpy arrays
    ts = np.array(ts_jnp)
    xs = np.array(xs_jnp)
    ctrl_memorecord = np.array(ctrl_memorecord_jnp)

    print('Simulation time: ', time.time()-timer, ' s')
    
    # make plots
    # PLOT - HOST CELL MODEL
    bkplot.output_file(filename="cellmodel_sim.html",
                       title="Cell Model Simulation")  # set up bokeh output file
    mass_fig = cellmodel_auxil.plot_protein_masses(ts, xs, par, synth_genes, synth_miscs, modules_name2pos)  # plot simulation results
    nat_mrna_fig, nat_prot_fig, nat_trna_fig, h_fig = cellmodel_auxil.plot_native_concentrations(ts, xs,
                                                                                                 par,
                                                                                                 synth_genes,
                                                                                                 synth_miscs,
                                                                                                 modules_name2pos)  # plot simulation results
    l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure = cellmodel_auxil.plot_phys_variables(ts,
                                                                                                           xs,
                                                                                                           par,
                                                                                                           synth_genes,
                                                                                                           synth_miscs,
                                                                                                           modules_name2pos)  # plot simulation results
    bkplot.save(bklayouts.grid([[mass_fig, nat_mrna_fig, nat_prot_fig],
                                [nat_trna_fig, h_fig, l_figure],
                                [e_figure, Fr_figure, ppGpp_figure]]))

    # PLOT - SYNTHETIC CIRCUITS AND CONTROLLER
    bkplot.output_file(filename="circuit_sim.html",
                       title="Synthetic Circuit Simulation")  # set up bokeh output file
    # plot synthetic circuit concentrations
    mRNA_fig, prot_fig, misc_fig = cellmodel_auxil.plot_circuit_concentrations(ts, xs,
                                                                              par, synth_genes, synth_miscs,
                                                                              modules_name2pos,
                                                                              modules_styles)  # plot simulation results
    # plot synthetic circuit regulation functions
    F_fig = cellmodel_auxil.plot_circuit_regulation(ts, xs, ctrl_memorecord, # time points and state vectors
                                                    module1_F_calc, module2_F_calc, # transcription regulation functions for both modules
                                                    controller_action, # control action calculator
                                                    par, # model parameters
                                                    synth_genes_total_and_each,     # list of synthetic genes - total and for each module
                                                    synth_miscs_total_and_each,     # list of synthetic miscellaneous species - total and for each module
                                                    modules_name2pos,   # dictionary mapping gene names to their positions in the state vector
                                                    controller_name2pos, # dictionary mapping controller species to their positions in the state vector
                                                    modules_styles)  # plot simulation results
    # plot controller memory, dynamic variables and actions
    ctrl_memo_fig, ctrl_dynvar_fig, ctrl_u_fig = cellmodel_auxil.plot_controller(ts, xs, ctrl_memorecord,
                                                                                 controller_memos, controller_dynvars,
                                                                                 controller_action, controller_update,
                                                                                 par,
                                                                                 modules_name2pos, controller_name2pos,
                                                                                 controller_styles)

    # save the plots
    bkplot.save(bklayouts.grid([[mRNA_fig, prot_fig, misc_fig],
                                [F_fig, None, None],
                                [ctrl_memo_fig, ctrl_dynvar_fig, ctrl_u_fig]]))
    
    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
