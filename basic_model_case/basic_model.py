'''
BASIC_MODEL.PY: Python/Jax implementation of the basic resource-aware and host-aware synthetic gene expression model
Class to enable resource-aware simulation of synthetic gene expression in the cell
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import functools
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts

import time

# CIRCUIT AND EXTERNAL INPUT IMPORTS -----------------------------------------------------------------------------------
# import basic_genetic_modules as gms
# import controllers as ctrls
# import reference_switchers as refsws
# import ode_solvers as odesols


# CELL MODEL FUNCTIONS -------------------------------------------------------------------------------------------------
# Definitions of functions appearing in the cell model ODEs
# growth rate
def l_calc(par, # model parameters
           e, # translation elongation rate
           B, # concentration of actively translating ribosomes
           prodeflux=None    # protein degradation flux (not considered here, kept for consistency with the cell model)
           ):
    return (e * B) / par['M']


# MODEL AUXILIARIES -----------------------------------------------------------------------------------------------
# Auxiliries for the cell model - set up default parameters and initial conditions, plot simulation outcomes
class ModelAuxiliary:
    # INITIALISE
    def __init__(self):
        # plotting colours
        self.gene_colours = {'r': "#7E2F8E", 'o': '#C0C0C0', # colours for rribosomas and other genes
                             'het': "#0072BD"   # colour for heterologous genes
                             }
        return

    # PROCESS SYNTHETIC CIRCUIT MODULE
    # add synthetic circuit to the cell model
    def add_modules_and_controller(self,
                                   # module 1
                                   module1_initialiser,  # function initialising the circuit
                                   module1_ode,  # function defining the circuit ODEs
                                   module1_F_calc, # function calculating the circuit genes' transcription regulation functions
                                   module1_specterms,  # function calculating the effective mRNA concentrations due to possible co-expression of genes from the same operon
                                   # module 2
                                   module2_initialiser,  # function initialising the circuit
                                   module2_ode,  # function defining the circuit ODEs
                                   module2_F_calc, # function calculating the circuit genes' transcription regulation functions
                                   module2_specterms,  # function calculating the effective mRNA concentrations due to possible co-expression of genes from the same operon
                                   # controller
                                   controller_initialiser,  # function initialising the controller
                                   controller_action,  # function calculating the control action
                                   controller_ode,  # function defining the controller ODEs
                                   controller_update,  # function updating the controller based on measurements
                                   # cell model parameters and initial conditions
                                   model_par, model_init_conds,
                                   # host cell model parameters and initial conditions
                                   # currently no support for hybrid simulations
                                   ):
        # call initialisers
        module1_par, module1_init_conds, module1_genes, module1_miscs, module1_name2pos, module1_styles = module1_initialiser()
        module2_par, module2_init_conds, module2_genes, module2_miscs, module2_name2pos, module2_styles = module2_initialiser()
        controller_par, controller_init_conds, controller_init_memory, \
            controller_memos, controller_dynvars, controller_ctrledvar, \
            controller_name2pos, controller_styles = controller_initialiser()

        # update parameter, initial condition
        model_par.update(module1_par)
        model_par.update(module2_par)
        model_par.update(controller_par)
        model_init_conds.update(module1_init_conds)
        model_init_conds.update(module2_init_conds)
        model_init_conds.update(controller_init_conds)

        # merge style dictionaries for the two modules
        modules_styles = module1_styles.copy()
        modules_styles['colours'].update(module2_styles['colours'])
        modules_styles['dashes'].update(module2_styles['dashes'])
        
        # merge name-to-position dictionaries for the two modules: requires rearranging the variables
        # new order in x: module 1 proteins - module 2 proteins - module 1 misc - module 2 misc
        # order in F the same
        modules_name2pos = module1_name2pos.copy()
        modules_name2pos.update(module2_name2pos)
        for key in modules_name2pos.keys():
            # module 1 proteins left as they are
            if((key[0:2]=='p_') and (key in module1_name2pos)):
                modules_name2pos[key] = modules_name2pos[key]
            # module 2 proteins shifted by the number of module 1 proteins
            elif((key[0:2]=='p_') and (key in module2_name2pos)):
                modules_name2pos[key] = modules_name2pos[key] + len(module1_genes)
            # module 1 resource demands left as they are
            elif((key[0:2]=='q_') and (key in module1_name2pos)):
                continue
            # module 2 resource demands shifted by the number of module 1 genes
            elif((key[0:2]=='q_') and (key in module2_name2pos)):
                modules_name2pos[key] = module2_name2pos[key] + len(module1_genes)
            # module 1 and 2 F values kept as they are
            elif((key[0:2]=='F_')):
                continue
            else: # miscellaneous species
                #  module 1 misc shifted by the number of module 2 proteins
                if(key in module1_name2pos):
                    modules_name2pos[key] = module1_name2pos[key] + len(module2_genes)
                #  module 2 misc shifted by the number of module 1 proteins and miscs
                else:
                    modules_name2pos[key] = module2_name2pos[key] + len(module1_genes) +len(module1_miscs)

        # update controller name-to-position dictionary now that genetic modules have been added
        # (for dynamic variables described with ODEs)
        for dynvar in controller_dynvars:
            if(dynvar in controller_name2pos):
                controller_name2pos[dynvar] = controller_name2pos[dynvar] + len(module1_genes) + len(module2_genes) + len(module1_miscs) + len(module2_miscs)

        # merge gene and miscellaneous species lists
        synth_genes = module1_genes + module2_genes
        synth_miscs = module1_miscs + module2_miscs

        # join the circuit ODEs with the transcription regulation functions
        module1_ode_with_F_calc = lambda t, x, u, e, l, R, q_het, D, par, name2pos: module1_ode(module1_F_calc,
                                                                                                t, x,
                                                                                                u,
                                                                                                e,
                                                                                                l,
                                                                                                R,
                                                                                                q_het, D,
                                                                                                par,
                                                                                                name2pos)
        module2_ode_with_F_calc = lambda t, x, u, e, l, R, q_het, D, par, name2pos: module2_ode(module2_F_calc,
                                                                                                t, x,
                                                                                                u,
                                                                                                e,
                                                                                                l,
                                                                                                R,
                                                                                                q_het, D,
                                                                                                par,
                                                                                                name2pos)

        # not doing stochastic simulations => v functions will be irrelevant

        # add the geetic module and controller ODEs (as well as control action calculator) to that of the host cell model
        model_ode = lambda t, x, us, args: odeuus(t, x, us,
                                                  module1_ode_with_F_calc, module2_ode_with_F_calc,
                                                  module1_specterms, module2_specterms,
                                                  controller_ode, controller_action,
                                                  args,
                                                  module1_F_calc, module2_F_calc)

        # return updated ode and parameter, initial conditions, circuit gene (and miscellaneous specie) names
        # name - position in state vector decoder and colours for plotting the circuit's time evolution
        return (model_ode,
                module1_F_calc, module2_F_calc,
                module1_specterms, module2_specterms,
                controller_action, controller_update,
                model_par, model_init_conds, controller_init_memory,
                (synth_genes, module1_genes, module2_genes),
                (synth_miscs, module1_miscs, module2_miscs),
                controller_memos, controller_dynvars, controller_ctrledvar,
                modules_name2pos, modules_styles,
                controller_name2pos, controller_styles,
                # currently not supporting hybrid simulations => return nones instead of v functions
                None, None)
    
    # add reference tracker to the cell model
    def add_reference_switcher(self,
                              model_par, # cell model parameters
                              reference_switcher_initialiser, # function initialising the reference tracker
                              reference_switcher_switcher # function switching to next reference when it is time to do so
                              ):
        # call initialiser
        reference_switcher_par = reference_switcher_initialiser()
        
        # update cell model parameters
        model_par.update(reference_switcher_par)
        
        # return
        return (model_par, reference_switcher_switcher)

    # function packaging synthetic gene parameters into jax arrays
    # namely, returning a jnp array of resource demands
    def synth_gene_params_for_jax(self, par,  # system parameters
                                  synth_genes  # circuit gene names
                                  ):
        # initialise parameter arrays
        q_het = np.zeros(len(synth_genes))

        # fill in the arrays
        for i in range(0, len(synth_genes)):
            q_het[i] = par['q_' + synth_genes[i]]

        # return as a tuple of arrays
        return (jnp.array(q_het),)

    # SET DEFAULTS
    # set default parameters
    def default_params(self):
        '''
        Parameters based on:
        Sechkar A et al. 2024 A coarse-grained bacterial cell model for resource-aware analysis and design of synthetic gene circuits
        '''

        params = {}  # initialise

        # GENERAL PARAMETERS
        params['M'] = 1.19e9  # mass of protein in the cell (aa)
        params['e'] = 66077.664  # translation elongation rate (aa/h)

        # GENE EXPRESSION parAMETERS
        # ribosomal genes
        params['q_r'] = 13005.314453125  # resource demand
        params['n_r'] = 7459.0  # protein length (aa)

        # other (non-ribosomal) native genes
        params['q_o'] = 61169.44140625  # resource demand
        params['n_o'] = 300.0  # protein length (aa)
        return params

    # set default initial conditions
    def default_init_conds(self, par):
        init_conds = {}  # initialise

        # protein concentrations - start with 50/50 allocation as a convention
        init_conds['R'] = par['M'] / (2 * par['n_r'])  # ribosomal
        init_conds['p_o'] = par['M'] / (2 * par['n_o'])  # other

        return init_conds

    # PREPARE FOR SIMULATIONS
    # set default initial condition vector
    def x0_from_init_conds(self, init_conds,
                           par,
                           synth_genes, synth_miscs, controller_dynvars,
                           modules_name2pos, controller_name2pos):
        # NATIVE GENES
        x0 = [
            # ribosomal proteins
            init_conds['R'],

            # other native proteins
            init_conds['p_o'],
        ]
        # pad out x0 for consistency with the cell model
        x0 += [0]*6
        # GENETIC MODULES
        x0 += [0]*(len(synth_genes)+len(synth_miscs))  # initialise synthetic circuit species to zero
        for gene in synth_genes:  # mRNAs and proteins
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
                if ('n_'+misc in par.keys()):
                    top_line += xs[:, modules_name2pos[misc]] * par['n_' + misc]

            mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                              line_width=0.5, line_color='black', fill_color=self.gene_colours['het'],
                              legend_label='het')
        else:
            top_line = np.zeros(xs.shape[0])

        # plot mass of ribosomes - only if there are any to begin with
        bottom_line = top_line
        top_line = bottom_line + xs[:, 0] * par['n_r']
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['r'],
                          legend_label='R')

        # plot other protein mass
        bottom_line = top_line
        top_line = bottom_line + xs[:, 1] * par['n_o']
        mass_figure.patch(np.concatenate((ts, flip_t)), np.concatenate((bottom_line, np.flip(top_line))),
                          line_width=0.5, line_color='black', fill_color=self.gene_colours['o'], legend_label='p_o')

        # add legend
        mass_figure.legend.label_text_font_size = "8pt"
        mass_figure.legend.location = "top_right"

        return mass_figure

    # plot mRNA, protein and tRNA concentrations over time
    def plot_native_concentrations(self, ts, xs,
                                   par,  # model parameters
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
            p_het += xs[:, modules_name2pos['p_' + gene]]
        for misc in synth_miscs: # some miscellaneous species may be heterologous mRNAs or proteins having undergone some changes
            if ('n_'+misc in par.keys()):
                p_het += xs[:, modules_name2pos[misc]]

        # Create a ColumnDataSource object for the plot
        source = bkmodels.ColumnDataSource(data={
            't': ts,
            'R': xs[:, 0],  # ribosomal protein
            'p_o': xs[:, 1],  # other native protein
            'p_het': p_het,  # heterologous protein
        })

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
        protein_figure.line(x='t', y='R', source=source, line_width=1.5, line_color=self.gene_colours['r'],
                            legend_label='R')  # plot ribosomal protein concentrations
        protein_figure.line(x='t', y='p_o', source=source, line_width=1.5, line_color=self.gene_colours['o'],
                            legend_label='p_o')  # plot other native protein concentrations
        protein_figure.line(x='t', y='p_het', source=source, line_width=1.5, line_color=self.gene_colours['het'],
                            legend_label='p_het')  # plot heterologous protein concentrations
        protein_figure.legend.label_text_font_size = "8pt"
        protein_figure.legend.location = "top_right"
        protein_figure.legend.click_policy = 'hide'


        # basic model only considers proteins => return None for mRNA/tRNA/chloramphenicol for consistency with cell model
        return None, protein_figure, None, None

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
        # record synthetic protein concentrations
        for i in range(0, len(synth_genes)):
            data_for_column['p_' + synth_genes[i]] = xs[:, 8 + i]
        # record miscellaneous species' concentrations
        for i in range(0, len(synth_miscs)):
            data_for_column[synth_miscs[i]] = xs[:, 8 + len(synth_genes) + i]
        source = bkmodels.ColumnDataSource(data=data_for_column)

        # PLOT mRNA and PROTEIN CONCENTRATIONS (IF ANY)
        if (len(synth_genes) > 0):
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

        # basic model only considers proteins => return None for mRNA for consistency with cell model
        return None, protein_figure, misc_figure

    # plot transcription regulation function values for the circuit's genes
    def plot_circuit_regulation(self, ts, xs,   # time points and state vectors
                                ctrl_memorecord, uexprecord, # controller memory and experienced control actions records
                                refrecord, # reference tracker records
                                module1_F_calc, module2_F_calc, # transcription regulation functions for both modules
                                controller_action, # control action calculator
                                par, # model parameters
                                synth_genes_total_and_each,     # list of synthetic genes - total and for each module
                                synth_miscs_total_and_each,     # list of synthetic miscellaneous species - total and for each module
                                modules_name2pos,   # dictionary mapping gene names to their positions in the state vector
                                module1_specterms, module2_specterms, # calculating module-specific terms of ODEs and cellular process rate definitions
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

        # if no circuitry, return no plots
        if (len(synth_genes) == 0):
            return None

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get the state vector with effective mRNA concs (due to possible synth gene co-expression from same operons)
        module1_specterms_vmap = jax.vmap(module1_specterms, in_axes=(0, None, None))
        module2_specterms_vmap = jax.vmap(module2_specterms, in_axes=(0, None, None))

        # find values of gene transcription regulation functions
        Fs1 = np.zeros((len(ts), len(module1_genes)))  # initialise
        Fs2 = np.zeros((len(ts), len(module2_genes)))  # initialise
        for i in range(0, len(ts)):
            Fs1[i, :] = np.array(module1_F_calc(ts[i], xs[i, :], uexprecord[i], par, modules_name2pos)[:])
            Fs2[i, :] = np.array(module2_F_calc(ts[i], xs[i, :], uexprecord[i], par, modules_name2pos)[:])

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
    def plot_controller(self, ts, xs, # time points and state vectors
                        ctrl_memorecord, uexprecord,  # controller memory and experienced control actions records
                        refrecord,  # reference tracker records
                        memos, dynvars, # controller memo and dynamic variable names
                        ctrled_var, # name of the variable read and steered by the controller
                        controller_action, controller_update, # control action calculator and controller state update
                        par, # model parameters
                        synth_genes, synth_miscs,  # list of synthetic genes and miscellaneous species in the cell
                        modules_name2pos,  # dictionary mapping gene names to their positions in the state vector
                        module1_specterms, module2_specterms, # calculating module-specific terms of ODEs and cellular process rate definitions
                        controller_name2pos, # dictionary mapping controller species to their positions in the state vector
                        controller_styles, # colours for the controller plots
                        u0, control_delay,  # initial control action and size of control action record needed to account for the delay
                        dimensions=(320, 180), tspan=None):

        # get calculated control action values
        us_calc = self.get_u_calc(ts, xs, ctrl_memorecord, refrecord,
                                  controller_action,
                                  par,
                                  synth_genes, synth_miscs,
                                  modules_name2pos,
                                  module1_specterms, module2_specterms,
                                  controller_name2pos,
                                  ctrled_var)

        # set default time span if unspecified
        if(tspan==None):
            tspan=(ts[0], ts[-1])

        # tracked reference figure
        # create a ColumnDataSource object for the plot
        data_for_column = {'t': ts}
        data_for_column['ref'] = refrecord
        source = bkmodels.ColumnDataSource(data=data_for_column)

        ref_fig = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="t, hours",
            y_axis_label="Reference values",
            x_range=tspan,
            title='Reference values',
            tools="box_zoom,pan,hover,reset"
        )
        ref_fig.line(x='t', y='ref', source=source, line_width=1.5, line_color='blue', legend_label='ref')
        # if we have a controlled variable, plot it
        if (ctrled_var != ''):
            ref_fig.line(x=ts, y=xs[:, modules_name2pos[ctrled_var]], line_width=1.5, line_color='red', legend_label=ctrled_var)
        # legend formatting
        ref_fig.legend.label_text_font_size = "8pt"
        ref_fig.legend.location = "top_right"
        ref_fig.legend.click_policy = 'hide'


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
        u_fig.line(x=ts, y=us_calc, line_width=1.5, line_color='blue', legend_label='u (calculated)')  # plot calculated control actions
        u_fig.line(x=ts, y=uexprecord, line_width=1.5, line_color='red', legend_label='u (experienced)')  # plot experienced control actions
        # legend formatting
        u_fig.legend.label_text_font_size = "8pt"
        u_fig.legend.location = "top_right"
        u_fig.legend.click_policy = 'hide'

        return ref_fig, memo_fig, dynvar_fig, u_fig


    # plot the control action vs the controlled variable trajectory
    def plot_control_action_vs_controlled_var(self, ts, xs, # time points and state vectors
                        ctrl_memorecord, uexprecord,  # controller memory and experienced control actions records
                        refrecord,  # reference tracker records
                        refs,  # reference tracker records
                        memos, dynvars, # controller memo and dynamic variable names
                        ctrled_var, # name of the variable read and steered by the controller
                        controller_action, controller_update, # control action calculator and controller state update
                        par, # model parameters
                        synth_genes, synth_miscs,   # list of synthetic genes and miscellaneous species in the cell
                        modules_name2pos,   # dictionary mapping gene names to their positions in the state vector
                        module1_specterms, module2_specterms, # calculating module-specific terms of ODEs and cellular process rate definitions
                        controller_name2pos, # dictionary mapping controller species to their positions in the state vector
                        controller_styles, # colours for the controller plots
                        u0, control_delay,  # initial control action and size of control action record needed to account for the delay
                        dimensions=(320, 180), tspan=None):
        # get controlled variable values
        x_ctrl = xs[:, modules_name2pos[ctrled_var]]

        # get calculated control action values
        us_calc = self.get_u_calc(ts, xs, ctrl_memorecord, refrecord,
                                  controller_action,
                                  par,
                                  synth_genes, synth_miscs,
                                  modules_name2pos,
                                  module1_specterms, module2_specterms,
                                  controller_name2pos,
                                  ctrled_var)

        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # PLOT
        u_vs_var_fig = bkplot.figure(
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            x_axis_label="u, control action",
            y_axis_label=ctrled_var+", controlled variable",
            x_range=(min(us_calc), max(us_calc)),
            y_range=(min(refs), max(refs)),
            title='Control action vs controlled variable',
            tools="box_zoom,pan,hover,reset"
        )
        u_vs_var_fig.line(x=us_calc, y=x_ctrl, line_width=1.5, line_color='blue', legend_label='u calculated')
        u_vs_var_fig.line(x=uexprecord, y=x_ctrl, line_width=1.5, line_color='red', legend_label='u experienced')
        # legend formatting
        u_vs_var_fig.legend.label_text_font_size = "8pt"
        u_vs_var_fig.legend.location = "top_right"
        u_vs_var_fig.legend.click_policy = 'hide'

        return u_vs_var_fig

    # plot physiological variables: growth rate, translation elongation rate, ribosomal gene transcription regulation function, ppGpp concentration, tRNA charging rate, RC denominator
    def plot_phys_variables(self, ts, xs,
                            par, synth_genes, synth_miscs, modules_name2pos,
                            module1_specterms, module2_specterms,
                            # inputs only used by this (basic) model
                            module1_F_calc, module2_F_calc,
                            uexprecord,
                            synth_genes_total_and_each, synth_miscs_total_and_each,
                            # optional inputs used by both models
                            dimensions=(320, 180), tspan=None,
                            ):
        # set default time span if unspecified
        if (tspan == None):
            tspan = (ts[0], ts[-1])

        # get cell variables' values over time
        es, ls, _, _, _, _, D = self.get_e_l_Fr_nu_psi_T_D(ts, xs, par, synth_genes, synth_miscs,
                                                            modules_name2pos, module1_specterms, module2_specterms,
                                                           module1_F_calc, module2_F_calc, uexprecord, synth_genes_total_and_each, synth_miscs_total_and_each)

        # Create a ColumnDataSource object for the plot
        data_for_column = {'t': np.array(ts), 'l': np.array(ls), 'D': np.array(D),
                           'e': np.array(es)
                           }
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
            title='Translation elongation rate',
            tools="box_zoom,pan,hover,reset"
        )
        e_figure.line(x='t', y='e', source=source, line_width=1.5, line_color='blue', legend_label='e')
        e_figure.legend.label_text_font_size = "8pt"
        e_figure.legend.location = "top_right"
        e_figure.legend.click_policy = 'hide'


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

        # basic cell model does not consider ribosome regulation by ppGpp or tRNA aminoacylation => return Nones for Fr, ppGpp and nu for consistency with the cell model
        return l_figure, e_figure, None, None, None, D_figure


    # find values of different cellular variables
    def get_e_l_Fr_nu_psi_T_D(self, t, x,
                              par,
                              synth_genes,
                              synth_miscs,
                              modules_name2pos,
                              module1_specterms, module2_specterms,
                              # inputs only used by this (basic) model
                              module1_F_calc, module2_F_calc,  # transcription regulation functions for both modules
                              uexprecord,  # experienced control actions records
                              synth_genes_total_and_each,   # list of synthetic genes - total and for each module
                              synth_miscs_total_and_each   # list of synthetic miscellaneous species - total and for each module
                              ):
        # get the state vector with effective mRNA concs (due to possible synth gene co-expression from same operons)
        module1_specterms_vmap = jax.vmap(module1_specterms, in_axes=(0, None, None))
        module2_specterms_vmap = jax.vmap(module2_specterms, in_axes=(0, None, None))
        x_eff = np.array(jnp.concatenate((x[:, 0:8],
                                 module1_specterms_vmap(x, par, modules_name2pos)[0],    # eff mRNA concs are the first output of specterms
                                 module2_specterms_vmap(x, par, modules_name2pos)[0],    # eff mRNA concs are the first output of specterms
                                 x[:,8+len(synth_genes):]), axis=1))

        # give the state vector entries meaningful names
        R = x_eff[:, 0]  # ribosomes
        p_o = x_eff[:, 1] # other (non-ribosomal) native proteins

        # CALCULATE PHYSIOLOGICAL VARIABLES
        # translation elongation rate
        e = par['e']*jnp.ones_like(t)  # translation elongation rate - assumed constant

        # heterologous genes' resource demand
        if(len(synth_genes)==0):
            q_het = jnp.zeros_like(t)
        else:
            # get lists of synthetic genes and miscellaneous species
            synth_genes = synth_genes_total_and_each[0]
            module1_genes = synth_genes_total_and_each[1]
            module2_genes = synth_genes_total_and_each[2]
            synth_miscs = synth_miscs_total_and_each[0]
            module1_miscs = synth_miscs_total_and_each[1]
            module2_miscs = synth_miscs_total_and_each[2]

            # find values of gene transcription regulation functions
            Fs1_np = np.zeros((len(t), len(module1_genes)))  # initialise np array
            Fs2_np = np.zeros((len(t), len(module2_genes)))  # initialise np array
            for i in range(0, len(t)):
                Fs1_np[i, :] = np.array(module1_F_calc(t[i], x[i, :], uexprecord[i], par, modules_name2pos)[:])
                Fs2_np[i, :] = np.array(module2_F_calc(t[i], x[i, :], uexprecord[i], par, modules_name2pos)[:])

            # find the synthetic genes' total resource demand
            q_het_np = np.zeros_like(t) # initialise np array
            for gene in module1_genes:
                q_het_np += Fs1_np[:, modules_name2pos['F_'+gene]]*par['q_'+gene]
            for gene in module2_genes:
                q_het_np += Fs2_np[:, modules_name2pos['F_'+gene]]*par['q_'+gene]
            # convert to jnp array
            q_het = jnp.array(q_het_np)

        # resource competition denominator
        D = 1 + par['q_r'] + par['q_o'] + q_het  # resource competition denominator
        B = R * (1 - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

        l = l_calc(par, e, B)  # growth rate

        # basic cell model does not consider ribosome regulation by ppGpp or tRNA aminoacylation => return Nones for Fr, nu, psiu or T for consistency with the cell model
        return e, l, None, None, None, None, D

    # find calculated control inputs
    def get_u_calc(self, t, x,
                   ctrl_memo,
                   ref,
                   controller_action,
                   par,
                   synth_genes, synth_miscs,
                   modules_name2pos,
                   module1_specterms, module2_specterms,
                   controller_name2pos,
                   ctrled_var
                   ):
        # get calculated control actions
        u_calc = np.zeros(len(t))
        for i in range(0, len(t)):
            u_calc[i] = controller_action(t[i], x[i, :], ctrl_memo[i], ref[i], par,
                                          modules_name2pos, controller_name2pos, ctrled_var)

            # return calculated and experienced control actions
        return u_calc

    # find values of circuit gene transcription regulation functions
    def get_Fs(self, ts, xs,  # time points and state vectors
           uexprecord,  # experienced control actions records
           module1_F_calc, module2_F_calc,  # transcription regulation functions for both modules
           par,  # model parameters
           synth_genes_total_and_each,  # list of synthetic genes - total and for each module
           module1_specterms, module2_specterms,
           # calculating module-specific terms of ODEs and cellular process rate definitions
           modules_name2pos,  # dictionary mapping gene names to their positions in the state vector
           ):
        # get lists of synthetic genes and miscellaneous species
        synth_genes = synth_genes_total_and_each[0]
        module1_genes = synth_genes_total_and_each[1]
        module2_genes = synth_genes_total_and_each[2]

        # find values of gene transcription regulation functions
        Fs1 = np.zeros((len(ts), len(module1_genes)))  # initialise np array
        Fs2 = np.zeros((len(ts), len(module2_genes)))  # initialise np array
        for i in range(0, len(ts)):
            Fs1[i, :] = np.array(module1_F_calc(ts[i], xs[i, :], uexprecord[i], par, modules_name2pos)[:])
            Fs2[i, :] = np.array(module2_F_calc(ts[i], xs[i, :], uexprecord[i], par, modules_name2pos)[:])

        return Fs1, Fs2


# DETERMINISTIC SIMULATION ---------------------------------------------------------------------------------------------
# ode
def odeuus(t, x, # simulation time and state vector
           us, # control action record - from the one calclated control_delay hours ago to the latest calculated control action
           module1_ode, module2_ode,  # ODEs for the genetic modules
           module1_specterms, module2_specterms,  # corrections for effective mRNA concentrations for the genetic modules
           controller_ode, controller_action,  # ODE and control action calculation for the controller
           args,
           module1_F_calc, module2_F_calc   # regulatory function calculators for the modules
           ):
    # unpack the args
    par = args[0]  # model parameters
    modules_name2pos = args[1]  # gene name - position in circuit vector decoder
    controller_name2pos = args[2]  # controller name - position in circuit vector decoder
    num_synth_genes, num_synth_genes1, num_synth_genes2 = args[3] # number of synthetic genes: total and for each module
    num_synth_miscs, num_synth_miscs1, num_synth_miscs2 = args[4]  # number of miscellaneous species: total and for each module
    q_het_max, = args[5]  # unpack jax-arrayed synthetic gene parameters for calculating heterologous genes' resource demand

    # get the name of the variable read and steered by the controller
    ctrledvar = args[6]

    # get the controller memory and currently tracked reference
    ctrl_memo = args[7]
    ref = args[8]

    # give the state vector entries meaningful names
    R = x[0]  # ribosomes
    p_o = x[1]  # other (non-ribosomal) native proteins
    # synthetic circuit genes and miscellaneous species can be accessed directly from x with modules_name2pos

    # CONTROL ACTION CALCULATION
    u_calculated = controller_action(t, x, ctrl_memo, ref, par, modules_name2pos, controller_name2pos, ctrledvar)
    us_with_latest_calculated = jnp.append(us, u_calculated)  # add the latest calculated control action to the record
    u = us_with_latest_calculated[0]  # exerting control action calculated control_delay hours ago

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = par['e']  # translation elongation rate - assumed constant

    # heterologous gene transcription regulation functions
    Fs1=module1_F_calc(t ,x, u, par, modules_name2pos)
    Fs2=module2_F_calc(t ,x, u, par, modules_name2pos)
    Fs=jnp.concatenate((Fs1,Fs2))

    # total resource demand of heterologous genes
    q_het = jnp.sum(Fs * q_het_max)  # heterologous gene resource demand

    # resource competition denominator
    D = 1 + par['q_r'] + par['q_o'] + q_het  # resource competition denominator
    B = R * (1 - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    # growth rate
    l = l_calc(par, e, B)

    # GENETIC MODULE ODE CALCULATION
    module1_ode_value = module1_ode(t, x, u, e, l, R, q_het, D, par, modules_name2pos)
    module2_ode_value = module2_ode(t, x, u, e, l, R, q_het, D, par, modules_name2pos)

    # return dx/dt for the host cell
    dxdt = jnp.array([
                        # ribosomal proteins
                        (e/par['n_r'])*(par['q_r']/D)*R - l*R,
                        # other (non-ribosomal) native proteins
                        (e/par['n_o'])*(par['q_o']/D)*R - l*p_o,
                     ]
                    +
                    # pad out x0 for consistency with the cell model
                    [0.0] * 6
                    +
                    # add synthetic protein ODEs
                    module1_ode_value[0:num_synth_genes1] + module2_ode_value[0:num_synth_genes2]
                    +
                    # add miscellaneous species ODEs
                    module1_ode_value[num_synth_genes1:] + module2_ode_value[num_synth_genes2:]
                    +
                    # add controller ODEs
                    controller_ode(t, x, ctrl_memo, ref, e, l, R, D, par, modules_name2pos, controller_name2pos, ctrledvar))
    # return the ODE value, experienced control action and the updated control actions record (but without the oldest calculated control action)
    return (dxdt, u, us_with_latest_calculated[1:])

# ode simulation loop
def ode_sim(par,  # dictionary with model parameters
            ode_solver,  # ODE solver for the cell with the synthetic gene circuit
            odeuus_complete,  # ODE function for the cell with the synthetic gene circuit and the controller (also gives calculated and experienced control actions)
            ctrledvar,  # name of the variable read and steered by the controller
            controller_update,  # function for updating the controller memory
            controller_action,  # function for calculating the control action
            x0,  # initial condition VECTOR
            ctrl_memo0,  # initial controller memory
            u0, # initial control action, applied before any measurement-informed actions reach the system
            num_synth_genes, num_synth_miscs, # number of synthetic genes and miscellaneous species in the circuit
            modules_name2pos, controller_name2pos, # variable name to position in the state vector decoders
            sgp4j, # some synthetic gene parameters in jax.array form - for efficient simulation
            tf,  # simulation time frame
            meastimestep,   # output measurement time window
            control_delay,  # control action delay before the calculated action reaches the system
            us_size,  # size of the control action record needed
            refs,  # reference values for the controller
            ref_switcher  # reference switcher
            ):
    # define the arguments for finding the next state vector
    args = (par,
            modules_name2pos, controller_name2pos,
            num_synth_genes, num_synth_miscs,
            sgp4j,
            ctrledvar)

    # time points at which we save the solution
    ts = jnp.arange(tf[0], tf[1] + meastimestep / 2, meastimestep)

    # initialise the calculated control actions record
    us0=jnp.array([u0]*us_size)

    # make the retrieval of next simulator state a lambda-function for jax.lax.scanning
    scan_step = lambda sim_state, t: ode_sim_step(sim_state, t,
                                                  meastimestep,
                                                  args,
                                                  ode_solver,
                                                  odeuus_complete,
                                                  controller_update,
                                                  jnp.array(refs), ref_switcher)

    # define the jac.lax.scan function
    ode_scan = lambda sim_state_rec0, ts: jax.lax.scan(scan_step, sim_state_rec0, ts)
    ode_scan_jit = jax.jit(ode_scan)

    # initalise the simulator state: (t, x, sim_step_cntr, record_step_cntr, key, tf, xs)
    sim_state0 = {'t': tf[0], 'x': x0,  # time, state vector
                  'ctrl_memo': jnp.array(ctrl_memo0),  # controller memory (jnp array format)
                  'i_ref': 0, 't_last_ref_switch': 0.0, # index of the currently tracked reference, time refernce was switched last
                  'tf': tf,  # overall simulation time frame
                  'us': us0  # control actions record (needed if control delay present)
                  }

    # run the simulation
    _, sim_outcome = ode_scan_jit(sim_state0, ts)

    # extract the simulation outcomes
    t= sim_outcome[0]
    xs = sim_outcome[1]
    ctrl_memorecord = sim_outcome[2]
    urecord = sim_outcome[3]
    refrecord = sim_outcome[4]

    # return the simulation outcomes
    return t, xs, ctrl_memorecord, urecord, refrecord

# one step of the ode simulation loop (from one measurement to the next)
def ode_sim_step(sim_state, t,
                 meastimestep,
                 args,
                 ode_solver,
                 odeuus_complete,
                 controller_update,
                 refs, ref_switcher,
                 ):
    # get the input experienced at this time step
    u_exp = odeuus_complete(t, sim_state['x'],  # simulation time and state vector
                                 sim_state['us'],  # control action record - empty as zero control delay
                                 args + (sim_state['ctrl_memo'], refs[sim_state['i_ref']])   # extra ODE arguments, including the controller memory and the currently tracked reference
                                 )[1]


    # get next measurement time
    next_t = sim_state['t'] + meastimestep

    # simulate the ODE until the next measurement
    # to get the state vector, experienced control action and the record of calculated control actions
    next_x, next_us=ode_solver(t0=t,
                                 x0=sim_state['x'],
                                 us0=sim_state['us'],
                                 # extra ODE arguments, including the controller memory and the currently tracked reference
                                 args=args + (sim_state['ctrl_memo'], refs[sim_state['i_ref']]))

    # update the controller memory
    # par= args[0]
    # modules_name2pos = args[1]
    # controller_name2pos = args[2]
    next_ctrl_memo = controller_update(next_t, next_x, sim_state['ctrl_memo'], refs[sim_state['i_ref']], 
                                       args[0], args[1], args[2], args[6],
                                       meastimestep)

    # check if by next measurement it will be time to switch the reference
    next_i_ref, next_t_last_ref_switch = ref_switcher(sim_state['i_ref'],  # current reference index
                                                      refs,  # list of references
                                                      sim_state['t_last_ref_switch'],  # time of last reference switch
                                                      next_t, next_x,
                                                      next_ctrl_memo,
                                                      args[0], args[1], args[2], args[6],
                                                      meastimestep)

    # update the overall simulation state
    next_sim_state = {'t': next_t,
                      'x': next_x,
                      'ctrl_memo': next_ctrl_memo,
                      'i_ref': next_i_ref,
                      't_last_ref_switch': next_t_last_ref_switch,
                      'tf': sim_state['tf'],
                      'us': next_us}

    return next_sim_state, (t, sim_state['x'], sim_state['ctrl_memo'], u_exp, refs[sim_state['i_ref']])


# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    # set up jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

    # chemically cybercontrolled probe
    # basic_circ_par={}
    # basic_circ_par['q_switch'] = 125.0  # resource competition factor for the switch gene
    # basic_circ_par['q_ofp_div_q_switch'] = 100.0  # ratio of the ofp's RC factor to that of the switch - defined like that due to co-expression from the same operon
    # basic_circ_par['baseline_switch'] = 0.05
    # basic_circ_par['K_switch'] = 250.0
    # basic_circ_par['I_switch'] = 0.1
    # basic_circ_par['q_ta'] = 45.0  # RC factor for the transcription activation factor
    # basic_circ_par['q_b'] = 6e4  # RC factor for the burdensome gene of the probe
    # basic_circ_par['mu_b'] = 1 / (13.6 / 60)
    # basic_circ_par['baseline_tai-dna'] = 0.01

    basic_circ_par = {}
    basic_circ_par['q_ofp'] = 125.0
    basic_circ_par['q_ofp2'] = 30062.5

    # initialise cell model
    model_auxil = ModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    model_par = model_auxil.default_params()  # get default parameter values
    init_conds = model_auxil.default_init_conds(model_par)  # get default initial conditions
    
    # add reference tracker switcher
    model_par_with_refswitch, ref_switcher = model_auxil.add_reference_switcher(model_par,   # cell model parameters
                                                                                   refsws.timed_switching_initialise,    # function initialising the reference switcher
                                                                                   refsws.timed_switching_switch   # function switching the references to be tracked
                                                                                   )

    # load synthetic genetic modules and the controller
    (odeuus_complete, \
        module1_F_calc, module2_F_calc, \
        module1_specterms, module2_specterms, \
        controller_action, controller_update, \
        par, init_conds, controller_memo0, \
        synth_genes_total_and_each, synth_miscs_total_and_each, \
        controller_memos, controller_dynvars, controller_ctrledvar,\
        modules_name2pos, modules_styles, controller_name2pos, controller_styles, \
        module1_v_with_F_calc, module2_v_with_F_calc) = model_auxil.add_modules_and_controller(
            # module 1
            gms.constfp_initialise,  # function initialising the circuit
            gms.constfp_ode,  # function defining the circuit ODEs
            gms.constfp_F_calc, # function calculating the circuit genes' transcription regulation functions
            gms.constfp_specterms,   # function correcting the effective mRNA concentrations due to possible co-expression from the same operons
            # module 2
            gms.constfp2_initialise,  # function initialising the circuit
            gms.constfp2_ode,  # function defining the circuit ODEs
            gms.constfp2_F_calc,
            gms.constfp2_specterms,  # function correcting the effective mRNA concentrations due to possible co-expression from the same operons
            # function calculating the circuit genes' transcription regulation functions
            # controller
            ctrls.ciref_initialise,  # function initialising the controller
            ctrls.ciref_action,  # function calculating the controller action
            ctrls.ciref_ode,  # function defining the controller ODEs
            ctrls.ciref_update,  # function updating the controller based on measurements
            # cell model parameters and initial conditions
            model_par_with_refswitch, init_conds)
    par.update(basic_circ_par)

    # unpack the synthetic genes and miscellaneous species lists
    synth_genes= synth_genes_total_and_each[0]
    module1_genes = synth_genes_total_and_each[1]
    module2_genes = synth_genes_total_and_each[2]
    synth_miscs= synth_miscs_total_and_each[0]
    module1_miscs = synth_miscs_total_and_each[1]
    module2_miscs = synth_miscs_total_and_each[2]

    # SET PARAMETERS
    # set the parameters for the synthetic genes
    # par['q_ofp']=0.5*(par['q_r']+par['q_o'])  # resource demand of the output fluorescent protein

    # controller
    init_conds['inducer_level']=1000.0

    # SET CONTROLLER PARAMETERS
    controller_ctrledvar='ofp_mature' # variable read and steered by the controller
    experiment_duration=24.0#24.0
    points_in_space=10
    refs=np.linspace(0.0, 15.0, points_in_space) # reference values
    par['t_switch_ref']=experiment_duration/points_in_space    # time of reference switch

    control_delay=0.0   # control action delay
    u0=0.0  # initial control action

    # inducer level when the bang-bang input is ON
    # par['inducer_level_on']=1e3
    # par['on_when_below_ref']=False

    par['Kp']=-0.01
    par['Ki']=0

    # DETERMINISTIC SIMULATION
    # set simulation parameters
    tf = (0.0, experiment_duration)  # simulation time frame

    # measurement time step
    meastimestep = 0.1  # hours

    # choose ODE solver
    # ode_solver, us_size = odesols.create_diffrax_solver(odeuus_complete,
    #                                                     control_delay=0,
    #                                                     meastimestep=meastimestep,
    #                                                     rtol=1e-6, atol=1e-6,
    #                                                     solver_spec='Kvaerno3')
    ode_solver, us_size = odesols.create_euler_solver(odeuus_complete,
                                                      control_delay=control_delay,
                                                      meastimestep=meastimestep,
                                                      euler_timestep=1e-5)

    # solve ODE
    timer= time.time()
    ts_jnp, xs_jnp,\
        ctrl_memorecord_jnp, uexprecord_jnp, \
        refrecord_jnp  = ode_sim(par,   # model parameters
                                 ode_solver,    # ODE solver for the cell with the synthetic gene circuit
                                 odeuus_complete,    # ODE function for the cell with the synthetic gene circuit and the controller (also gives calculated and experienced control actions)
                                 controller_ctrledvar,    # name of the variable read and steered by the controller
                                 controller_update, controller_action,   # function for updating the controller memory and calculating the control action
                                 model_auxil.x0_from_init_conds(init_conds,
                                                                    par,
                                                                    synth_genes, synth_miscs, controller_dynvars,
                                                                    modules_name2pos,
                                                                    controller_name2pos),   # initial condition VECTOR
                                 controller_memo0,  # initial controller memory record
                                 u0,    # initial control action, applied before any measurement-informed actions reach the system
                                 (len(synth_genes), len(module1_genes), len(module2_genes)),    # number of synthetic genes
                                 (len(synth_miscs), len(module1_miscs), len(module2_miscs)),    # number of miscellaneous species
                                 modules_name2pos, controller_name2pos, # dictionaries mapping gene names to their positions in the state vector
                                 model_auxil.synth_gene_params_for_jax(par, synth_genes),   # synthetic gene parameters in jax.array form
                                 tf, meastimestep,  # simulation time frame and measurement time step
                                 control_delay,  # delay before control action reaches the system
                                 us_size,  # size of the control action record needed
                                 refs, ref_switcher,  # reference values and reference switcher
                                 )

    # convert simulation results to numpy arrays
    ts = np.array(ts_jnp)
    xs = np.array(xs_jnp)
    ctrl_memorecord = np.array(ctrl_memorecord_jnp)
    uexprecord = np.array(uexprecord_jnp)
    refrecord= np.array(refrecord_jnp)

    print('Simulation time: ', time.time()-timer, ' s')

    # get the cell growth rates over time
    _, ls_jnp, _, _, _, _, _ = model_auxil.get_e_l_Fr_nu_psi_T_D(ts, xs, par,
                                                                 synth_genes, synth_miscs,
                                                                 modules_name2pos,
                                                                 module1_specterms, module2_specterms,
                                                                 # arguments only used by the basic model
                                                                 module1_F_calc, module2_F_calc,
                                                                 uexprecord,
                                                                 synth_genes_total_and_each, synth_miscs_total_and_each
                                                                 )
    ls = np.array(ls_jnp)
    l=ls[-1]

    print(
        par['M']/par['n_ofp2']*(par['q_ofp2'])/(1+par['q_r']+par['q_o']+par['q_ofp']+par['q_ofp2']) * (par['mu_ofp2']/(l+par['mu_ofp2']))
    )
    print(xs[-1, modules_name2pos['ofp2_mature']])

    # make plots
    # PLOT - HOST CELL MODEL
    bkplot.output_file(filename="../sim_tools/model_sim.html",
                       title="Cell Model Simulation")  # set up bokeh output file
    mass_fig = model_auxil.plot_protein_masses(ts, xs, par, synth_genes, synth_miscs, modules_name2pos)  # plot simulation results
    nat_mrna_fig, nat_prot_fig, nat_trna_fig, h_fig = model_auxil.plot_native_concentrations(ts, xs,
                                                                                                 par,
                                                                                                 synth_genes,
                                                                                                 synth_miscs,
                                                                                                 modules_name2pos)  # plot simulation results
    l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure = model_auxil.plot_phys_variables(ts,
                                                                                                       xs,
                                                                                                       par,
                                                                                                       synth_genes,
                                                                                                       synth_miscs,
                                                                                                       modules_name2pos,
                                                                                                       module1_specterms,
                                                                                                       module2_specterms,
                                                                                                       module1_F_calc, module2_F_calc,
                                                                                                       uexprecord,
                                                                                                       synth_genes_total_and_each,
                                                                                                       synth_miscs_total_and_each
                                                                                                       )  # plot simulation results
    bkplot.save(bklayouts.grid([[mass_fig, nat_mrna_fig, nat_prot_fig],
                                [nat_trna_fig, h_fig, l_figure],
                                [e_figure, Fr_figure, ppGpp_figure]]))

    # PLOT - SYNTHETIC CIRCUITS AND CONTROLLER
    bkplot.output_file(filename="../sim_tools/circuit_sim.html",
                       title="Synthetic Circuit Simulation")  # set up bokeh output file
    # plot synthetic circuit concentrations
    mRNA_fig, prot_fig, misc_fig = model_auxil.plot_circuit_concentrations(ts, xs,
                                                                              par, synth_genes, synth_miscs,
                                                                              modules_name2pos,
                                                                              modules_styles)  # plot simulation results
    # plot synthetic circuit regulation functions
    F_fig = model_auxil.plot_circuit_regulation(ts, xs, # time points and state vectors
                                                    ctrl_memorecord, uexprecord,    # controller memory and experienced control actions records
                                                    refrecord,  # reference tracker records
                                                    module1_F_calc, module2_F_calc, # transcription regulation functions for both modules
                                                    controller_action, # control action calculator
                                                    par, # model parameters
                                                    synth_genes_total_and_each,     # list of synthetic genes - total and for each module
                                                    synth_miscs_total_and_each,     # list of synthetic miscellaneous species - total and for each module
                                                    modules_name2pos,   # dictionary mapping gene names to their positions in the state vector
                                                    module1_specterms, module2_specterms, # corrections for effective mRNA concentrations for the genetic modules
                                                    controller_name2pos, # dictionary mapping controller species to their positions in the state vector
                                                    modules_styles)  # plot simulation results
    # plot controller memory, dynamic variables and actions
    ctrl_ref_fig, ctrl_memo_fig, ctrl_dynvar_fig, ctrl_u_fig = model_auxil.plot_controller(ts, xs,
                                                                                            ctrl_memorecord, uexprecord, # controller memory and experienced control actions records
                                                                                            refrecord, # reference tracker records
                                                                                            controller_memos, controller_dynvars,
                                                                                            controller_ctrledvar,
                                                                                            controller_action, controller_update,
                                                                                            par,
                                                                                            synth_genes, synth_miscs,
                                                                                            modules_name2pos,
                                                                                            module1_specterms, module2_specterms,
                                                                                            controller_name2pos,
                                                                                            controller_styles,
                                                                                            u0, control_delay)
    # control action vs controlled variable figure
    u_vs_ctrledvar_fig = model_auxil.plot_control_action_vs_controlled_var(ts, xs,
                                                             ctrl_memorecord, uexprecord, # controller memory and experienced control actions records
                                                             refrecord, # reference tracker records
                                                             refs,
                                                             controller_memos, controller_dynvars,
                                                             controller_ctrledvar,
                                                             controller_action, controller_update,
                                                             par,
                                                             synth_genes, synth_miscs,
                                                             modules_name2pos,
                                                             module1_specterms, module2_specterms,
                                                             controller_name2pos,
                                                             controller_styles,
                                                             u0, control_delay)
    # save the plots
    bkplot.save(bklayouts.grid([[mRNA_fig, prot_fig, misc_fig],
                                [F_fig, None, ctrl_ref_fig],
                                [ctrl_memo_fig, ctrl_dynvar_fig, ctrl_u_fig, u_vs_ctrledvar_fig]]))

    args = (par,
            modules_name2pos, controller_name2pos,
            (len(synth_genes), len(module1_genes), len(module2_genes)),  # number of synthetic genes
            (len(synth_miscs), len(module1_miscs), len(module2_miscs)),
            model_auxil.synth_gene_params_for_jax(par, synth_genes),
            '',jnp.array([0.0]),jnp.array([0.0]))
    print(odeuus_complete(ts_jnp[-1],xs_jnp[:,-1],jnp.array([0.0]),args))
    
    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
