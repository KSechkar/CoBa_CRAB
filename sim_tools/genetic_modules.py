'''
GENETIC_MODULES.PY: Describing different synthetic gene circuits
for the Python/Jax implementation of the coarse-grained resource-aware E.coli model
'''


# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import jax
import jax.numpy as jnp
import numpy as np

# NO SYNTHETIC GENES ---------------------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def nocircuit_initialise():
    default_par={'cat_gene_present':0, 'prot_gene_present':0} # chloramphenicol resistance or synthetic protease gene not present, natrually
    default_init_conds={}
    genes={}
    miscs={}
    name2pos={'p_cat':0, } # placeholders, will never be used but required for correct execution'}
    circuit_colours={}
    return default_par, default_init_conds, genes, miscs, name2pos, circuit_colours

# transcription regulation functions
def nocircuit_F_calc(t ,x, par, name2pos):
    return jnp.array([])

# ode
def nocircuit_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # RETURN THE ODE
    return []


# stochastic reaction propensities for hybrid tau-leaping simulations
def nocircuit_v(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            mRNA_count_scales, # scaling factors for mRNA counts
            par,  # system parameters
            name2pos
            ):
    # RETURN THE PROPENSITIES
    return []


# effective mRNA concentrations for genes expressed from the same operon with some others
def nocircuit_eff_mRNA(x, par, name2pos):
    return jnp.array([])


# CONSTITUTIVELY EXPRESSED FLUORESCENT PROTEIN -------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def constfp_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['ofp']  # names of genes in the circuit: output fluorescent protein
    miscs = []  # names of miscellaneous species involved in the circuit (e.g. metabolites)
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] =  i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] =  i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation by synthetic protease - zero by default (/h/nM)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat']=0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot']=0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#de3163ff", '#ff6700ff', '#48d1ccff', '#bb3385ff', '#fcc200ff']
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    # gene styles
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    # miscellaneous species styles
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    circuit_styles['colours']['ofp']='#00af00ff'
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def constfp_F_calc(t ,x, u, par, name2pos):
    F_ofp = 1 # constitutive gene
    return jnp.array([F_ofp])

# ode
def constfp_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state
            u, # controller input
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, u, par, name2pos)

    # RETURN THE ODE
    return [# mRNAs
            par['func_ofp'] * l * F[name2pos['F_ofp']] * par['c_ofp'] * par['a_ofp'] - (par['b_ofp'] + l) * x[name2pos['m_ofp']],
            # proteins
            (e / par['n_ofp']) * (x[name2pos['m_ofp']] / k_het[name2pos['k_ofp']] / D) * R - (l + par['d_ofp']*p_prot) * x[name2pos['p_ofp']]
    ]

# effective mRNA concentrations for genes expressed from the same operon with some others
def constfp_eff_mRNA(x, par, name2pos):
    return jnp.array([x[name2pos['m_ofp']]]) # no co-expression


# SELF-ACTIVATING BISTABLE SWITCH --------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
def sas_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    # names of genes in the circuit
    genes = ['switch',  # self0activating swich
             'ofp']  # burdensome controlled gene
    # names of miscellaneous species involved in the circuit (none)
    miscs = []
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] =  i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation by synthetic protease - zero by default (/h/nM)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat'] = 0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot'] = 0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # transcription regulation function
    default_par['I_switch'] = 1  # share of switch protein bound by the corresponding inducer (0<=I_switch<=1)
    default_par['K_switch'] = 100 # dissociation constant of the ta-inducer complex from the DNA
    default_par['eta_switch'] = 2 # Hill coefficient of the ta protein binding to the DNA
    default_par['baseline_switch'] = 0.1 # baseline expression of the burdensome gene

    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#de3163ff", '#ff6700ff', '#48d1ccff', '#bb3385ff', '#fcc200ff']
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles={'colours':{}, 'dashes':{}} # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    circuit_styles['colours']['switch']='#48d1ccff'
    circuit_styles['colours']['ofp']='#bb3385ff'
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def sas_F_calc(t ,x,
                u,  # controller input: external inducer concentration
                par, name2pos):

    # switch protein-dependent term - the concentration of active (inducer-bound) switch proteins divided by half-saturation constant
    p_switch_term = (x[name2pos['p_switch']] * par['I_switch'])/par['K_switch']

    F_switch = par['baseline_switch'] + (1 - par['baseline_switch']) * (p_switch_term**par['eta_switch'])/(p_switch_term**par['eta_switch']+1)
    return jnp.array([
        F_switch,
        F_switch    # ofp co-expressed with the switch gene from the same protein; this value will have no bearing on the ODE but is repeated for illustrative purposes
    ])

# ODE
def sas_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state
            u, # controller input
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, u, par, name2pos)

    # ofp reporter co-expressed from the same operon as the switch => same mRNA conc., up to scaling for multiple-ribosome translation
    # commented out as NOW TREATED USING SAS_EFF_MRNA FUNCTION!
    # m_ofp = x[name2pos['m_switch']]*par['n_ofp']/par['n_switch']

    # RETURN THE ODE
    return [# mRNAs
            par['func_switch'] * l * F[name2pos['F_switch']] * par['c_switch'] * par['a_switch'] - (par['b_switch'] + l) * x[name2pos['m_switch']],
            0,  # ofp co-expressed with the switch
            # proteins
            (e / par['n_switch']) * (x[name2pos['m_switch']] / k_het[name2pos['k_switch']] / D) * R - (l + par['d_switch']*p_prot) * x[name2pos['p_switch']],
            (e / par['n_ofp']) * (x[name2pos['m_ofp']] / k_het[name2pos['k_ofp']] / D) * R - (l + par['d_ofp']*p_prot) * x[name2pos['p_ofp']],
    ]

# effective mRNA concentrations for genes expressed from the same operon with some others
def sas_eff_mRNA(x, par, name2pos):
    m_ofp = x[name2pos['m_switch']] * par['n_ofp'] / par['n_switch']
    return jnp.array([x[name2pos['m_switch']], m_ofp]) # ofp co-expressed with the switch gene

# CHEMICALLY INDUCED, CYBERCONTROLLED GENE [tau-leap compatible, includes a synthetic protease]-------------------------
def cicc_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    # names of genes in the circuit
    genes = ['ta',  # transcription activation factor
             'b']  # burdensome controlled gene
    # names of miscellaneous species involved in the circuit (none)
    miscs = []
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)
    for i in range(0, len(genes)):
        name2pos['mscale_' + genes[i]] =  i  # mRNA count scaling factors (in mRNA_count_scales, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation by synthetic protease - zero by default (/h/nM)

    # special genes - must be handled in a particular way if not presemt
    # chloramphenicol acetlytransferase gene - antibiotic resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        # add placeholder to the position decoder dictionary - will never be used but are required for correct execution
        name2pos['p_cat'] = 0
    # synthetic protease gene - synthetic protein degradation
    if ('prot' in genes):
        default_par['prot_gene_present'] = 1
    else:
        default_par['prot_gene_present'] = 0
        name2pos['p_prot'] = 0

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # transcription regulation function
    default_par['K_ta-i'] = 100 # dissociation constant of the ta-inducer binding
    default_par['K_tai-dna'] = 100 # dissociation constant of the ta-inducer complex from the DNA
    default_par['eta_tai-dna'] = 2 # Hill coefficient of the ta protein binding to the DNA
    default_par['baseline_tai-dna'] = 0.1 # baseline expression of the burdensome gene

    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#de3163ff", '#ff6700ff', '#48d1ccff', '#bb3385ff', '#fcc200ff']
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles={'colours':{}, 'dashes':{}} # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def cicc_F_calc(t ,x,
                u,  # controller input: external inducer concentration
                par, name2pos):
    F_ta = 1 # constitutive gene

    # burdensome gene expression is regulated by the ta protein
    tai_conc = x[name2pos['p_ta']] * u/(u+par['K_ta-i'])
    F_b = par['baseline_tai-dna'] + (1 - par['baseline_tai-dna']) * (tai_conc**par['eta_tai-dna'])/(tai_conc**par['eta_tai-dna']+par['K_tai-dna']**par['eta_tai-dna'])
    return jnp.array([F_ta,
            F_b])

# ODE
def cicc_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state
            u, # controller input
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            p_prot, # synthetic protease concentration
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, u, par, name2pos)

    # RETURN THE ODE
    return [# mRNAs
            par['func_ta'] * l * F[name2pos['F_ta']] * par['c_ta'] * par['a_ta'] - (par['b_ta'] + l) * x[name2pos['m_ta']],
            par['func_b'] * l * F[name2pos['F_b']] * par['c_b'] * par['a_b'] - (par['b_b'] + l) * x[name2pos['m_b']],
            # proteins
            (e / par['n_ta']) * (x[name2pos['m_ta']] / k_het[name2pos['k_ta']] / D) * R - (l + par['d_ta']*p_prot) * x[name2pos['p_ta']],
            (e / par['n_b']) * (x[name2pos['m_b']] / k_het[name2pos['k_b']] / D) * R - (l + par['d_b']*p_prot) * x[name2pos['p_b']],
    ]

# effective mRNA concentrations for genes expressed from the same operon with some others
def cicc_eff_mRNA(x, par, name2pos):
    return jnp.array([x[name2pos['m_ta']], x[name2pos['m_b']]]) # no co-expression