'''
BASIC_GENETIC_MODULES.PY: Describing different synthetic gene circuits
for the Python/Jax implementation of the basic resource-aware and host-aware synthetic gene expression model
'''


# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import jax
import jax.numpy as jnp
import numpy as np

# NO SYNTHETIC GENES ---------------------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def nocircuit_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = []  # names of genes in the circuit: output fluorescent protein
    miscs = []  # names of miscellaneous species involved in the circuit (e.g. metabolites)
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['p_' + genes[i]] = 8 + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['q_' + genes[i]] = i  # resource demands (in q_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['q_' + gene] = 1.0  # resource demand (unitless)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
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
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def nocircuit_F_calc(t ,x, par, name2pos):
    return jnp.array([])

# ode
def nocircuit_ode(F_calc,  # calculating the transcription regulation functions
                  t, x,  # time, cell state
                  u,  # controller input
                  e, l,  # translation elongation rate, growth rate
                  R,  # ribosome count in the cell, resource
                  q_het, D,  # resource demands for synthetic genes, resource competition denominator
                  par,  # system parameters
                  name2pos  # name to position decoder
                  ):
    # RETURN THE ODE
    return []

# specific terms of ODEs and cellular process rate definitions
# 1) effective mRNA concentrations for genes expressed from the same operon with some others
# 2) contribution to protein degradation flux from miscellaneous species (normally, mature proteins)
def nocircuit_specterms(x, par, name2pos):
    return (jnp.array([]), 0)


# CONSTITUTIVELY EXPRESSED FLUORESCENT PROTEIN -------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def constfp_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['ofp']  # names of genes in the circuit: output fluorescent protein
    miscs = ['ofp_mature']  # names of miscellaneous species involved in the circuit (e.g. metabolites)
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['p_' + genes[i]] = 8 + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['q_' + genes[i]] =  i  # resource demands (in q_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] =  i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['q_' + gene] = 1.0  # resource demand (unitless)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    default_par['mu_ofp']=np.log(2)/(13.6/60)   # sfGFP maturation time of 13.6 min
    default_par['n_ofp_mature'] = default_par['n_ofp'] # protein length - same as the freshly synthesised protein
    default_par['d_ofp_mature'] = default_par['d_ofp']  # mature ofp degradation rate - same as the freshly synthesised protein
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
    circuit_styles['colours']['ofp_mature'] = '#00af00ff'
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def constfp_F_calc(t ,x, u, par, name2pos):
    F_ofp = 1 # constitutive gene
    return jnp.array([F_ofp])


# ode
def constfp_ode(F_calc,  # calculating the transcription regulation functions
                t, x,  # time, cell state
                u,  # controller input
                e, l,  # translation elongation rate, growth rate
                R,  # ribosome count in the cell, resource
                q_het, D,  # resource demands for synthetic genes, resource competition denominator
                par,  # system parameters
                name2pos  # name to position decoder
                ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, u, par, name2pos)

    # RETURN THE ODE
    return [# proteins
            (e/par['n_ofp'])*(par['q_ofp']/D)*R - l*x[name2pos['p_ofp']] - par['mu_ofp']*x[name2pos['p_ofp']],
            # mature fluorescent proteins
            par['mu_ofp']*x[name2pos['p_ofp']] - l*x[name2pos['ofp_mature']]
    ]

# specific terms of ODEs and cellular process rate definitions
# 1) effective mRNA concentrations for genes expressed from the same operon with some others
# 2) contribution to protein degradation flux from miscellaneous species (normally, mature proteins)
def constfp_specterms(x, par, name2pos):
    return (
        jnp.array([0.0]), # 1) the cell model would have mRNA concentration here, here this term is just for consistency
        0.0 # 2) mature ofp degradation - not considered, this term is just for consistency
    )

# SECOND CONSTITUTIVELY EXPRESSED FLUORESCENT PROTEIN ------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def constfp2_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['ofp2']  # names of genes in the circuit: output fluorescent protein
    miscs = ['ofp2_mature']  # names of miscellaneous species involved in the circuit (e.g. metabolites)
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['p_' + genes[i]] = 8 + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['q_' + genes[i]] =  i  # resource demands (in q_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] =  i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['q_' + gene] = 1.0  # resource demand (unitless)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    default_par['mu_ofp2']=np.log(2)/(13.6/60)   # sfGFP maturation time of 13.6 min
    default_par['n_ofp2_mature'] = default_par['n_ofp2'] # protein length - same as the freshly synthesised protein
    default_par['d_ofp2_mature'] = default_par['d_ofp2']  # mature ofp2 degradation rate - same as the freshly synthesised protein
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
    circuit_styles['colours']['ofp2']='#fe0087ff'
    circuit_styles['colours']['ofp2_mature'] = '#fe0087ff'
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def constfp2_F_calc(t ,x, u, par, name2pos):
    F_ofp2 = 1 # constitutive gene
    return jnp.array([F_ofp2])

# ode
def constfp2_ode(F_calc,  # calculating the transcription regulation functions
                t, x,  # time, cell state
                u,  # controller input
                e, l,  # translation elongation rate, growth rate
                R,  # ribosome count in the cell, resource
                q_het, D,  # resource demands for synthetic genes, resource competition denominator
                par,  # system parameters
                name2pos  # name to position decoder
                ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, u, par, name2pos)

    # RETURN THE ODE
    return [# proteins
            (e/par['n_ofp2'])*(par['q_ofp2']/D)*R - l*x[name2pos['p_ofp2']] - par['mu_ofp2']*x[name2pos['p_ofp2']],
            # mature fluorescent proteins
            par['mu_ofp2']*x[name2pos['p_ofp2']] - l*x[name2pos['ofp2_mature']]
    ]

# specific terms of ODEs and cellular process rate definitions
# 1) effective mRNA concentrations for genes expressed from the same operon with some others
# 2) contribution to protein degradation flux from miscellaneous species (normally, mature proteins)
def constfp2_specterms(x, par, name2pos):
    return (
        jnp.array([0.0]), # 1) the cell model would have mRNA concentration here, here this term is just for consistency
        0.0 # 2) mature ofp2 degradation - not considered, this term is just for consistency
    )

# SELF-ACTIVATING BISTABLE SWITCH --------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def sas_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    # names of genes in the circuit
    genes = ['s',  # self-activating swich
             'ofp']  # burdensome controlled gene
    # names of miscellaneous species involved in the circuit (none)
    miscs = ['ofp_mature']  # mature ofp
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['p_' + genes[i]] = 8 + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['q_' + genes[i]] = i  # resource demands (in q_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['q_' + gene] = 1.0  # resource demand (unitless)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # transcription regulation function
    default_par['I_s'] = 1  # share of switch protein bound by the corresponding inducer (0<=I_switch<=1)
    default_par['K_s,s'] = 100 # dissociation constant of the ta-inducer complex from the DNA
    default_par['eta_s,s'] = 2 # Hill coefficient of the ta protein binding to the DNA
    default_par['F_s,0'] = 0.1 # baseline expression of the burdensome gene

    # output fluorescent protein maturation
    default_par['mu_ofp'] = np.log(2) / (13.6 / 60)  # sfGFP maturation time of 13.6 min
    default_par['n_ofp_mature'] = default_par['n_ofp']  # protein length - same as the freshly synthesised protein
    default_par['d_ofp_mature'] = default_par['d_ofp']  # mature ofp degradation rate - same as the freshly synthesised protein

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
    circuit_styles['colours']['s'] = '#de3163ff'
    circuit_styles['colours']['ofp'] = '#00af00ff'
    circuit_styles['colours']['ofp_mature'] = '#00af00ff'
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def sas_F_calc(t ,x, u, par, name2pos):
    # switch protein-dependent term - the concentration of active (inducer-bound) switch proteins divided by half-saturation constant
    p_s_term = (x[name2pos['p_s']] * par['I_s']) / par['K_s,s']
    # switch protein regulation function
    F_s = par['F_s,0'] + (1 - par['F_s,0']) * (p_s_term ** par['eta_s,s']) / (
                p_s_term ** par['eta_s,s'] + 1)
    return jnp.array([
        F_s,
        F_s    # ofp co-expressed with the switch gene from the same protein; this value will have no bearing on the ODE but is repeated for illustrative purposes
    ])

# ODE
def sas_ode(F_calc,  # calculating the transcription regulation functions
                t, x,  # time, cell state
                u,  # controller input
                e, l,  # translation elongation rate, growth rate
                R,  # ribosome count in the cell, resource
                q_het, D,  # resource demands for synthetic genes, resource competition denominator
                par,  # system parameters
                name2pos  # name to position decoder
                ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, u, par, name2pos)

    # RETURN THE ODE
    return [# proteins
            (e / par['n_s']) * (F[name2pos['F_s']] * par['q_s'] / D) * R - l * x[name2pos['p_s']],
            (e/par['n_ofp']) * (F[name2pos['F_ofp']] * par['q_ofp'] / D) * R - l * x[name2pos['p_ofp']] - par['mu_ofp'] * x[name2pos['p_ofp']],
            # mature fluorescent proteins
            par['mu_ofp']*x[name2pos['p_ofp']] - l*x[name2pos['ofp_mature']]
    ]

# specific terms of ODEs and cellular process rate definitions
# 1) effective mRNA concentrations for genes expressed from the same operon with some others
# 2) contribution to protein degradation flux from miscellaneous species (normally, mature proteins)
def sas_specterms(x, par, name2pos):
    return (
        jnp.array([0.0]), # 1) the cell model would have mRNA concentration here, here this term is just for consistency
        0.0 # 2) mature ofp2 degradation - not considered, this term is just for consistency
    )

# SECOND SELF-ACTIVATING BISTABLE SWITCH -------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def sas2_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    # names of genes in the circuit
    genes = ['s2',  # self-activating swich
             'ofp2']  # burdensome controlled gene
    # names of miscellaneous species involved in the circuit (none)
    miscs = ['ofp2_mature']  # mature ofp
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['p_' + genes[i]] = 8 + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['q_' + genes[i]] = i  # resource demands (in q_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['q_' + gene] = 1.0  # resource demand (unitless)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # transcription regulation function
    default_par['I_s2'] = 1  # share of switch protein bound by the corresponding inducer (0<=I_switch<=1)
    default_par['K_s2,s2'] = 100 # dissociation constant of the ta-inducer complex from the DNA
    default_par['eta_s2,s2'] = 2 # Hill coefficient of the ta protein binding to the DNA
    default_par['F_s2,0'] = 0.1 # baseline expression of the burdensome gene

    # output fluorescent protein maturation
    default_par['mu_ofp2'] = np.log(2) / (13.6 / 60)  # sfGFP maturation time of 13.6 min
    default_par['n_ofp2_mature'] = default_par['n_ofp2']  # protein length - same as the freshly synthesised protein
    default_par['d_ofp2_mature'] = default_par['d_ofp2']  # mature ofp degradation rate - same as the freshly synthesised protein

    # special case: due to co-expression from the same operon, q_ofp2 = q_s2 * q_ofp2_div_q_s2
    default_par['q_ofp2_div_q_s2'] = 1.0
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
    circuit_styles['colours']['s2'] = '#48d1ccff'
    circuit_styles['colours']['ofp2'] = '#bb3385ff'
    circuit_styles['colours']['ofp2_mature'] = '#bb3385ff'
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def sas2_F_calc(t ,x, u, par, name2pos):
    # switch protein-dependent term - the concentration of active (inducer-bound) switch proteins divided by half-saturation constant
    p_s2_term = (x[name2pos['p_s2']] * par['I_s2']) / par['K_s2,s2']
    # switch protein regulation function
    F_s2 = par['F_s2,0'] + (1 - par['F_s2,0']) * (p_s2_term ** par['eta_s2,s2']) / (
                p_s2_term ** par['eta_s2,s2'] + 1)
    return jnp.array([
        F_s2,
        F_s2
        # ofp co-expressed with the switch gene from the same protein; this value will have no bearing on the ODE but is repeated for illustrative purposes
    ])

# ODE
def sas2_ode(F_calc,  # calculating the transcription regulation functions
                t, x,  # time, cell state
                u,  # controller input
                e, l,  # translation elongation rate, growth rate
                R,  # ribosome count in the cell, resource
                q_het, D,  # resource demands for synthetic genes, resource competition denominator
                par,  # system parameters
                name2pos  # name to position decoder
                ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, u, par, name2pos)

    # RETURN THE ODE
    return [# proteins
            (e / par['n_s2']) * (F[name2pos['F_s2']] * par['q_s2']/ D) * R - l * x[name2pos['p_s2']],
            (e / par['n_ofp2']) * (F[name2pos['F_ofp2']] * par['q_ofp2'] / D) * R - l * x[name2pos['p_ofp2']] - par['mu_ofp2'] * x[name2pos['p_ofp2']],
            # mature fluorescent proteins
            par['mu_ofp2']*x[name2pos['p_ofp2']] - l*x[name2pos['ofp2_mature']]
    ]

# specific terms of ODEs and cellular process rate definitions
# 1) effective mRNA concentrations for genes expressed from the same operon with some others
# 2) contribution to protein degradation flux from miscellaneous species (normally, mature proteins)
def sas2_specterms(x, par, name2pos):
    return (
        jnp.array([0.0]), # 1) the cell model would have mRNA concentration here, here this term is just for consistency
        0.0 # 2) mature ofp2 degradation - not considered, this term is just for consistency
    )

# CHEMICALLY INDUCED, CYBERCONTROLLED GENE -----------------------------------------------------------------------------
def cicc_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    # names of genes in the circuit
    genes = ['ta',  # transcription activation factor
             'b']  # burdensome controlled gene
    # names of miscellaneous species involved in the circuit (none)
    miscs = ['b_mature']  # mature burdensome protein
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_b']] will return the concentration of mRNA of the gene 'b'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['p_' + genes[i]] = 8 + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['q_' + genes[i]] = i  # resource demands (in q_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['q_' + gene] = 1.0  # resource demand (unitless)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # transcription regulation function
    default_par['K_ta,i'] = 100  # dissociation constant of the ta-inducer binding
    default_par['K_tai,b'] = 100  # dissociation constant of the ta-inducer complex from the DNA
    default_par['eta_tai,b'] = 2  # Hill coefficient of the ta protein binding to the DNA
    default_par['F_b,0'] = 0.1  # baseline expression of the burdensome gene

    # (fluorescent) mature burdensome protein parameters
    default_par['mu_b'] = np.log(2) / (13.6 / 60)  # sfGFP maturation time of 13.6 min
    default_par['n_b_mature'] = default_par['n_b']  # protein length - same as the freshly synthesised protein
    default_par['d_b_mature'] = default_par['d_b']  # mature ofp degradation rate - same as the freshly synthesised protein
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
    circuit_styles['colours']['ta'] = '#48d1ccff'
    circuit_styles['colours']['b'] = '#bb3385ff'
    circuit_styles['colours']['b_mature'] = '#ff6700ff'
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def cicc_F_calc(t ,x,
                u,  # controller input: external inducer concentration
                par, name2pos):
    F_ta = 1 # constitutive gene

    # burdensome gene expression is regulated by the ta protein
    tai_conc = x[name2pos['p_ta']] * u/(u+par['K_ta,i'])
    F_b = par['F_b,0'] + (1 - par['F_b,0']) * (tai_conc**par['eta_tai,b'])/(tai_conc**par['eta_tai,b']+par['K_tai,b']**par['eta_tai,b'])
    return jnp.array([F_ta,
            F_b])

# ODE
def cicc_ode(F_calc,  # calculating the transcription regulation functions
                t, x,  # time, cell state
                u,  # controller input
                e, l,  # translation elongation rate, growth rate
                R,  # ribosome count in the cell, resource
                q_het, D,  # resource demands for synthetic genes, resource competition denominator
                par,  # system parameters
                name2pos  # name to position decoder
                ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, u, par, name2pos)

    # RETURN THE ODE
    return [# proteins
            (e / par['n_ta']) * (F[name2pos['F_ta']] * par['q_ta'] / D) * R - l * x[name2pos['p_ta']],
            (e / par['n_b']) * (F[name2pos['F_b']] * par['q_b'] / D) * R - l * x[name2pos['p_b']] - par['mu_b'] * x[name2pos['p_b']],
            # mature fluorescent proteins
            par['mu_b']*x[name2pos['p_b']] - l*x[name2pos['b_mature']]
            ]

# specific terms of ODEs and cellular process rate definitions
# 1) effective mRNA concentrations for genes expressed from the same operon with some others
# 2) contribution to protein degradation flux from miscellaneous species (normally, mature proteins)
def cicc_specterms(x, par, name2pos):
    return (
        jnp.array([0.0]), # 1) the cell model would have mRNA concentration here, here this term is just for consistency
        0.0 # 2) mature ofp2 degradation - not considered, this term is just for consistency
    )
