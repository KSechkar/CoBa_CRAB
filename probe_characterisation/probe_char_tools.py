# SELFACT_AN_BIF.PY: Tools for probe characterisation

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
import sim_tools.genetic_modules as gms
import sim_tools.controllers as ctrls
import sim_tools.reference_switchers as refsws
import sim_tools.ode_solvers as odesols

# CALCULATING NORMALISED RESOURCE COMPETITION FACTORS ------------------------------------------------------------------
# find Q values for the two modules, given the results of observing them individually and in a pair
def Q_calc_both(mu_first, mu_second,                                            # maturation rates of the first and second output fluorescent proteins
                mature_ofp_ss_individual_first, l_ss_individual_first,          # individual measurements for the first module: output prot. conc., growth rate
                mature_ofp_ss_individual_second, l_ss_individual_second,        # individual measurements for the second module: output prot. conc., growth rate
                mature_ofp_ss_pair_first, mature_ofp_ss_pair_second, l_ss_pair  # pair measurements: output prot. concs., growth rate
                ):
    # get the expression rate ratio (pair-to-individual) for the first reporter in the pair
    exp_rate_ratio_first = (
            (l_ss_pair + mu_first) / (l_ss_individual_first + mu_first) *  # scaling factor due to maturation
            (mature_ofp_ss_pair_first / mature_ofp_ss_individual_first)  # mature protein concentration ratio (measurable)
    )
    # get the expression rate ratio (pair-to-individual) for the second reporter in the pair
    exp_rate_ratio_second = (
            (l_ss_pair + mu_second) / (l_ss_individual_second + mu_second) *  # scaling factor due to maturation
            (mature_ofp_ss_pair_second / mature_ofp_ss_individual_second)  # mature protein concentration ratio (measurable)
    )

    # get Q for the first module
    Q_first = (1 - exp_rate_ratio_second) / (exp_rate_ratio_first + exp_rate_ratio_second - 1)

    # get Q for the second module
    Q_second = (1 - exp_rate_ratio_first) / (exp_rate_ratio_first + exp_rate_ratio_second - 1)

    # return normalised resource competition factors
    return Q_first, Q_second

# find the Q value for an unknown module, given the results of observing it with a known constitutive reporter
def Q_calc_one(mu_constrep,                                                     # maturation rate of the constitutive reporter's output fluorescent protein
               mature_ofp_ss_individual_constrep, l_ss_individual_constrep,     # individual measurements for the const. reporter: output prot. conc., growth rate
               mature_ofp_ss_pair_constrep, l_ss_pair_constrep,                 # pair measurements for the const. reporter: output prot. concs., growth rate
               Q_constrep                                                       # normalised resource competition factor for the constitutive reporter
               ):
    # get the expression rate ratio (pair-to-individual) for the constitutive reporter
    exp_rate_ratio_constrep = (
            (l_ss_pair_constrep + mu_constrep) / (l_ss_individual_constrep + mu_constrep) *  # scaling factor due to maturation
            (mature_ofp_ss_pair_constrep / mature_ofp_ss_individual_constrep)  # mature protein concentration ratio (measurable)
    )

    # get Q for the unknown module
    Q_unknown = (1 + Q_constrep) * ((1 - exp_rate_ratio_constrep) / exp_rate_ratio_constrep)

    # return normalised resource competition factor for the unknown module
    return Q_unknown