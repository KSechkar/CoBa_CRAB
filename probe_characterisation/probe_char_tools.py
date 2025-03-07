# SELFACT_AN_BIF.PY: Tools for probe characterisation

# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import scipy as sp
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

import matplotlib as mpl, matplotlib.pyplot as plt
from pandas.tests.config.test_localization import test_get_locales_at_least_one

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
from sim_tools.cell_model import *
import sim_tools.cell_genetic_modules as gms
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

# INTERPOLATE A MAPPING FROM DATA --------------------------------------------------------------------------------------
# CUSTOM (works badly)
# return an interpolation function for a mapping from data
def make_custom_interpolator(mapping_data):
    # unpack the experimental data for mapping reconstruction
    u_data = jnp.array(mapping_data[0])
    y_data = jnp.array(mapping_data[1])
    Q_data = jnp.array(mapping_data[2])


    # create an interpolation function
    interpolator = functools.partial(interpolate_Q_custom, u_data=u_data, y_data=y_data, Q_data=Q_data)

    # return the interpolation function
    return interpolator

# custom mapping function: interpolate Q or Q' value for a (u, y_probe) pair from data
@jax.jit
def interpolate_Q_custom(u, y,
                        u_data, y_data, Q_data):
    # get normalising factors for distances in u and y
    u_norm = jnp.max(u_data) - jnp.min(u_data)
    y_norm = jnp.max(y_data) - jnp.min(y_data)

    # get distance to each query point
    distance = jnp.sqrt(jnp.square((u_data - u)/u_norm) + jnp.square((y_data - y)/y_norm))

    # make array with True values for u indices left of the query point
    u_below_query = u_data <= u
    u_above_query = u_data > u
    # make array with True values for y indices below the query point
    y_below_query = y_data <= y
    y_above_query = y_data > y

    # get 1/distance to the nearest datapoint with u below and y below the query point
    one_div_dist_ubelow_ybelow_all = one_div_dist_conditional(distance, jnp.multiply(u_below_query,y_below_query))
    i_ubelow_ybelow = jnp.argmax(one_div_dist_ubelow_ybelow_all)
    one_div_dist_ubelow_ybelow = one_div_dist_ubelow_ybelow_all[i_ubelow_ybelow]
    Q_ubelow_ybelow = Q_data[i_ubelow_ybelow]

    # get 1/distance to of the nearest datapoint with u above and y below the query point
    one_div_dist_uabove_ybelow_all = one_div_dist_conditional(distance, jnp.multiply(u_above_query, y_below_query))
    i_uabove_ybelow = jnp.argmax(one_div_dist_uabove_ybelow_all)
    one_div_dist_uabove_ybelow = one_div_dist_uabove_ybelow_all[i_uabove_ybelow]
    Q_uabove_ybelow = Q_data[i_uabove_ybelow]

    # get 1/distance to of the nearest datapoint with u below and y above the query point
    one_div_dist_ubelow_yabove_all = one_div_dist_conditional(distance, jnp.multiply(u_below_query, y_above_query))
    i_ubelow_yabove = jnp.argmax(one_div_dist_ubelow_yabove_all)
    one_div_dist_ubelow_yabove = one_div_dist_ubelow_yabove_all[i_ubelow_yabove]
    Q_ubelow_yabove = Q_data[i_ubelow_yabove]

    # get 1/distance to of the nearest datapoint with u above and y above the query point
    one_div_dist_uabove_yabove_all = one_div_dist_conditional(distance, jnp.multiply(u_above_query, y_above_query))
    i_uabove_yabove = jnp.argmax(one_div_dist_uabove_yabove_all)
    one_div_dist_uabove_yabove = one_div_dist_uabove_yabove_all[i_uabove_yabove]
    Q_uabove_yabove = Q_data[i_uabove_yabove]

    # otherwise, get an inverse-distance-weighted average of the Q values of the four nearest datapoints
    Q_interp_numerator = (one_div_dist_ubelow_ybelow * Q_ubelow_ybelow +
                          one_div_dist_uabove_ybelow * Q_uabove_ybelow +
                          one_div_dist_ubelow_yabove * Q_ubelow_yabove +
                          one_div_dist_uabove_yabove * Q_uabove_yabove)
    Q_interp_denominator = (one_div_dist_ubelow_ybelow +
                            one_div_dist_uabove_ybelow +
                            one_div_dist_ubelow_yabove +
                            one_div_dist_uabove_yabove)
    Q_interp = Q_interp_numerator / Q_interp_denominator

    # if the query point is exactly on a datapoint, return the Q value for that datapoint instea
    Q = jax.lax.select(pred=(distance[i_ubelow_ybelow] == 0.0),
                       on_true=Q_ubelow_ybelow,
                       on_false=Q_interp)

    return Q

#  for custom mapping: return 1/distance if condition true , 0 otherwise
def one_div_dist_conditional(distance, condition):
    return jax.lax.select(pred = condition,
                          on_true = 1.0/distance,
                          on_false = jnp.zeros_like(distance))


# SCIPY RBF
# return an interpolation function for a mapping from data
def make_interpolator_sp_rbf(mapping_data,  # experimental data for mapping reconstruction
                             rbf_func='cubic',  # radial basis function to use for interpolation
                             normalise_u_and_y=True  # whether to normalise u and y values to their range (as u and y scales are different)
                             ):
    # unpack the experimental data for mapping reconstruction
    u_data = np.array(mapping_data[0])
    y_data = np.array(mapping_data[1])
    Q_data = np.array(mapping_data[2])

    # get normalising factors for distances in u and y
    if(normalise_u_and_y):
        u_norm = np.max(u_data) - np.min(u_data)
        y_norm = np.max(y_data) - np.min(y_data)
    else:
        u_norm = 1.0
        y_norm = 1.0

    # create a scipy interpolator
    sp_rbf_interpolator = sp.interpolate.Rbf(u_data / u_norm, y_data / y_norm,
                                             Q_data,
                                             function=rbf_func)

    # include normalisation if needed, clip Q to be non-negative
    interpolator = lambda u, y: max(sp_rbf_interpolator(u/u_norm, y/y_norm),0.0)

    return interpolator

# SCIPY LINEAR ND
# return an interpolation function for a mapping from data
def make_interpolator_sp_lnd(mapping_data,  # experimental data for mapping reconstruction
                             normalise_u_and_y=True  # whether to normalise u and y values to their range (as u and y scales are different)
                             ):
    # unpack the experimental data for mapping reconstruction
    u_data = np.array(mapping_data[0])
    y_data = np.array(mapping_data[1])
    Q_data = np.array(mapping_data[2])

    # get normalising factors for distances in u and y
    if(normalise_u_and_y):
        u_norm = np.max(u_data) - np.min(u_data)
        y_norm = np.max(y_data) - np.min(y_data)
    else:
        u_norm = 1.0
        y_norm = 1.0

    # create a scipy linear ND interpolator
    sp_lnd_interpolator = sp.interpolate.LinearNDInterpolator(list(zip(u_data, y_data)),
                                                              Q_data,
                                                              rescale=normalise_u_and_y,
                                                              fill_value=0.0)

    # include normalisation if needed, clip Q to be non-negative
    interpolator = lambda u, y: max(sp_lnd_interpolator(u, y),0.0)

    return interpolator

# SCIPY CLOUGH-TOCHER
# return an interpolation function for a mapping from data
def make_interpolator_sp_clt(mapping_data,  # experimental data for mapping reconstruction
                             normalise_u_and_y=True
                             # whether to normalise u and y values to their range (as u and y scales are different)
                             ):
    # unpack the experimental data for mapping reconstruction
    u_data = np.array(mapping_data[0])
    y_data = np.array(mapping_data[1])
    Q_data = np.array(mapping_data[2])

    # create a scipy Clough-Tocher interpolator
    sp_clt_interpolator = sp.interpolate.CloughTocher2DInterpolator(list(zip(u_data, y_data)),
                                                                    Q_data,
                                                                    rescale=normalise_u_and_y,
                                                                    fill_value=0.0)

    # clip Q to be non-negative
    interpolator = lambda u, y: max(sp_clt_interpolator(u, y),0.0)

    return interpolator


# DOUBLE-HILL FUNCTION FITTING
# return an interpolation function for a mapping from data
def make_interpolator_dhf(mapping_data,  # experimental data for mapping reconstruction
                          tol=1e-6  # tolerance for fitting the double-Hill function
                          ):
    # unpack the experimental data for mapping reconstruction
    u_data = np.array(mapping_data[0])
    y_data = np.array(mapping_data[1])
    Q_data = np.array(mapping_data[2])

    # make an initial guess for the parameters
    Y_init = np.max(y_data) # overall maximum output - set to the maximum measured y value
    F0_Q_init = 0.5        # baseline Hill function value for Q=0 - set to 0.5 as we don't know if it's Q (increasing y) or Q' (decreasing y)
    K_Q_init = np.mean(Q_data)         # Half-saturation constant for Q - set to the mean of all measured Q values
    eta_Q_init = 1.0       # Hill coefficient for Q - set to 1.0 as we don't know the cooperativity
    F0_u_init = 0.0        # baseline Hill function value for u=0 - set to 0.0, assuming no leaky expression
    K_u_init = np.mean(u_data)         # Half-saturation constant for u - set to the mean of all measured u values
    eta_u_init = 1.0       # Hill coefficient for u - set to 1.0 as we don't know the cooperativity
    params_inits = np.array([Y_init, F0_Q_init, K_Q_init, eta_Q_init, F0_u_init, K_u_init, eta_u_init])

    # define the constraints for the parameters
    Y_const = (0.0, np.inf) # overall maximum output - must be non-negative
    F0_Q_const = (0.0, 1.0) # baseline Hill function value for Q=0 - must be between 0 and 1
    K_Q_const = (1e-6, np.inf) # Half-saturation constant for Q - must be non-negative, but made non-zero to avoid division by zero
    eta_Q_const = (0.0, np.inf) # Hill coefficient for Q - must be non-negative
    F0_u_const = (0.0, 1.0) # baseline Hill function value for u=0 - must be between 0 and 1
    K_u_const = (1e-6, np.inf) # Half-saturation constant for u - must be non-negative, but made non-zero to avoid division by zero
    eta_u_const = (0.0, np.inf) # Hill coefficient for u - must be non-negative
    params_consts = (Y_const, F0_Q_const, K_Q_const, eta_Q_const, F0_u_const, K_u_const, eta_u_const)

    # create a cost function for fitting the double-Hill function to the data
    cost_func_sse = lambda params: double_hill_sse(u_data=u_data,
                                                   y_data=y_data,
                                                   Q_data=Q_data,
                                                   params=params)

    # create an optimiser for fitting
    fitter = sp.optimize.minimize(fun=cost_func_sse,
                                  x0=params_inits,
                                  method='COBYLA',
                                  bounds=params_consts,
                                  tol=1e-12
                                  )


    return 0, fitter.x

# function solving the double-Hill equation for Q for given u and y
def solve_double_hill(u, y, params):
    # UNPACK PARAMETERS
    # overall maximum output
    Y = params[0]
    # Hill function for Q
    F0_Q = params[1]    # baseline Hill function value for Q=0
    K_Q = params[2]     # Half-saturation constant for Q
    eta_Q = params[3]   # Hill coefficient for Q
    # Hill function for u
    F0_u = params[4]    # baseline Hill function value for u=0
    K_u = params[5]     # Half-saturation constant for u
    eta_u = params[6]   # Hill coefficient for u

    # calculate the Hill function for u
    Hill_u = (F0_u + (1 - F0_u) * ((u/K_u) ** eta_u)) / (1 + (u/K_u) ** eta_u)

    return Q

# function returning a sum of squared errors for double-Hill predictionsw vs experimental data
def double_hill_sse(u_data, y_data, Q_data, params):
    ys = double_hill(us=u_data, Qs=Q_data, params=params)

    return np.sum(np.square(y_data - ys))

# function yielding double-Hill prediction for y values based on u and Q values
def double_hill(us, Qs, params):
    # UNPACK PARAMETERS
    # overall maximum output
    Y = params[0]
    # Hill function for Q
    F0_Q = params[1]    # baseline Hill function value for Q=0
    K_Q = params[2]     # Half-saturation constant for Q
    eta_Q = params[3]   # Hill coefficient for Q
    # Hill function for u
    F0_u = params[4]    # baseline Hill function value for u=0
    K_u = params[5]     # Half-saturation constant for u
    eta_u = params[6]   # Hill coefficient for u

    # calculate the Hill functions
    Hills_Q = (F0_Q + (1 - F0_Q) * ((Qs/K_Q) ** eta_Q)) / (1 + (Qs/K_Q) ** eta_Q)
    Hills_u = (F0_u + (1 - F0_u) * ((us/K_u) ** eta_u)) / (1 + (us/K_u) ** eta_u)
    ys = Y * Hills_Q * Hills_u

    return ys




# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    Q_probe_data = np.load('Q_probe_data.npy')

    params=make_interpolator_dhf(Q_probe_data)
    
    # unpack the experimental data for mapping reconstruction
    us = np.array(Q_probe_data[0])
    y_probes = np.array(Q_probe_data[1])
    Q_probes = np.array(Q_probe_data[2])

    # create a double-Hill interpolator
    interpolator, params = make_interpolator_dhf(Q_probe_data)
    
    y_dhf=double_hill(Q_probes,us,params)
    
    # plot the interpolated mapping
    u_Q_to_y_fig=plt.figure(1)
    plt.clf()
    u_Q_to_y_ax = u_Q_to_y_fig.add_subplot(111, projection='3d')

    # make a 3D plot of estimated y values
    u_Q_to_y_ax.scatter(
        us,
        Q_probes,
        y_dhf,
        color='red',
        label='Est y')
    # make a scatter plot of the measured y values
    u_Q_to_y_ax.scatter(us,
                        Q_probes,
                        y_probes,
                        color='cyan',
                        label='Real y')
    
    

    # format the plot
    u_Q_to_y_ax.set_title("(u, Q\'_probe) -> Q_probe")
    u_Q_to_y_ax.set_xlabel('u')
    u_Q_to_y_ax.set_ylabel('Q\'_probe')
    u_Q_to_y_ax.set_zlabel('Q_probe')

    # show plot
    plt.show()
    



    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()