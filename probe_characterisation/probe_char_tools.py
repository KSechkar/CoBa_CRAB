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

import matplotlib as mpl, matplotlib.pyplot as plt
from pandas.tests.config.test_localization import test_get_locales_at_least_one

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

# INTERPOLATE A MAPPING FROM DATA --------------------------------------------------------------------------------------
# create an interpolation function for a mapping
def make_interpolator(mapping_data):
    # get the u and y values from the data
    u_data = jnp.array(mapping_data[0])
    y_data = jnp.array(mapping_data[1])
    Q_data = jnp.array(mapping_data[2])


    # create an interpolation function
    interpolator = functools.partial(interpolate_Q, u_data=u_data, y_data=y_data, Q_data=Q_data)

    # return the interpolation function
    return interpolator

# interpolate Q or Q' value for a (u, y_probe) pair from data
# @jax.jit
def interpolate_Q(u, y,
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

# return 1/distance if condition true , 0 otherwise
def one_div_dist_conditional(distance, condition):
    return jax.lax.select(pred = condition,
                          on_true = 1.0/distance,
                          on_false = jnp.zeros_like(distance))


# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    u_data=jnp.array([[0, 0, 0],
                      [1, 1, 2],
                      [2, 2, 2]], dtype=jnp.float32)

    y_data = jnp.array([[0, 0, 1],
                        [1, 2, 2],
                        [4, 4, 3]], dtype=jnp.float32).T

    Q_data = jnp.array([[0, 4, 8],
                        [6, 12, 24],
                        [14, 28, 56]], dtype=jnp.float32).T

    # define the u and y values to estimate Q and Q' for
    u_vals_est = np.linspace(np.min(u_data), np.max(u_data), 5)
    y_vals_est = np.linspace(np.min(y_data), np.max(y_data), 5)

    # make a mesh grid
    us_mesh_est = np.zeros((len(u_vals_est), len(y_vals_est)))
    y_mesh_est = np.zeros((len(u_vals_est), len(y_vals_est)))
    for i in range(0, len(u_vals_est)):
        for j in range(0, len(y_vals_est)):
            us_mesh_est[i, j] = u_vals_est[i]
            y_mesh_est[i, j] = y_vals_est[j]

    print(us_mesh_est)
    print(y_mesh_est)

    # create interpolation functions
    interpolator_Q = make_interpolator(np.stack((u_data.ravel(), y_data.ravel(), Q_data.ravel()),axis=0))
    interpolator_Qdash = make_interpolator(np.stack((u_data.ravel(), y_data.ravel(), Q_data.ravel()),axis=0))

    # get the estimates
    Q_est = np.zeros_like(us_mesh_est)
    Qdash_est = np.zeros_like(us_mesh_est)
    for i in range(0, len(us_mesh_est)):
        for j in range(0, len(y_mesh_est)):
            Q_est[i, j] = interpolator_Q(u=us_mesh_est[i, j], y=y_mesh_est[i, j])
            Qdash_est[i, j] = interpolator_Qdash(u=us_mesh_est[i, j], y=y_mesh_est[i, j])

    # FIGURE
    u_y_to_Q_fig = plt.figure(2)
    plt.clf()
    u_y_to_Q_ax = u_y_to_Q_fig.add_subplot(111, projection='3d')

    # # make a 3D plot of experimentally measured Q values
    u_y_to_Q_ax.scatter(
        u_data,
        y_data,
        Q_data,
        c='cyan', label='Measured Q values')

    # make a 3D plot of estimated Q values
    u_y_to_Q_ax.plot_surface(
        us_mesh_est,
        y_mesh_est,
        Q_est,
        cmap=mpl.colormaps['plasma'], label='Measured Q values')

    # format the plot
    u_y_to_Q_ax.set_title("(u, y_probe) -> Q_probe")
    u_y_to_Q_ax.set_xlabel('u')
    u_y_to_Q_ax.set_ylabel('y_probe')
    u_y_to_Q_ax.set_zlabel('Q_probe')

    # show plot
    plt.show()



    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()