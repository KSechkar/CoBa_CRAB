# SELFACT_JOINTEXP.PY: Tools for predicting the behaviour of a two self-activating switches when they are expressed jointly

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

# FIND OUTPUT CORRESPONDING TO A GIVEN BURDEN EXERTED BY A SWITCH ------------------------------------------------------
def q_sas_to_ofp_mature(q_sas_query,    # burden for which the output is to be found
                        q_sass_bifcurve, # bifurcation curve points: burden
                        ofps_bifcurve   # bifurcation curve points: output
                        ):
    return np.interp(q_sas_query, q_sass_bifcurve, ofps_bifcurve)


# FIND THE INTERSECTION OF LINES DEFINED BY TWO SEGMENTS ---------------------------------------------------------------
# based on https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
def segments_intersection(x11, y11,     # first point of the first segment
                          x12, y12,     # second point of the first segment
                          x21, y21,     # first point of the second segment
                          x22, y22      # second point of the second segment
                          ):
    # factor by which the 11-12 segment is multiplied to get the intersection point by adding the result to the 11 point
    t = ((x11-x21)*(y21-y22)-(y11-y21)*(x21-x22))/((x11-x12)*(y21-y22)-(y11-y12)*(x21-x22))

    # return the factor t and the intersection point
    return t, (x11+t*(x12-x11), y11+t*(y12-y11))



# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    intersect = segments_intersection(0, 0,
                                      1, 1,
                                      0, 1,
                                      1, 2)
    print(intersect)
    return

# MAIN CALL
if __name__ == '__main__':
    main()
