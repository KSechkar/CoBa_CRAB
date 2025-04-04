# SELFACT_JOINTEXP.PY: Tools for predicting the behaviour of a two self-activating switches when they are expressed jointly

# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
from cell_model_case.cell_model import *


# FIND OUTPUT CORRESPONDING TO A GIVEN BURDEN EXERTED BY A SWITCH ------------------------------------------------------
def Q_sas_to_ofp_mature(Q_sas_query,        # burden for which the output is to be found
                        Q_sass_bifcurve,    # bifurcation curve points: burden
                        ofps_bifcurve       # bifurcation curve points: output
                        ):
    # for np.interp to work properly, Q_sass_bifcurve must be in ascending order
    if(Q_sass_bifcurve[-1]>=Q_sass_bifcurve[0]):
        return np.interp(Q_sas_query, Q_sass_bifcurve, ofps_bifcurve)
    else:
        return np.interp(Q_sas_query, np.flip(Q_sass_bifcurve), np.flip(ofps_bifcurve))


# FIND INTERSECTIONS OF BIFURCATION CURVES -----------------------------------------------------------------------------
# find the intersection of two bifurcation curves
def bifcurves_intersection(Q_sas1s_1,  # bifurcation curve for switch 1: burden imposed by itself
                           Q_sas2s_1,  # bifurcation curve for switch 1: burden experienced by it due to switch 2
                           Q_sas1s_2,  # bifurcation curve for switch 2: burden experienced by it due to switch 1
                           Q_sas2s_2,  # bifurcation curve for switch 2: burden imposed by itself
                           ):
    # initialise the storage of intersection points
    intersection_Q_sas1s = []
    intersection_Q_sas2s = []
    # for each pair of bifurcation curve segments, find if they intersect and where
    for i in range(0,len(Q_sas1s_1)-1):
        for j in range(0,len(Q_sas1s_2)-1):
            t, u, intersection_point = segments_intersection(Q_sas1s_1[i], Q_sas2s_1[i],
                                                             Q_sas1s_1[i + 1], Q_sas2s_1[i + 1],
                                                             Q_sas1s_2[j], Q_sas2s_2[j],
                                                             Q_sas1s_2[j + 1], Q_sas2s_2[j + 1])
            if((0<=t and t<=1) and (0<=u and u<=1)):
                intersection_Q_sas1s.append(intersection_point[0])
                intersection_Q_sas2s.append(intersection_point[1])

    # return the intersection point coordinates in numpy array format
    return np.array(intersection_Q_sas1s), np.array(intersection_Q_sas2s)


# find intersection of two line segements (actually exists if 0<=t<=1)
# based on https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
# warning: does not handle well the situation when segments overlap (i.e. have infinitely many intersections)
def segments_intersection(x11, y11,     # first point of the first segment
                          x12, y12,     # second point of the first segment
                          x21, y21,     # first point of the second segment
                          x22, y22      # second point of the second segment
                          ):
    # factor by which the 11-12 segment is multiplied to get the intersection point by adding the result to the 11 point
    t = np.divide((x11-x21)*(y21-y22)-(y11-y21)*(x21-x22),(x11-x12)*(y21-y22)-(y11-y12)*(x21-x22))
    # factor by which the 21-22 segment is multiplied to get the intersection point by adding the result to the 21 point
    u = -np.divide((x11-x12)*(y11-y21)-(y11-y12)*(x11-x21),(x11-x12)*(y21-y22)-(y11-y12)*(x21-x22))

    # return the factor t and the intersection point
    return t, u, (x11+t*(x12-x11), y11+t*(y12-y11))



# MAIN FUNCTION (FOR TESTING) ------------------------------------------------------------------------------------------
def main():
    intersect = segments_intersection(0, 0,
                                      1, 1,
                                      3, 3,
                                      2, 2)
    print(intersect)
    return

# MAIN CALL
if __name__ == '__main__':
    main()
