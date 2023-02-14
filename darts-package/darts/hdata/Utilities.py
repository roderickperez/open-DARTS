import numpy as np
import pandas as pd

def sodm_bhp(well_top_TVD, p_gradient=0.0135):
    """
    Calculates the maximum pore pressure allowed by SodM regulations at the TVD
    of the well according to guidance:
        https://www.sodm.nl/binaries/staatstoezicht-op-de-mijnen/documenten/publicaties/2013/11/23/protocol-bepaling-maximale-injectiedrukken-bij-aardwarmtewinning/Protocol_injectiedrukken_bij_aardwarmte.pdf

    inputs:
        well_TVD_top_reservoir : True Vertical Depth at the top part of the reservoir (meters)
        p_gradient             : pressure gradient dictated by SodM (MPa/meter)

    output:
        max_bhp     : maximum bottom hole pressure allowed (MPa)

    """

    max_bhp_MPa = well_top_TVD * p_gradient  # MPa

    return (max_bhp_MPa)


def well_tvds(reservoir):
    """

    :param well_depth_array: input depth array for well
    :return: top and bottom TVD to be used in SodM and economic calculations
    """
    well_names = []
    well_top_tvd = []
    well_bottom_tvd = []
    well_mid_tvd = []

    for well in reservoir.wells:
    #     if 'E' in well.name:
    #         pass
    #     else:
        well_names.append(well.name)
        perf_top = well.perforations[0]
        perf_bottom = well.perforations[-1]
        well_top_tvd.append(reservoir.mesh.depth[perf_top[1]])
        well_bottom_tvd.append(reservoir.mesh.depth[perf_bottom[1]])
        well_mid_tvd.append(reservoir.mesh.depth[perf_top[1]] + (reservoir.mesh.depth[perf_bottom[1]] - reservoir.mesh.depth[perf_top[1]])/2)

    return(well_names, well_top_tvd, well_mid_tvd, well_bottom_tvd)

def get_well_distances(reservoir, demand1 = [94, 65], demand2 = [125, 170], dx=50, dy=50, verbose=False):

    dist1 = []
    dist2 = []
    well_names = []
    for well in reservoir.wells:
        well_names.append(well.name)
        # get well index on first perforation
        local_idx = well.perforations[0][1]

        # convert to global index
        global_idx = reservoir.discretizer.local_to_global[local_idx]

        # reshape cell array to 3D
        cell_array = np.arange(0,reservoir.n).reshape(reservoir.nz, reservoir.ny, reservoir.nx)

        # get 3D coordinates of well index
        coords = np.argwhere(cell_array == global_idx)

        # assign coordinates
        k, j, i = coords[0][0], coords[0][1]+1, coords[0][2]+1

        # compute distance to demand positions 1 and 2
        dist1.append(round(np.sqrt((demand1[0] * dx - i * dx) ** 2 + (demand1[1] * dy - j * dy) ** 2), 2))
        dist2.append(round(np.sqrt((demand2[0] * dx - i * dx) ** 2 + (demand2[1] * dy - j * dy) ** 2), 2))
        if verbose:
            print(well.name,i,j)

    # find the minimum distance for the well to any demand point
    well_demand_dist = np.stack((dist1, dist2)).min(axis=0)

    # compile a dictionary with well names and distance
    well_demand_dist_dict = dict(zip(well_names, well_demand_dist))
    return(well_demand_dist_dict)