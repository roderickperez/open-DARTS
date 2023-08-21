import abc
import numpy as np
import pickle
import atexit

from darts.engines import conn_mesh, timer_node, ms_well_vector


class ReservoirBase:
    wells: ms_well_vector = []
    perforations: list = []

    def __init__(self, timer: timer_node, cache: bool = False):
        # Initialize timer for initialization and caching
        self.timer = timer.node["initialization"]
        self.cache = cache

        # is used on destruction to save cache data
        if self.cache:
            self.created_itors = []
            atexit.register(self.write_cache)

    def init_reservoir(self) -> (conn_mesh, ms_well_vector):
        mesh = self.discretize()
        wells = self.init_wells(mesh)
        return mesh, wells

    @abc.abstractmethod
    def discretize(self) -> conn_mesh:
        pass

    @abc.abstractmethod
    def set_boundary_volume(self, mesh: conn_mesh):
        pass

    @abc.abstractmethod
    def add_well(self, name):
        pass

    @abc.abstractmethod
    def add_perforations(self, mesh):
        pass

    def get_well(self, well_name: str):
        """
        Find well by name

        :param well_name: Well name
        :returns: :class:`ms_well` object
        """
        for w in self.wells:
            if w.name == well_name:
                return w

    def init_wells(self, mesh: conn_mesh) -> ms_well_vector:
        for w in self.wells:
            assert (len(w.perforations) > 0), "Well %s does not perforate any active reservoir blocks" % w.name
        mesh.add_wells(ms_well_vector(self.wells))

        mesh.reverse_and_sort()
        mesh.init_grav_coef()
        return self.wells

    @abc.abstractmethod
    def output_to_vtk(self, output_directory, output_filename, property_data, ith_step):
        pass

    def write_cache(self):
        return

    def __del__(self):
        # first write cache
        if self.cache:
            self.write_cache()
        # Now destroy all objects in physics
        for name in list(vars(self).keys()):
            delattr(self, name)
