import abc

import numpy as np
from darts.engines import conn_mesh


class ReservoirBase:
    nb: int

    def init_reservoir(self) -> conn_mesh:
        return mesh

    @abc.abstractmethod
    def discretize(self):
        pass

    def define_boundary(self):
        pass

    def add_well(self):
        pass

    def output_vtk(self):
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
