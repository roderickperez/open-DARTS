from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import value_vector
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

from math import fabs

class Model(DartsModel):
    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        cache = True
        permx = load_single_keyword('SPE10_input.txt', 'PERMX', cache=cache)
        permz = load_single_keyword('SPE10_input.txt', 'PERMZ', cache=cache)
        poro = load_single_keyword('SPE10_input.txt', 'PORO', cache=cache)
        depth = load_single_keyword('SPE10_input.txt', 'DEPTH', cache=cache)

        nz = 85
        nb = 60 * 220 * nz
        self.reservoir = StructReservoir(self.timer, nx=60, ny=220, nz=nz, dx=6.096, dy=3.048, dz=0.6096,
                                         permx=permx[:nb], permy=permx[:nb], permz=permz[:nb], poro=poro[:nb],
                                         depth=depth[:nb])
        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        for k in range(self.reservoir.nz):
            self.reservoir.add_perforation("I1", cell_index=(30, 110, k+1), multi_segment=False)

        self.reservoir.add_well("P1")
        for k in range(self.reservoir.nz):
            self.reservoir.add_perforation("P1", cell_index=(1, 1, k+1), multi_segment=False)

        self.reservoir.add_well("P2")
        for k in range(self.reservoir.nz):
            self.reservoir.add_perforation("P2", cell_index=(60, 1, k+1), multi_segment=False)

        self.reservoir.add_well("P3")
        for k in range(self.reservoir.nz):
            self.reservoir.add_perforation("P3", cell_index=(1, 220, k+1), multi_segment=False)

        self.reservoir.add_well("P4")
        for k in range(self.reservoir.nz):
            self.reservoir.add_perforation("P4", cell_index=(60, 220, k+1), multi_segment=False)


    def set_physics(self):
        """Physical properties"""
        zero = 1e-13
        components = ["w", "o"]
        phases = ["wat", "oil"]

        self.ini = np.array([0.2357])
        self.inj_comp = np.array([1 - 1e-6])
        self.inj_temp = 300.

        property_container = ModelProperties(phases_name=phases, components_name=components, min_z=zero/10)

        property_container.density_ev = dict([('wat', DensityBasic(compr=4.5e-5, dens0=1025.18)),
                                              ('oil', DensityBasic(compr=2e-5, dens0=848.979))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                ('oil', ConstFunc(3))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("gas", 0.2, 0.2)),
                                               ('oil', PhaseRelPerm("oil", 0.2, 0.2))])

        #property_container.rock_compr_ev =

        # create physics
        self.physics = Compositional(components, phases, self.timer,
                                     n_points=400, min_p=0, max_p=1000, min_z=zero, max_z=1-zero,
                                     state_spec=Compositional.StateSpecification.P)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 413.6854,
                              self.physics.vars[1]: self.ini[0],
                              }
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # should be 794.9365 m3/day
                self.physics.set_well_controls(wctrl=w.control, is_inj=True, control_type=well_control_iface.MOLAR_RATE,
                                               target=8e4, phase_name='wat', inj_composition=self.inj_comp, inj_temp=self.inj_temp)
                self.physics.set_well_controls(wctrl=w.constraint, is_inj=True, control_type=well_control_iface.BHP,
                                               target=689.4757, inj_composition=self.inj_comp, inj_temp=self.inj_temp)
            else:
                self.physics.set_well_controls(wctrl=w.control, is_inj=False, control_type=well_control_iface.BHP,
                                               target=275.7903)


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z, temperature=1.)

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = np.array([0, 1], dtype=np.intp)


        for j in self.ph:
            # molar weight of mixture
            M = np.sum(self.x[j, :] * self.Mw)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.nu = zc
        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        return

    def evaluate_at_cond(self, pressure, zc):
        self.sat[:] = 0

        ph = np.array([0, 1], dtype=np.intp)

        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m
