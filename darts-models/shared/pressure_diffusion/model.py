from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import value_vector, sim_params
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

class Model(CICDModel):
    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        self.nx = self.ny = 501
        dx = 300025.1762 / self.nx
        dy = 295118.1414 / self.ny
        perm = 10
        
        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=1, dx=dx, dy=dy, dz=155.0708745,
                                         permx=perm, permy=perm, permz=perm, poro=0.342649063, hcap=0, rcond=0,
                                         depth=100, cache=True)
        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", cell_index=(int(self.nx*0.263894263), int(self.ny*0.546976327), 1))

    def set_physics(self):
        """Physical properties"""
        zero = 1e-10
        components = ["w", "o"]
        phases = ["wat"]

        self.inj = value_vector([1 - zero])
        self.ini = value_vector([1 - zero])

        property_container = ModelProperties(phases_name=phases, components_name=components, min_z=zero)

        property_container.density_ev = dict([('wat', DensityBasic(compr=0.000131147, dens0=1014))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.3))])

        # create physics
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=400, min_p=0, max_p=1000, min_z=zero/100, max_z=1-zero/100, cache=True)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 246.8734336,
                              self.physics.vars[1]: self.ini[0],
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        w = self.reservoir.wells[0]
        self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                                       is_inj=True, target=4.9e9/365, phase_name='wat', inj_composition=self.inj)


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
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = np.array([0], dtype=np.intp)

        j = 0
        # molar weight of mixture
        M = self.Mw[j]
        self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure)  # output in [kg/m3]
        self.dens_m[j] = self.dens[j] / M
        self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.sat[j] = 1.0
        self.kr[j] = 1.0
        self.pc[j] = 0

        return

    def evaluate_at_cond(self, pressure, zc):
        self.sat[:] = 0

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m
