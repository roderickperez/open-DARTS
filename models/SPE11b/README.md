# SPE11b Carbon Capture and Storage Benchmark
The SPE11 comparative solution project (CSP) aims to provide a reference case for the development of numerical simulation of GCS and offers a baseline for simulation of CO2 storage in aquifers. 
The reservoir is a heterogeneous reservoir storage complex reminicent of the Norwegian continental shelf. 
Three versions of the benchmark are presented in the CSP. The second, 11b, is a 2D model at reservoir scale and conditions. 
The SPE11 CSP explicitly specifies all reservoir and fluid properties in Nordbotten et al. (2024). 
This repository contains a model implementation for the SPE11b utilizing the Delft Advanced Research Terra Simulator (DARTS) of Delft University of Technology. 

## Running the Simulation

Simulation parameters are configured using the `model_specs` list, where each entry is a dictionary describing a specific realization. 
The reporting grid of the SPE11b corresponds to a grid block of 10m by 10m and thus contains approx. 100K grid blocks.  

```python
nx = 840
nz = 120
model_specs = [
    {
        'check_rates': True,
        'temperature': 273.15 + 40,
        '1000years': False,
        'RHS': True,
        'components': ['CO2', 'H2O'],
        'inj_stream': [1 - 1e-10, 283.15],
        'nx': nx,
        'nz': nz,
        'dispersion': False,
        'output_dir': None,
        'post_process': None,
        'gpu_device': False
    }
]
```

| Option         | Description                                                                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `check_rates`  | If `True`, checks mass in place before and after each timestep to verify injected mass.                                                                      |
| `temperature`  | Temperature in Kelvin; if `None`, the model runs isothermally.                                                                                               |
| `1000years`    | If `True`, introduces a 1000-year initialization period before the main simulation.                                                                          |
| `RHS`          | If `True`, uses a right-hand side correction to introduce mass to the well block. If `False`, uses `dartsmodel.physics.set_well_controls()` with mass rates. |
| `components`   | List of components in the simulation (e.g., `['CO2', 'H2O']`).                                                                                               |
| `inj_stream`   | Injection stream: list with component compositions (up to `nc-1` components) and the final entry as temperature (in Kelvin).                                 |
| `nx`           | Horizontal resolution of the model grid.                                                                                                                     |
| `nz`           | Vertical resolution of the model grid.                                                                                                                       |
| `dispersion`   | If `True`, enables dispersion with dispersivity set to 10; otherwise, dispersion is excluded.                                                                |
| `output_dir`   | Output directory for simulation results. If `None`, a directory is auto-generated based on model specs.                                                      |
| `post_process` | If a string is provided, post-processes previously saved data in that folder instead of running the simulation.                                              |
| `gpu_device`   | Runs the model on GPU if `True`; otherwise, on CPU.                                                                                                          |



## Model physics

### Dispersion 

### Thermodynamics 
Thermodynamics properties are provided by the DARTS-flash python module. 
A negative flash procedure with successive substitution is employed for resolving thermodynamic equilibrium
calculations (Michelsen, 1982; Whitson and Michelsen, 1989). The fugacities of the vapor phase are evaluated
using a cubic equation of state (Peng and Robinson, 1976) and the fugacities of the water phase are calculated
using an activity model based on Henry’s constants (Ziabakhsh-Ganji and Kooi, 2012). The property correlations for SPE11
and open-DARTS are presented in the following table. 

![properties](properties.PNG "properties")

In the model, these property correlations are specified in the `property_containers` that are defined per region in the `compositional` physics class of the `DarstsModel`. 

```python
        for i, (region, corey_params) in enumerate(corey.items()):
            diff_w = 1e-9 * 86400
            diff_g = 2e-8 * 86400
            property_container = PropertyContainer(components_name=self.components, phases_name=phases, Mw=comp_data.Mw,
                                                   min_z=zero / 10, temperature=temperature)

            # property_container.flash_ev = ConstantK(nc=2, ki=[0.001, 100])
            property_container.flash_ev = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])
            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                                  ('Aq', Garcia2001(self.components)), ])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(self.components)), ])
            property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(nc) * diff_g)),
                                                    ('Aq', ConstFunc(np.ones(nc) * diff_w))])
            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                   ('Aq', EoSEnthalpy(eos=aq)), ])
            property_container.conductivity_ev = dict([('V', ConstFunc(8.4)),
                                                       ('Aq', ConstFunc(170.)),])
            property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params, 'V')),
                                                   ('Aq', ModBrooksCorey(corey_params, 'Aq'))])
            property_container.capillary_pressure_ev = ModCapillaryPressure(corey_params)

            self.physics.add_property_region(property_container, i)
```


