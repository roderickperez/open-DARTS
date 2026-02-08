import numpy as np

#%% Wilson correlation (for pressures below 500 psia or 34.5 bar)
system_pressure_bar = 5.5
system_temperature_C = 25


# CO2 props
Pci_CO2 = 73.75 * 14.5038   # Convert bar to psi
Tci_CO2 = ((304.10 - 273.15) * 1.8 + 32) + 459.67   # Convert K to R
wi_CO2 = 0.239

# H2O props
Pci_H2O = 220.50 * 14.5038   # Convert bar to psi
Tci_H2O = ((647.14 - 273.15) * 1.8 + 32) + 459.67   # Convert K to R
wi_H2O = 0.328

Pci = np.array([Pci_CO2, Pci_H2O])
Tci = np.array([Tci_CO2, Tci_H2O])
wi = np.array([wi_CO2, wi_H2O])

system_pressure_psi = system_pressure_bar * 14.5038
system_temperature_R = (system_temperature_C * 1.8 + 32) + 459.67

P = system_pressure_psi
T = system_temperature_R

Ki = Pci/P * np.exp(5.37 * (1 + wi) * (1 - Tci/T))

print('K_CO2 = ' + str(Ki[0]) + ', K_H2O = ' + str(Ki[1]))
