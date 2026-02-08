from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


m = Model()
time = 20*365

m.init()
m.set_output(verbose = False)
# m.export_pro_vtk()

for i in range(20):
    m.run(365)
# m.print_timers()
# m.print_stat()
# m.export_pro_vtk()
m.output.output_to_vtk(output_properties = m.physics.vars + m.output.properties)


td = pd.DataFrame.from_dict(m.output.store_well_time_data())
td.to_pickle("darts_time_data.pkl")
writer = pd.ExcelWriter('time_data.xlsx')
td.to_excel(writer, sheet_name='Sheet1')
writer.close()

string = 'well_PRD_BHT'
ax1 = td.plot(x='time', y=[string])

ax1.plot([0, time], [348, 348])
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Days', fontsize=14)
ax1.legend(['temp', 'limit'], fontsize=14)
ax1.set(xlim=(0, time), ylim=(346, 351))
plt.grid()
plt.savefig('out.png')


