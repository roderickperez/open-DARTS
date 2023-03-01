import sys
sys.path.append('../physics_sup/')

from model import Model
import pandas as pd


m = Model()

m.init()
m.export_pro_vtk()
m.run(365*50)
m.print_timers()
m.print_stat()
m.export_pro_vtk()


time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
time_data.to_pickle("darts_time_data.pkl")
writer = pd.ExcelWriter('time_data.xlsx')
time_data.to_excel(writer, 'Sheet1')
writer.save()
