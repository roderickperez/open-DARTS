well_list = ['INJ', 'PRD', ]
well_type = ['INJ', 'PRD', ]
well_x = [900, 900]
well_y = [420, 1380]
from paraview.simple import *
well_cyl = Cylinder();
SetProperties(well_cyl, Height=200, Radius=10);
for idx, val in enumerate(well_list):
	# wellbore
	t = Transform(well_cyl);
	t.Transform.Translate = [well_x[idx], well_y[idx], -2000];
	t.Transform.Rotate = [90, 0, 0];
	dp = GetDisplayProperties(t);
	if (well_type[idx] == 'PRD'):
		dp.DiffuseColor = [255, 69, 0];
	else:	
		dp.DiffuseColor = [0, 255, 255];
	Show(t);
	# well name
	#title = a3DText();
	#name = well_list[idx];#well_type[idx] + ': ' + well_list[idx];
	#SetProperties(title, Text=name);
	#tt = Transform(title);
	#tt.Transform.Translate = [X0 + (well_x[idx]-0.5)*DX, Y0 + (well_y[idx]-0.5)*DY, 2100];
	#tt.Transform.Scale = [80, 80, 80];
	#tt.Transform.Rotate = [60, 30, 0];
	#dp = GetDisplayProperties(tt);
	#dp.DiffuseColor = [0, 1, 0];
	#Show(tt);


Render();

	
	





