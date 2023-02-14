well_list = ['I01', 'P01', 'I02', 'P02', 'I03', 'P03', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', ]
well_type = ['INJ', 'PRD', 'INJ', 'PRD', 'INJ', 'PRD', 'EXP', 'EXP', 'EXP', 'EXP', 'EXP', 'EXP', ]
well_x = [10696, 9240, 9003, 9869, 6874, 7794, 11647, 10464, 6183, 5026, 4356, 7822, ]
well_y = [9257, 9135, 6808, 7991, 7641, 8501, 8234, 5608, 3056, 5272, 8027, 11739, ]


from paraview.simple import *
well_cyl = Cylinder();
SetProperties(well_cyl,Height=1400,Radius=30);
for idx, val in enumerate(well_list):
	# wellbore
	t = Transform(well_cyl);
	t.Transform.Translate=[well_x[idx],well_y[idx],2100];
	t.Transform.Rotate = [90,0,0];
	dp = GetDisplayProperties(t);
	if (well_type[idx] == 'PRD'):
		dp.DiffuseColor=[1,0,0];
	elif (well_type[idx] == 'INJ'):	
		dp.DiffuseColor=[0,0,1];
	else:
		dp.DiffuseColor=[0,1,0];
	
	Show(t);
	# well name
	title = a3DText();
	name = well_list[idx];#well_type[idx] + ': ' + well_list[idx];
	SetProperties(title,Text=name);
	tt = Transform(title);
	tt.Transform.Translate=[well_x[idx], well_y[idx],1400];
	tt.Transform.Scale = [80, 80, 80];
	tt.Transform.Rotate = [-90, 30, 0];
	dp = GetDisplayProperties(tt);
	dp.DiffuseColor = [0, 0, 0];
	Show(tt);


Render();

	
	





