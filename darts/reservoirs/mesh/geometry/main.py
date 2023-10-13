import numpy as np
from structured import Structured
from unstructured import Unstructured
from geometry import Geometry
from shapes import *
from wells import *



if __name__ == "__main__":
    geo = 'fluidflower'
    if geo == 'cylinder':
        filename = geo

        dim = 3
        thickness = 3
        orientation = 1
        angle = 360

        c0 = [0., 0., 0.]
        c = Cylinder(c0, radius=1, length=thickness, orientation=orientation, angle=angle)
        c.add_boundary("top")
        c.add_boundary("bottom")
        c.add_boundary("outer")
        # w = Cylinder(c0, radius=1, length=thickness, orientation=orientation, angle=angle)

        u = Unstructured(dim)
        u.add_shape(c)
        u.add_shape(w)

        u.write_geo(filename, lc=[0.2])
        u.generate_msh(filename)
    elif geo == 'cylinder_with_well':
        filename = geo

        dim = 3
        thickness = 20
        orientation = 2
        angle = 30

        c0 = [0., 0., 0.]
        c = CylinderWithHole(c0, radius=1000, rw=1, length=thickness, orientation=orientation, angle=angle)

        g = Geometry(dim)
        g.add_shape(c)

        g.write_geo(filename + '.geo', lc=[40, 2])
    elif geo == 'extruded_circle':
        filename = geo

        dim = 2
        thickness = 10
        orientation = 2
        angle = 30

        c0 = [0., 0., 0.]
        c = Circle(c0, radius=1000, orientation=orientation, angle=angle)
        # w = Circle(c0, radius=1, orientation=orientation, angle=angle)

        g = Geometry(dim)
        g.add_shape(c)
        # g.add_shape(w)

        g.extrusion['Axis'] = orientation
        g.extrusion['Distance'] = thickness
        g.extrusion['Layers'] = 1
        g.extrusion['Recombine'] = True

        g.write_geo(filename + '.geo', lc=[40])
    elif geo == 'extruded_circle_with_well':
        filename = geo

        dim = 2
        thickness = 20
        orientation = 2
        angle = 30

        c0 = [0., 0., 0.]
        c = Circle(c0, radius=1000, rw=0.075, orientation=orientation, angle=angle)

        g = Geometry(dim)
        g.add_shape(c)

        g.extrusion['Axis'] = orientation
        g.extrusion['Distance'] = thickness
        g.extrusion['Layers'] = 1
        g.extrusion['Recombine'] = True

        g.write_geo(filename + '.geo', lc=[40, 0.05])
    elif geo == 'box':
        filename = geo

        dim = 3

        b = Box([0, 1], [0, 1], [0, 1])

        m = Unstructured(dim)
        m.add_shape(b)

        m.lc = [0.1]
    elif geo == 'layered':
        filename = geo

        dim = 3

        b1 = Box([0, 1], [0, 1], [0, 1])
        b2 = Box([0, 1], [0, 1], [1, 2])

        c_0 = [0., 0., 0.]
        c1 = Cylinder(c_0, radius=1, length=2, orientation=0, angle=350)
        c_1 = [2., 0., 0.]
        c2 = Cylinder(c_1, radius=1, length=2, orientation=0, angle=30)

        m = Unstructured(dim)
        # m.add_shape(b1)
        # m.add_shape(b2)
        m.add_shape(c1)
        m.add_shape(c2)
        m.lc = [0.1]
    elif geo == 'fluidflower':
        from fluidflower import FluidFlower
        filename = geo

        f = FluidFlower()
        f.plot_shape_2D()

        dim = 2
        structured = False

        well_centers = [[0.925, 0.0, 0.32942],
                        [1.72806, 0.0, 0.72757]]

        if structured:
            m = Structured(dim=2, axs=[0, 2])
            m.add_shape(FluidFlower())

            m.generate_mesh2(nx=400, ny=1, nz=200)

        else:
            m = Unstructured(dim=2, axs=[0, 2])
            m.add_shape(FluidFlower())

            for i, center in enumerate(well_centers):
                in_surfaces = m.find_surface(center)
                m.refine_around_point(center, radius=0.015, lc=1)

                w = CircularWell(center, re=0.01, lc_well=1, axs=[0, 2], in_surfaces=in_surfaces)
                m.add_shape(w)

            m.extrude_mesh(length=0.025, layers=1, axis=1, recombine=True)
            m.lc = [0.025, 0.005]

    else:
        filename = 'mesh'

        dim = 3

        m = Unstructured(dim, axs=[0, 1])

        center = [0., 0., 0.]
        angle = 80.

        if dim == 2:
            # m.add_shape(Circle(center=center, radii=[1., 0.2], angle=angle, hole=True))
            m.add_shape(Square(center=center, ax0_len=2, ax1_len=2, orientation='xy', radii=[0.5, 0.1], hole=True))
            # m.add_shape(CircularWell([0.5, 0.5, 0.], re=0.01, lc_well=1, axs=[0, 1], in_surfaces=[1]))
            m.extrude_mesh(length=0.1, layers=3, axis=2, recombine=True)
        else:
            m.add_shape(Box(xdim=[-1., 1.], ydim=[-1., 1.], zdim=[-1., 1.], radii=[]))
            # m.add_shape(Cylinder(center=center, radii=[1., 0.5], length=1., angle=angle, hole=True))

        # m.embed_point([0.5, 0.5, 0.], lc=1)
        # m.embed_circle([0.5, 0.5, 0.], radius=0.25, orientation='xy', lc=0)
        # m.embed_circle([0.5, 0.5, 0.], radius=0.05, orientation='xy', lc=1)
        # m.embed_circle(center=center, radius=0.5, angle=angle, lc=1, orientation='xy')

        m.lc = [0.25, 0.1, 0.05]

    m.write_geo(filename)
    m.generate_msh(filename)
