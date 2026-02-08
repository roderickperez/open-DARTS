
# import meshio
import gmsh
import meshio
import numpy as np
import os

def convert_to_ascii(input_file: str, output_file: str):
    # Initialize the Gmsh API
    gmsh.initialize()

    # Open the binary .msh file
    gmsh.open(input_file)

    # Set the Gmsh option to write .msh files in ASCII format
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)

    # Write the ASCII .msh file
    gmsh.write(output_file)

    # Finalize the Gmsh API
    gmsh.finalize()

def bend_model_meshio(mesh_file):
    mesh = meshio.read(filename=mesh_file)

    x0 = (np.max(mesh.points, axis=0)[0] + np.min(mesh.points, axis=0)[0]) / 2
    y0 = (np.max(mesh.points, axis=0)[1] + np.min(mesh.points, axis=0)[1]) / 2
    # coefficient responsible for the degree of bending
    # a = 5 * np.max(mesh.points, axis=0)[2] / (x0 ** 2 + y0 ** 2)

    thickness = (np.max(mesh.points, axis=0)[2] - np.min(mesh.points, axis=0)[2])
    a = 5 * thickness / (x0 ** 2 + y0 ** 2)
    mesh.points[:,2] = mesh.points[:,2] + \
        a * (mesh.points[:,0] * (2 * x0 - mesh.points[:,0]) + mesh.points[:,1] * (2 * y0 - mesh.points[:,1]))

    # gmsh.model.mesh.setNodes(dim=0, tag=1, nodeTags=node_tags, coord=mesh.points.flatten())

    out_filename = mesh_file.split('.')[0] + '_bended.msh'
    cell_data = mesh.cell_data if mesh.cell_data else {}
    cell_sets = mesh.cell_sets if mesh.cell_sets else {}

    # Write the mesh with the entity information
    tmp_file_name = 'tmp.msh'
    meshio.write(
        tmp_file_name,
        meshio.Mesh(
            points=mesh.points,
            cells=mesh.cells,
            cell_data=cell_data,
            cell_sets=cell_sets
        ),
        file_format='gmsh22'  # or another format you wish to use
    )

    convert_to_ascii(input_file=tmp_file_name, output_file=out_filename)

    # After using the temporary file, remove it
    try:
        os.remove(tmp_file_name)
        print(f"{tmp_file_name} has been removed successfully")
    except FileNotFoundError:
        print(f"The file {tmp_file_name} does not exist")
    except PermissionError:
        print(f"Permission denied: Unable to delete {tmp_file_name}")
    except Exception as e:
        print(f"An error occurred while trying to remove {tmp_file_name}: {e}")


name = ""
mesh_file = 'data_10_10_10/spe10%s.msh' % name
bend_model_meshio(mesh_file=mesh_file)
mesh_file = 'data_20_40_40/spe10%s.msh' % name
bend_model_meshio(mesh_file=mesh_file)
mesh_file = 'data_40_80_80/spe10%s.msh' % name
bend_model_meshio(mesh_file=mesh_file)


name = "-tpfa"  # remove the tags for surface, otherwise the python discretizer will recognize the surface as fracture
mesh_file = 'data_10_10_10/spe10%s.msh' % name
bend_model_meshio(mesh_file=mesh_file)
mesh_file = 'data_20_40_40/spe10%s.msh' % name
bend_model_meshio(mesh_file=mesh_file)
mesh_file = 'data_40_80_80/spe10%s.msh' % name
bend_model_meshio(mesh_file=mesh_file)