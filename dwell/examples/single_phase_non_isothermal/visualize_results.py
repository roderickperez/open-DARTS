import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from dwell.utilities.units import *

primary_variables_df = pd.read_pickle('stored_primary_variables.pkl')

num_segments = int(len(primary_variables_df['Initial conditions']) / 2)

time_steps = primary_variables_df.columns

# Initialize the plot
plt.figure(figsize=(12, 6))

# Loop through each time step and plot pressure values
for time_step in [time_steps[-1]]:   # Plot only the last time step
# for time_step in time_steps[0::5]:
    # Filter the data for the pressure profile of the current time step
    pressure_profile = primary_variables_df[time_step][0:num_segments] / bar()

    # Plot the values
    plt.plot(pressure_profile, list(range(num_segments)), label=time_step)

# Reverse the y-axis
# plt.gca().invert_yaxis()

# Set the y-axis ticks
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
# plt.gca().xaxis.set_major_locator(MultipleLocator(0.01))

# Set x-axis limits
# plt.xlim(4, 12)

# Add labels and legend
plt.xlabel('Pressure [bar]', fontsize=14)
plt.ylabel('Segment index', fontsize=14)
# plt.title('Pressure profile along the wellbore for different time steps')
plt.title('Pressure profile/profiles along the wellbore', fontsize=14)
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
# plt.tight_layout(rect=[0, 0, 0.99, 1])  # Adjust the size of the axes to make space for the legend
plt.legend(loc='upper right')
plt.show()

# Initialize the plot
plt.figure(figsize=(12, 6))

for time_step in [time_steps[-1]]:   # Plot only the last time step
# for time_step in time_steps[0::5]:
    # Filter the data for the temperature profile of the current time step
    temperature_profile = primary_variables_df[time_step][num_segments:] - 273.15

    # Plot the values
    plt.plot(temperature_profile, list(range(num_segments)), label=time_step)

# Reverse the y-axis
# plt.gca().invert_yaxis()

# Set the y-axis ticks
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
# plt.gca().xaxis.set_major_locator(MultipleLocator(1))

# Set x-axis limits
# plt.xlim(10, 90)

# Add labels and legend
plt.xlabel('Temperature [\u00B0C]', fontsize=14)
plt.ylabel('Segment index', fontsize=14)
# plt.title('Temperature profile along the wellbore for different time steps')
plt.title('Temperature profile/profiles along the wellbore', fontsize=14)
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
# plt.tight_layout(rect=[0, 0, 0.99, 1])  # Adjust the size of the axes to make space for the legend
plt.legend(loc='upper right')
plt.show()


"""---------------------------- Heat map visualization ----------------------------"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import pickle

from dwell.utilities.units import *


# y axis: standard direction and segment depths in meters
y_axis_convention = "standard"
y_axis = "segment_depth"

# y axis: T2Well direction and segment indices
# y_axis_convention = "T2Well"
# y_axis = "segment_index"


# x_axis = "time_step_index"
x_axis = "simulation_time"


if y_axis == "segment_depth":
    # Load the PipeModel instance from the pickle file
    with open('stored_dt_pipe_geometry.pkl', 'rb') as file:
        _, pipe_geom = pickle.load(file)

    measured_depths_segments = sum(pipe_geom.segments_lengths) - pipe_geom.z
    true_vertical_depths_segments = measured_depths_segments * np.cos(pipe_geom.inclination_angle_radian)
    if y_axis_convention == "standard":
        true_vertical_depths_segments = true_vertical_depths_segments
    elif y_axis_convention == "T2Well":
        true_vertical_depths_segments = true_vertical_depths_segments[::-1]

if x_axis == "simulation_time":
    # Load dt from the pickle file
    with open('stored_dt_pipe_geometry.pkl', 'rb') as file:
        dt, _ = pickle.load(file)

    simulation_time = np.concatenate(([0], np.cumsum(dt)))

# Load primary variables
primary_variables_df = pd.read_pickle('stored_primary_variables.pkl')

num_segments = int(len(primary_variables_df['Initial conditions']) / 2)
time_steps = primary_variables_df.columns

# --- First Plot: Pressure Profile ---

# Initialize the pressure matrix
pressure_matrix = np.zeros((num_segments, len(time_steps)))

# Fill the pressure matrix
for i, time_step in enumerate(time_steps):
    pressure_profile = primary_variables_df[time_step][0:num_segments] / bar()
    pressure_matrix[:, i] = pressure_profile

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(len(time_steps)), range(num_segments), pressure_matrix, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_segments), pressure_matrix, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(len(time_steps)), true_vertical_depths_segments, pressure_matrix, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, pressure_matrix, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

# Add title
ax.set_title('Pressure profile along the wellbore over time', fontsize=14)

# Add a colorbar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Pressure [bar]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()

plt.show()


# --- Second Plot: Temperature Profile ---

# Initialize the temperature matrix
temperature_matrix = np.zeros((num_segments, len(time_steps)))

# Fill the pressure matrix
for i, time_step in enumerate(time_steps):
    # Filter the data for the temperature profile of the current time step
    temperature_profile = primary_variables_df[time_step][num_segments:] - 273.15
    temperature_matrix[:, i] = temperature_profile

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(len(time_steps)), range(num_segments), temperature_matrix, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_segments), temperature_matrix, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(len(time_steps)), true_vertical_depths_segments, temperature_matrix, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, temperature_matrix, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

# Add title
ax.set_title('Temperature profile along the wellbore over time', fontsize=14)

# Add a colorbar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Temperature [\u00B0C]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()

plt.show()