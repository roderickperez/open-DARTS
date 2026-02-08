import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from dartsflash.dartsflash import DARTSFlash, R
from dartsflash.diagram import Diagram


class PlotEoS:
    @staticmethod
    def pressure_volume(flash: DARTSFlash, temperatures: list, compositions: list, p_props: xr.Dataset = None,
                        v_props: xr.Dataset = None, p_range: list = None, v_range: list = None,
                        logx: bool = False, logy: bool = False):
        """
        Method to plot pressure-volume (PV) diagram for EoS

        :param flash: DARTSFlash object
        :param temperatures: List of temperatures to plot
        :param compositions: List of compositions to plot
        :param p_props: xarray.Dataset with P-based properties
        :param v_props: xarray.Dataset with V-based properties
        :param p_range: Pressure range
        :param v_range: Volume range
        :return: Plot object
        """
        # Initialize Plot object
        from dartsflash.diagram import Diagram
        plot = Diagram(figsize=(8, 5))
        plot.add_attributes(#suptitle="PV diagram for " + flash.mixture_name,
                            ax_labels=["volume, m3/mol", "pressure, bar"])

        # Plot P vs. V from VT-properties
        if v_props is not None:
            # Slice Dataset at composition
            comps = {comp: compositions[i] for i, comp in enumerate(flash.components[:-1]) if comp in v_props.dims}

            volume = v_props.volume.values
            pressure = np.array([v_props.sel(comps, method='nearest').sel(
                {"temperature": temp}, method='nearest').P.values for temp in temperatures])
            plot.draw_line(X=volume, Y=pressure,
                           datalabels=['{} K'.format(temp) for temp in temperatures])
            plot.set_axes(xlim=v_range, ylim=p_range)

        # Plot P vs. V from PT-properties
        if p_props is not None:
            # Slice Dataset at composition
            comps = {comp: compositions[i] for i, comp in enumerate(flash.components[:-1]) if comp in p_props.dims}

            volume = np.array([p_props.sel(comps, method='nearest').sel(
                {"temperature": temp}, method='nearest').V.values for temp in temperatures])
            pressure = np.tile(p_props.pressure.values, (len(temperatures), 1))
            plot.draw_line(X=volume, Y=pressure, styles='dashed',
                           datalabels=['{} K'.format(temp) for temp in temperatures] if v_props is None else None)
        plot.add_attributes(legend=True)
        plot.set_axes(xlim=v_range, ylim=p_range)

        if logx:
            plot.ax[plot.subplot_idx].set_xscale("log")
        if logy:
            plot.ax[plot.subplot_idx].set_yscale("log")

        return plot

    @staticmethod
    def compressibility(flash: DARTSFlash, temperatures: list, compositions: list, p_props: xr.Dataset = None,
                        v_props: xr.Dataset = None, p_range: list = None, z_range: list = None):
        """
        Method to plot compressibility factor Z versus pressure for EoS

        :param flash: DARTSFlash object
        :param temperatures: List of temperatures to plot
        :param compositions: List of compositions to plot
        :param p_props: xarray.Dataset with P-based properties
        :param v_props: xarray.Dataset with V-based properties
        :param p_range: Pressure range
        :param z_range: Compressibility factor range
        :return: Plot object
        """
        # Initialize Plot object
        from dartsflash.diagram import Diagram
        plot = Diagram(figsize=(8, 5))
        plot.add_attributes(#suptitle="Compressibility factor Z of " + flash.mixture_name,
                            ax_labels=["pressure, bar", "Z, -"])

        # Plot Z vs. P from VT-properties
        if v_props is not None:
            # Slice Dataset at composition
            comps = {comp: compositions[i] for i, comp in enumerate(flash.components[:-1]) if comp in v_props.dims}

            Z = np.array([v_props.sel(comps, method='nearest').sel({"temperature": temp}, method='nearest').Z.values for temp in temperatures])
            pressure = np.array([v_props.sel(comps, method='nearest').sel({"temperature": temp}, method='nearest').P.values for temp in temperatures])
            plot.draw_line(X=pressure, Y=Z,
                           datalabels=['{} K'.format(temp) for temp in temperatures])
            plot.set_axes(xlim=p_range, ylim=z_range)

        # Plot Z vs. P from PT-properties
        if p_props is not None:
            # Slice Dataset at composition
            comps = {comp: compositions[i] for i, comp in enumerate(flash.components[:-1]) if comp in p_props.dims}

            Z = np.array([p_props.sel(comps, method='nearest').sel({"temperature": temp}, method='nearest').Z.values for temp in temperatures])
            pressure = np.array([p_props.pressure.values for temp in temperatures])
            plot.draw_line(X=pressure, Y=Z, styles="dashed",
                           datalabels=['{} K'.format(temp) for temp in temperatures] if v_props is None else None)
            plot.set_axes(xlim=p_range, ylim=z_range)
        plot.add_attributes(legend=True)

        return plot

    @staticmethod
    def plot(flash: DARTSFlash, props: xr.Dataset, x_var: str, prop_names: list, composition: list,
             state: dict = None, title: list = None, ax_labels: list = None, colours: list = None, datalabels: list = None):
        """
        Method to plot EoS properties

        :param flash: DARTSFlash object
        :param props: xarray.Dataset with properties
        :param x_var: Name of x-axis variable
        :param prop_names: List of properties to plot
        :param composition: List of compositions to plot
        :param state: Dictionary of states to plot
        :param title: Title of plot
        :param ax_labels: List of axes labels
        :param colours: List of colours
        :param datalabels: List of labels
        :return: Plot object
        """
        # Slice dataset at current state
        prop_names = [prop_names] if isinstance(prop_names, str) else prop_names
        comps = {comp: composition[i] for i, comp in enumerate(flash.components[:-1])
                 if comp in props.dims and comp != x_var}
        props_at_state = props.sel(comps, method='nearest').sel(state, method='nearest').squeeze().transpose(..., x_var)
        x = props_at_state.coords[x_var].values

        # Create Diagram object
        from dartsflash.diagram import Diagram
        nplots = len(prop_names)
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))

        assert props is not None, "Please provide the properties to plot"
        for i, prop_name in enumerate(prop_names):
            plot.subplot_idx = i
            prop_array = eval("props_at_state." + prop_name + ".values")

            plot.draw_line(X=x, Y=prop_array, colours=colours, datalabels=datalabels)
            plot.add_attributes(title=title if isinstance(title, str) else (title[i] if title is not None else None),
                                ax_labels=ax_labels)
        return plot

    @staticmethod
    def surf(flash: DARTSFlash, props: xr.Dataset, x_var: str, y_var: str, prop_names: list, composition: list,
             state: dict = None, logx: bool = False, logy: bool = False, title: list = None, ax_labels: list = None, cmap: str = 'winter'):
        """
        Method to plot 2D diagram of EoS properties

        :param flash: DARTSFlash object
        :param props: xarray.Dataset with properties
        :param x_var: Name of x variable
        :param y_var: Name of y variable
        :param prop_names: List of properties to plot
        :param composition: List of compositions to plot
        :param state: Dictionary of states to plot
        :param title: Title of plot
        :param ax_labels: List of axes labels
        :param cmap: Color map to use
        :return: Plot object
        """
        # Slice dataset at current state
        comps = {comp: composition[i] for i, comp in enumerate(flash.components[:-1]) if comp in props.dims
                 and comp != x_var and comp != y_var}
        props_at_state = props.sel(comps, method='nearest').sel(state, method='nearest').squeeze().transpose(x_var, y_var)
        x = props_at_state.coords[x_var].values
        y = props_at_state.coords[y_var].values

        # Create Diagram object
        from dartsflash.diagram import Diagram
        nplots = len(prop_names)
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))
        is_float = [type(props.dtypes[prop]) is np.dtypes.Float64DType for prop in prop_names]

        assert props is not None, "Please provide the properties to plot"
        for i, prop_name in enumerate(prop_names):
            plot.subplot_idx = i
            prop_array = eval("props_at_state." + prop_name + ".values")

            plot.draw_surf(x=x, y=y, data=prop_array, colours=cmap, colorbar=True, contour=True, fill_contour=True,
                           is_float=is_float[i], ax_labels=ax_labels, logx=logx, logy=logy,
                           )
            plot.add_attributes(title=title if isinstance(title, str) else (title[i] if title is not None else None),
                                ax_labels=ax_labels)
        return plot


class PlotProps:
    @staticmethod
    def plot(flash: DARTSFlash, props: xr.Dataset, x_var: str, prop_names: list, composition: list, variable_comp: str = None,
             state: dict = None, title: list = None, ax_labels: list = None, colours: list = None, styles: list = None,
             datalabels: list = None):
        """
        Method to plot thermodynamic surfaces

        :param flash: DARTSFlash object
        :param props: xarray.Dataset with properties
        :param x_var: Name of x-axis variable
        :param prop_names: List of properties to plot
        :param composition: List of compositions to plot
        :param state: Dictionary of states to plot
        :param title: Title of plot
        :param ax_labels: List of axes labels
        :param colours: List of colours
        :param styles: List of linestyles
        :param datalabels: List of labels
        :return: Plot object
        """
        # Slice dataset at current state
        prop_names = [prop_names] if isinstance(prop_names, str) else prop_names
        if isinstance(composition, dict):
            comps = composition
        else:
            comps = {comp: composition[i] for i, comp in enumerate(flash.components[:-1])
                     if comp in props.dims and comp != x_var}
        variable_comp = variable_comp if variable_comp is not None else flash.components[0]
        # comps[variable_comp] = [variable_comp] if np.isscalar(comps[variable_comp]) else comps[variable_comp]

        props_at_state = props.sel(comps, method='nearest').sel(state, method='nearest').squeeze().transpose(..., x_var)
        x = props_at_state.coords[x_var].values

        # Create Diagram object
        from dartsflash.diagram import Diagram
        nplots = len(prop_names)
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))

        assert props is not None, "Please provide the properties to plot"
        for i, prop_name in enumerate(prop_names):
            plot.subplot_idx = i

            prop_array = eval("props_at_state." + prop_name + ".values")
            for j, zc in enumerate(comps[variable_comp]):
                plot.draw_line(X=x, Y=prop_array[j], colours=colours,
                               styles=styles[j] if styles is not None else Diagram.linestyles[j],
                               datalabels=datalabels)
                plot.add_attributes(title=title if isinstance(title, str) else (title[i] if title is not None else None),
                                    ax_labels=ax_labels)
        return plot

    @staticmethod
    def binary(flash: DARTSFlash, state: dict, prop_name: str, variable_comp_idx: int, dz: float,
               min_z: list = None, max_z: list = None, plot_1p: bool = True, props: xr.Dataset = None,
               flash_results: xr.Dataset = None, sp_results: xr.Dataset = None, composition_to_plot: list = None,
               title: str = None, ax_label: str = None, datalabels: list = None, legend_loc: str = None):
        """
        Method to plot binary thermodynamic surfaces.

        :param flash: DARTSFlash object
        :param state: Dictionary of states to plot surfaces for
        :param variable_comp_idx: Index of variable component
        :param dz: Composition interval
        :param min_z: Minimum composition for composition variable, will default to 0
        :param max_z: Maximum composition for composition variable, will default to 1
        :param plot_1p: Plot single-phase or multiphase results
        :param props: xarray.DataArray of properties
        :param flash_results: xarray.Dataset of flash results
        :param sp_results: xarray.Dataset of stationary points results
        :param composition_to_plot: Composition to plot results for
        :param prop_name: Name of property for indexing
        :param title: Title of plot
        :param ax_label: Axes label
        :param datalabels: List of data labels
        """
        variable_comp = flash.components[variable_comp_idx]
        comp_str = flash.comp_data.comp_labels[variable_comp_idx]
        min_z = [0.] if min_z is None else min_z
        max_z = [1.] if max_z is None else max_z
        z0 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)
        state_len = [len(arr) if not np.isscalar(arr) else 1 for arr in state.values()]
        assert not (state_len[0] > 1 and state_len[1] > 1), "Can't specify array for both state specifications"
        n_surfaces = np.amax(state_len)

        # Initialize plot
        from dartsflash.diagram import Diagram
        plot = Diagram(figsize=(8, 5))

        # Slice results at current state
        res_at_state = props.sel(state, method='nearest').transpose(..., variable_comp).squeeze()
        ydata = eval("res_at_state." + prop_name + ".values")
        minY, maxY = np.nanmin(ydata), np.nanmax(ydata)

        # Loop over state specifications
        for i in range(n_surfaces):
            # Plot 1P/equilibrium state
            data = ydata[i, ...] if n_surfaces > 1 else ydata
            if plot_1p:
                # Loop over EoS to plot 1P curves

                for j in range(data.shape[0]):
                    plot.draw_line(X=z0, Y=data[j, ...] if data.ndim >= 2 else data,
                                   colours=plot.colours[j], styles=plot.linestyles[i],
                                   datalabels=datalabels[j] if i == 0 and datalabels is not None else None)

            else:
                # Plot equilibrium state
                plot.draw_line(X=z0, Y=data, colours=plot.colours[i],
                               datalabels=datalabels[i] if datalabels is not None else None)

            # Plot flash results at composition
            if flash_results is not None:
                assert composition_to_plot is not None, "Please provide composition to plot equilibrium phases for"

                # Slice flash results at current state
                flash_at_pt = flash_results.sel(state, method='nearest').squeeze()
                comps = {comp: composition_to_plot[i] for i, comp in enumerate(flash.components[:-1])}

                # Find flash results at P,T,z
                flash_at_ptz = flash_at_pt.sel(comps, method='nearest').squeeze()
                X = flash_at_ptz.X.values
                eos_idx = flash_at_ptz.eos_idx.values
                root = flash_at_ptz.root_type.values
                Xj = np.array([X[jj * flash.ns:(jj + 1) * flash.ns] for jj in range(flash.np_max)])

                # Plot phase compositions xij at feed composition z
                xx, yy = [], []
                for j, xj in enumerate(Xj):
                    if flash_at_ptz.nu.values[j] > 0.:
                        comps = {comp: Xj[j, i] for i, comp in enumerate(flash.components[:-1])}
                        data = res_at_state.sel(comps, method='nearest')
                        value = eval("data." + prop_name + ".values")[eos_idx[j] + (0 if root[j] != 1 else 1)]

                        xx.append(xj)
                        yy.append(np.float64(value))
                xx = np.array(xx)
                plot.draw_point(X=xx[:, 0], Y=yy, colours='r')
                plot.draw_line(X=xx[:, 0], Y=yy, colours='r')

            # Plot stationary points from stability analysis
            if sp_results is not None:
                assert composition_to_plot is not None, "Please provide composition to plot stationary points for"

                # Slice results at current state
                sp_at_pt = sp_results.sel(state, method='nearest').squeeze()
                comps = {comp: composition_to_plot[i] for i, comp in enumerate(flash.components[:-1])}

                # Find stationary points at P,T,z
                sp_at_ptz = sp_at_pt.sel(comps, method='nearest').squeeze()
                Y = sp_at_ptz.y.values
                tot_sp = sp_at_ptz.tot_sp.values
                tpd = sp_at_ptz.tpd.values
                eos_idx = sp_at_ptz.eos_idx.values
                roots = sp_at_ptz.root_type.values

                Yj = np.array([Y[jj * flash.ns:(jj + 1) * flash.ns] for jj in range(tot_sp)])
                yj = []
                for j, xj in enumerate(Yj):
                    if not np.isnan(xj[0]):
                        comps = {comp: Yj[j, i] for i, comp in enumerate(flash.components[:-1])}
                        data = res_at_state.sel(comps, method='nearest')
                        value = eval("data." + prop_name + ".values")
                        value = value[eos_idx[j]] if len(flash.eos.keys()) > 1 else value
                    else:
                        value = np.nan
                    yj.append(np.float64(value))

                # Plot phase compositions xij at feed composition z
                plot.draw_point(X=Yj[:, 0], Y=yj, colours='g')

        plot.add_attributes(suptitle=title, ax_labels=[comp_str, ax_label], grid=True,
                            legend=(datalabels is not None), legend_loc=legend_loc)
        plot.ax[plot.subplot_idx].set(xlim=[min_z[0], max_z[0]], ylim=[minY, maxY])

        return plot

    @staticmethod
    def ternary(flash: DARTSFlash, state: dict, variable_comp_idx: list, dz: float, min_z: list = None, max_z: list = None,
                plot_1p: bool = True, props: xr.Dataset = None, flash_results: xr.Dataset = None, sp_results: xr.Dataset = None,
                composition_to_plot: list = None, prop_name: str = "", title: str = None, cmap: str = 'winter'):
        """
        Method to plot ternary thermodynamic surfaces.

        :param flash: DARTSFlash object
        :param state: Dictionary of states to plot surfaces for
        :param variable_comp_idx: Indices of variable components
        :param dz: Composition interval
        :param min_z: Minimum composition for composition variables, will default to 0 for each
        :param max_z: Maximum composition for composition variables, will default to 1 for each
        :param props: xarray.DataArray of properties
        :param flash_results: xarray.Dataset of flash results
        :param sp_results: xarray.Dataset of stationary points results
        :param composition_to_plot: Composition to plot results for
        :param prop_name: Name of property for indexing
        :param title: Title of plot
        :param cmap: plt.Colormap for plotting
        """
        variable_comps = [flash.components[idx] for idx in variable_comp_idx]
        comp_str = [flash.comp_data.comp_labels[idx] for idx in variable_comp_idx]
        min_z = [0., 0.] if min_z is None else min_z
        max_z = [1., 1.] if max_z is None else max_z
        x0 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)
        x1 = np.arange(min_z[1], max_z[1] + dz * 0.1, dz)
        assert np.all([1 if np.isscalar(arr) else len(arr) == 1 for arr in state.values()]), \
            "Can't specify array for state specifications"

        # Slice results at current state
        res_at_state = props.sel(state, method='nearest').transpose(variable_comps[0], variable_comps[1], ...).squeeze()
        ydata = eval("res_at_state." + prop_name + ".values")

        if plot_1p:
            # Create TernaryDiagram object
            from dartsflash.diagram import TernaryDiagram
            nplots = flash.n_phase_types
            plot = TernaryDiagram(ncols=nplots, figsize=(nplots * 5 + 3, 5), dz=dz)

            # Plot Gibbs energy surface of each EoS
            for j in range(flash.n_phase_types):
                data = ydata[..., j] if ydata.ndim > 2 else ydata
                plot.subplot_idx = j
                if np.nanmin(data):
                    plot.draw_surf(X1=x0, X2=x1, data=data, is_float=True, nlevels=10,
                                   colours=cmap, colorbar=True, contour=True, fill_contour=True,
                                   corner_labels=flash.comp_data.comp_labels
                                   )
                # plot.add_attributes(title=eosname)
        else:
            # Create TernaryDiagram object
            from dartsflash.diagram import TernaryDiagram
            plot = TernaryDiagram(ncols=1, figsize=(8, 5), dz=dz)

            # Plot Gibbs energy surface of equilibrium state
            plot.draw_surf(X1=x0, X2=x1, data=ydata, is_float=True, nlevels=10,
                           colours=cmap, colorbar=True, contour=True, fill_contour=True,
                           corner_labels=flash.comp_data.comp_labels
                           )

        # Plot flash results at specified composition
        if flash_results is not None:
            assert composition_to_plot is not None, "Please provide composition to plot equilibrium phases for"

            # Plot phase compositions xij at feed composition z
            comps = {comp: composition_to_plot[i] for i, comp in enumerate(flash.components[:-1])}
            # plot.draw_compositions(compositions=composition_to_plot, colours='k')

            # Find flash results at P,T,z
            flash_at_ptz = flash_results.sel({**state, **comps}, method='nearest').squeeze()
            X = flash_at_ptz.X.values
            Xj = [X[jj * flash.ns:(jj + 1) * flash.ns] for jj in range(flash.np_max)]

            # Plot phase compositions
            if flash_at_ptz.np.values > 1:
                eos_idx = flash_at_ptz.eos_idx.values
                roots = flash_at_ptz.root_type.values
                axes = [eos_idx[jj] + (1 if roots[jj] == 0 else 0) for jj in range(flash.np_max)]
                plot.draw_compositions(compositions=Xj, axes=axes,
                                       colours='r', connect_compositions=True)
            else:
                # For single phase conditions, skip
                pass

        # Plot stationary points from stability analysis
        if sp_results is not None:
            assert composition_to_plot is not None, "Please provide composition to plot stationary points for"

            # Slice results at current state
            comps = {comp: composition_to_plot[i] for i, comp in enumerate(flash.components[:-1])}

            # Find stationary points at P,T,z
            sp_at_ptz = sp_results.sel({**state, **comps}, method='nearest').squeeze()
            Y = sp_at_ptz.y.values
            tot_sp = sp_at_ptz.tot_sp.values
            tpd = sp_at_ptz.tpd.values

            Yj = np.array([Y[jj * flash.ns:(jj + 1) * flash.ns] for jj in range(tot_sp)])

            # Plot phase compositions
            if not np.all(np.isnan(tpd)):
                eos_idx = sp_at_ptz.eos_idx.values
                roots = sp_at_ptz.root_type.values
                axes = [eos_idx[jj] + (1 if roots[jj] == 0 else 0) for jj in range(tot_sp)]

                plot.draw_compositions(compositions=Yj, axes=axes,
                                       colours='g', connect_compositions=False)
            else:
                # For single phase conditions, skip
                pass

        plot.add_attributes(suptitle=title)

        return plot


class PlotPhaseDiagram:
    @staticmethod
    def binary(flash: DARTSFlash, flash_results: xr.Dataset, variable_comp_idx: int, dz: float, state: dict,
               min_z: list = None, max_z: list = None):
        """
        Method to plot P-x and T-x diagrams

        :param flash: DARTSFlash object
        :param flash_results: xarray.DataArray
        :param state: State to plot GE surfaces for
        :param variable_comp_idx: Index of variable component
        """
        variable_comp = flash.components[variable_comp_idx]
        comp_str = flash.comp_data.comp_labels[variable_comp_idx]

        # Slice dataset at current state
        px = 0 if np.isscalar(state["pressure"]) else len(state["pressure"]) > 1
        y = state["pressure"] if px else state["temperature"]
        min_z = [0.] if min_z is None else min_z
        max_z = [1.] if max_z is None else max_z
        z0 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)

        comps = {comp: flash_results.dims[comp] for i, comp in enumerate(flash.components[:-1])
                 if comp != variable_comp and comp in flash_results.dims}
        results_at_state = flash_results.sel(state, method='nearest').sel(comps, method='nearest').squeeze()

        # Create TernaryDiagram object
        from dartsflash.diagram import Diagram
        nplots = 1
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))
        plot.add_attributes(ax_labels=[comp_str, "pressure, bar" if px else "temperature, K"],
                            suptitle="Phase diagram for " + flash.mixture_name +
                                     (" at T = {} K".format(state["temperature"]) if px else " at P = {} bar".format(state["pressure"])))

        # Plot phase compositions xij at feed composition z
        for i, yi in enumerate(y):
            state = {("pressure" if px else "temperature"): yi}
            j = 0
            while j < len(z0):
                # Find flash results at P,T,z
                comps = {variable_comp: z0[j]}
                results_at_ptz = results_at_state.sel(state, method='nearest').sel(comps, method='nearest').squeeze()

                # For single phase conditions, skip
                if results_at_ptz.np.values > 1:
                    Xj = [results_at_ptz.X.values[jj * flash.ns + variable_comp_idx] for jj in range(flash.np_max)]
                    Y = [yi for jj in range(flash.np_max)]
                    j = np.where(z0 >= np.nanmax(Xj))[0][0]
                    plot.draw_point(X=Xj, Y=Y, colours='k')
                else:
                    j += 1
                    pass
        plot.ax[plot.subplot_idx].set(xlim=[min_z[0], max_z[0]], ylim=[y[0], y[-1]])

        return plot

    @staticmethod
    def ternary(flash: DARTSFlash, flash_results: xr.Dataset, dz: float, state: dict, min_z: list = None, max_z: list = None,
                plot_tielines: bool = False):
        """
        Method to plot ternary phase diagram. The phase diagram is generated by plotting the compositions
        of the equilibrium phases at the specified feed compositions. Optional tielines connect these compositions.

        :param flash: DARTSFlash object
        :param flash_results: xarray.Dataset
        :param dz: float
        :param state: dict
        :param min_z: list
        :param max_z: list
        :param plot_tielines: bool
        """
        # Slice dataset at current state
        for spec in state.keys():
            state[spec] = state[spec][0] if not np.isscalar(state[spec]) else state[spec]
        flash_at_pt = flash_results.sel(state, method='nearest').squeeze()

        min_z = [0., 0.] if min_z is None else min_z
        max_z = [1., 1.] if max_z is None else max_z
        z0 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)
        z1 = np.arange(min_z[1], max_z[1] + dz * 0.1, dz)

        # Create TernaryDiagram object
        from dartsflash.diagram import TernaryDiagram
        nplots = 1
        plot = TernaryDiagram(ncols=nplots, figsize=(nplots * 5 + 3, 5), dz=dz)
        plot.add_attributes(suptitle="Phase diagram for " + flash.mixture_name + " at P = {} bar and T = {} K"
                            .format(state["pressure"], state["temperature"]))
        plot.triangulation(z0, z1, corner_labels=flash.comp_data.comp_labels)

        # Plot phase compositions xij at feed composition z
        comps = {comp: None for comp in flash.components[:-1]}
        for i, zi in enumerate(z0):
            comps[flash.components[0]] = zi
            for j, zj in enumerate(z1):
                comps[flash.components[1]] = zj

                # Find flash results at P,T,z
                flash_at_ptz = flash_at_pt.sel(comps, method='nearest').squeeze()
                X = flash_at_ptz.X.values
                Xj = [X[jj * flash.ns:(jj + 1) * flash.ns] for jj in range(flash.np_max)]

                # For single phase conditions, skip
                if flash_at_ptz.np.values == 3:
                    plot.draw_compositions(compositions=Xj, colours='b', connect_compositions=True)
                elif flash_at_ptz.np.values == 2:
                    plot.draw_compositions(compositions=Xj, colours='r', connect_compositions=plot_tielines)
                else:
                    pass

        return plot

    @staticmethod
    def pt(flash: DARTSFlash, flash_results: xr.Dataset, compositions: list, state: dict = None,
           logx: bool = False, logy: bool = False):
        """
        Method to plot PT phase diagram at composition z. The phase diagram is generated by plotting the boundaries
        of the phase regions.

        """
        # Create Diagram object
        from dartsflash.diagram import Diagram
        nplots = 1
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))
        plot.add_attributes(suptitle="PT diagram for " + flash.mixture_name,
                            ax_labels=["temperature, K", "pressure, bar"])

        # Loop over different compositions
        compositions = [compositions] if not hasattr(compositions[0], '__len__') else compositions
        levels, _ = plot.get_levels(compositions[:][0], is_float=True, nlevels=len(compositions))
        cmap, _ = plot.get_cmap(levels)
        plot.ax[plot.subplot_idx].set_prop_cycle('color', [cmap(i) for i in np.linspace(0, 1, len(compositions))])

        for i, z in enumerate(compositions):
            # Slice dataset at current state
            comps = {comp: z[i] for i, comp in enumerate(flash.components[:-1]) if comp in flash_results.dims}
            flash_at_z = flash_results.sel(comps, method='nearest').squeeze()
            state = state if state is not None else {"pressure": flash_results.pressure.values,
                                                     "temperature": flash_results.temperature.values}

            # Plot phase compositions xij at feed composition z
            T_ranges = np.empty((len(state["pressure"]), 2))
            for ii, p in enumerate(state["pressure"]):
                # Slice flash results at P,z
                np_at_pz = flash_at_z.sel({"pressure": p}, method='nearest').squeeze().np.values
                T_ranges[ii, :] = None
                for j, t in enumerate(state["temperature"]):
                    if np_at_pz[j] >= 1.:
                        T_ranges[ii, 0] = t
                        break
                for jj, t in enumerate(state["temperature"][j:]):
                    if np_at_pz[j + jj] == 1.:
                        T_ranges[ii, 1] = t
                        break
                if T_ranges[ii, 0] == T_ranges[ii, 1]:
                    break

            plot.draw_line(np.append(np.append(T_ranges[:, 0], [None]), T_ranges[:, 1]),
                           np.append(np.append(state["pressure"], [None]), state["pressure"]))

        if logx:
            plot.ax[plot.subplot_idx].set_xscale("log")
        if logy:
            plot.ax[plot.subplot_idx].set_yscale("log")

        return


class PlotFlash:
    @staticmethod
    def solubility(flash: DARTSFlash, flash_results: xr.Dataset, dissolved_comp_idx: list, phase_idx: list, x_var: str,
                   state: dict = None, concentrations: list = None, logy: bool = False, xlim: list = None, ylim: list = None,
                   colours: list = None, styles: list = None, labels: list = None, plot_1p: bool = True,
                   legend: bool = True, legend_loc: str = "upper right", sharex: bool = False):
        # Slice Dataset at current state
        state = state if state is not None else {"pressure": flash_results.pressure.values,
                                                 "temperature": flash_results.temperature.values}
        dims_to_squeeze = [dim for dim, size in flash_results.dims.mapping.items() if size == 1 and dim is not x_var and dim not in state.keys()]
        flash_at_state = flash_results.sel(state, method='nearest').squeeze(dim=dims_to_squeeze)

        # Initialize Plot object
        from dartsflash.diagram import Diagram
        dissolved_comp_idx = [dissolved_comp_idx] if not hasattr(dissolved_comp_idx, "__len__") else dissolved_comp_idx
        nplots = len(dissolved_comp_idx)
        plot = Diagram(nrows=nplots, figsize=(8, nplots * 5), sharex=sharex)

        # Loop over components
        for j, idx in enumerate(dissolved_comp_idx):
            plot.subplot_idx = j
            phase_name = flash.flash_params.eos_order[phase_idx]
            comp_name = flash.components[idx]
            dissolved_comp = flash.comp_data.comp_labels[idx]
            comp_idx = phase_idx * flash.ns + idx
            plot.add_attributes(#title=dissolved_comp + " solubility in " + phase_name,
                                ax_labels=[("pressure, bar" if x_var == 'pressure' else "temperature, K"),
                                           r"x{}".format(dissolved_comp)])

            if concentrations is not None:
                assert styles is not None, "Please specify styles for plotting of different concentrations"
                for i, concentration in enumerate(concentrations):
                    x_at_state = flash_at_state.isel(concentrations=i, X_array=comp_idx).X.transpose(..., x_var).values

                    # Plot solubility data
                    colour = colours[i] if colours is not None else plot.colours[i]
                    plot.draw_line(X=state[x_var], Y=x_at_state, colours=colour, styles=styles[i],
                                   datalabels=labels[i])

            else:
                x_at_state = flash_at_state.isel(X_array=comp_idx).X.transpose(..., x_var).values
                x_at_state = np.where(flash_at_state.np.transpose(..., x_var).values > (0 if plot_1p else 1), x_at_state, np.nan)
                x_at_state = x_at_state.squeeze()

                # Plot solubility data
                plot.draw_line(X=state[x_var], Y=x_at_state, colours=colours, styles=styles, datalabels=labels)

            plot.add_attributes(legend=legend, legend_loc=legend_loc, grid=True)
            plot.set_axes(xlim=xlim, ylim=ylim, logy=logy)

        return plot

    @staticmethod
    def binary(flash: DARTSFlash, flash_results: xr.Dataset, state: dict, variable_comp_idx: int, dz: float, min_z: list = None, max_z: list = None,
               composition_to_plot: list = None, plot_phase_fractions: bool = False, cmap: str = 'winter'):
        """
        Method to plot P-x and T-x diagrams

        :param flash: DARTSFlash object
        :param flash_results: xarray.DataArray
        :param state: State to plot GE surfaces for
        :param variable_comp_idx: Index of variable component
        :param composition_to_plot: list
        """
        variable_comp = flash.components[variable_comp_idx]
        comp_str = flash.comp_data.comp_labels[variable_comp_idx]

        # Slice dataset at current state
        px = hasattr(state["pressure"], "__len__")
        y = state["pressure"] if px else state[flash.get_vars[flash.flash_type][1]]
        ylabel = "pressure, bar" if px else flash.get_labels[flash.flash_type][1]
        min_z = [0.] if min_z is None else min_z
        max_z = [1.] if max_z is None else max_z
        z0 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)

        comps = {comp: flash_results.dims[comp] for i, comp in enumerate(flash.components[:-1])
                 if comp != variable_comp and comp in flash_results.dims}
        results_at_state = flash_results.sel(state, method='nearest').sel(comps, method='nearest').squeeze()
        np_max = int(np.nanmax(results_at_state.np.values))

        # Create TernaryDiagram object
        from dartsflash.diagram import Diagram
        nplots = np_max - 1 if plot_phase_fractions else 1
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))

        if plot_phase_fractions:
            # Plot flash results at feed composition z
            for j in range(np_max - 1):
                plot.subplot_idx = j
                data = results_at_state.nu.isel(nu_array=j).values
                data[data == 0.] = np.nan
                # data[data >= 1.] = np.nan
                plot.draw_surf(z0, y, data=data, is_float=True, colours=cmap, colorbar=True,
                               contour=False, fill_contour=True, min_val=0.+1e-10, max_val=1., nlevels=11,
                               )
                plot.add_attributes(title="Phase {}".format(j),
                                    ax_labels=[comp_str, ylabel])
            plot.add_attributes(suptitle="Phase fraction for " + flash.mixture_name +
                                         (" at T = {} K".format(state["temperature"]) if px else " at P = {} bar".format(state["pressure"])))
        else:
            # Plot np at feed composition z
            phase_state_ids = results_at_state.phase_state_id.values
            # contours = plot.get_contours(z0, y, phase_state_ids)
            plot.draw_contours(z0, y, phase_state_ids, colours='k')
            plot.subplot_idx = 0
            # plot.draw_surf(z0, y, data=phase_state_ids, is_float=False, colours=cmap, colorbar=True,
            #                contour=False, fill_contour=True, min_val=0, max_val=len(flash.flash_params.phase_states_str)-2,
            #                # colorbar_labels=flash.flash_params.phase_states_str
            #                colorbar_labels=["Aq", "V", "L", "Aq-V", "Aq-L", "V-L", "Aq-V-L"],
            #                )
            plot.add_attributes(ax_labels=[comp_str, ylabel],
                                # suptitle=flash.mixture_name +
                                #          (" at T = {} K".format(state["temperature"]) if px else " at P = {} bar".format(state["pressure"]))
                                )

            # # Plot equilibrium phases at composition
            # if composition_to_plot is not None:
            #     assert len(composition_to_plot) == flash.ns
            #     comps = {comp: composition_to_plot[i] for i, comp in enumerate(flash.components[:-1])}
            #
            #     X = results_at_state.sel(comps, method='nearest').squeeze().X.values
            #     Xj = [X[j * flash.ns:(j + 1) * flash.ns] for j in range(flash.np_max)]
            #     plot.draw_point(compositions=composition_to_plot, colours='k')
            #     plot.draw_point(compositions=Xj, colours='r', connect_compositions=True)

        return plot

    @staticmethod
    def ternary(flash: DARTSFlash, flash_results: xr.Dataset, state: dict, dz: float, min_z: list = None, max_z: list = None,
                composition_to_plot: list = None, plot_phase_fractions: bool = False, cmap: str = 'winter'):
        """
        Method to plot flash results
        """
        # Slice dataset at current state
        for spec in state.keys():
            state[spec] = state[spec][0] if hasattr(state[spec], "__len__") else state[spec]
        flash_at_pt = flash_results.sel(state, method='nearest').squeeze()
        np_max = int(np.nanmax(flash_at_pt.np.values))

        min_z = [0., 0.] if min_z is None else min_z
        max_z = [1., 1.] if max_z is None else max_z
        z0 = np.arange(min_z[0], max_z[0]+dz*0.1, dz)
        z1 = np.arange(min_z[1], max_z[1]+dz*0.1, dz)

        # Create TernaryDiagram object
        from dartsflash.diagram import TernaryDiagram
        nplots = np_max - 1 if plot_phase_fractions else 1
        plot = TernaryDiagram(ncols=nplots, figsize=(nplots * 6, 5), dz=dz, min_z=min_z, max_z=max_z)

        if plot_phase_fractions:
            # Plot flash results at feed composition z
            for j in range(np_max - 1):
                plot.subplot_idx = j
                plot.draw_surf(z0, z1, data=flash_at_pt.nu.isel(nu_array=j).values, is_float=True, nlevels=10,
                               colours=cmap, colorbar=True, contour=False, fill_contour=True, min_val=0., max_val=1.,
                               corner_labels=flash.comp_data.comp_labels
                               )
                plot.add_attributes(title="Phase {}".format(j))
            plot.add_attributes(suptitle="Phase fraction for " + flash.mixture_name + " at P = {} bar and T = {} K"
                                .format(state["pressure"], state["temperature"]))
        else:
            # Plot phase state ids at feed composition z
            phase_state_ids = flash_at_pt.phase_state_id.values
            # phase_state_ids[phase_state_ids == 0] = -1
            plot.subplot_idx = 0
            plot.draw_contours(z0, z1, phase_state_ids, colours='k', corner_labels=flash.comp_data.comp_labels,
                               # mask=0,
                               linewidth=0.5)
            # plot.draw_surf(z0, z1, data=phase_state_ids, is_float=False, colours=cmap, colorbar=True,
            #                contour=False, fill_contour=True, min_val=0, max_val=len(flash.flash_params.phase_states_str)-2,
            #                corner_labels=flash.comp_data.comp_labels,
            #                # colorbar_labels=flash.flash_params.phase_states_str,
            #                colorbar_labels=["Aq", "V", "L", "Aq-V", "Aq-L", "V-L", "Aq-V-L"],
            #                )
            # plot.add_attributes(suptitle="Number of phases for " + flash.mixture_name + " at P = {} bar and T = {} K"
            #                     .format(state["pressure"], state["temperature"]))

            # Plot TPD at composition
            if composition_to_plot is not None:
                assert len(composition_to_plot) == flash.ns
                comps = {comp: composition_to_plot[i] for i, comp in enumerate(flash.components[:-1])}

                X = flash_at_pt.sel(comps, method='nearest').squeeze().X.values
                Xj = [X[j * flash.ns:(j + 1) * flash.ns] for j in range(flash.np_max)]
                plot.draw_compositions(compositions=composition_to_plot, colours='k')
                plot.draw_compositions(compositions=Xj, colours='r', connect_compositions=True)

        return plot

    @staticmethod
    def pt(flash: DARTSFlash, flash_results: xr.Dataset, composition: list, state: dict = None, plot_phase_fractions: bool = False,
           min_val: float = 0., max_val: float = 1., nlevels: int = 11, critical_point: dict = None,
           logT: bool = False, logP: bool = False, pt_props: xr.Dataset = None, cmap: str = 'winter'):
        """ Method to plot flash results at z """
        # Slice dataset at current state
        comps = {comp: composition[i] for i, comp in enumerate(flash.components[:-1]) if comp in flash_results.dims}
        flash_at_z = flash_results.sel(comps, method='nearest').squeeze()
        np_max = max(int(np.nanmax(flash_at_z.np.values)), 2)
        state = state if state is not None else {"pressure": flash_results.pressure.values,
                                                 "temperature": flash_results.temperature.values}

        # Create Diagram object
        from dartsflash.diagram import Diagram
        nplots = np_max - 1 if plot_phase_fractions else 1
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))
        # plot.add_attributes(suptitle="PT diagram for " + flash.mixture_name)

        if plot_phase_fractions:
            # Plot flash results at feed composition z
            for j in range(np_max - 1):
                plot.subplot_idx = j
                plot.draw_surf(x=state["temperature"], y=state["pressure"], data=flash_at_z.sel(nu_array=j).nu.T.values,
                               ax_labels=["temperature, K", "pressure, bar"], is_float=True, min_val=0., max_val=1.,
                               colours=cmap, colorbar=True, contour=True, fill_contour=True, nlevels=11, logx=logT, logy=logP)
                plot.add_attributes(title="Phase {}".format(j))
        else:
            # Plot np at feed composition z
            plot.subplot_idx = 0

            # if critical_point is not None:
            #     states_to_draw = {spec: arr[arr <= critical_point[spec]] for spec, arr in state.items()}
            # else:
            #     states_to_draw = state

            phase_state_ids = flash_at_z.phase_state_id.T.values
            if critical_point is not None:
                mask = np.zeros(np.shape(phase_state_ids), dtype=bool)
                mask[state["temperature"] > critical_point["temperature"], :] = True
                mask[:, state["pressure"] > critical_point["pressure"]] = True
                phase_state_ids[mask] = -1

            plot.draw_contours(state["temperature"], state["pressure"], phase_state_ids,
                               colours='w' if pt_props is not None else 'k', mask=-1)

            plot.add_attributes(ax_labels=["temperature, K", "pressure, bar"],
                                # suptitle=flash.mixture_name
                                )

            # plot.draw_surf(x=state["temperature"], y=state["pressure"], data=flash_at_z.phase_state_id.T.values,
            #                ax_labels=["temperature, K", "pressure, bar"], is_float=False,
            #                colours=cmap, colorbar=True, contour=False, fill_contour=True, logx=logT, logy=logP)

            if pt_props is not None:
                # Plot enthalpy/entropy
                data = pt_props.sel(comps, method='nearest').squeeze().H_total.T.values
                min_val, max_val = np.nanmin(data), np.nanmax(data)

                plot.draw_surf(x=state['temperature'], y=state['pressure'], data=data,
                               min_val=min_val, max_val=max_val, nlevels=nlevels, contour=True, fill_contour=True,
                               colours='winter', colorbar=True, colorbar_title="enthalpy, kJ/kmol"
                               )

        # Draw critical point
        if critical_point is not None:
            plot.draw_point(X=critical_point["temperature"], Y=critical_point["pressure"],
                            colours='w' if pt_props is not None else 'k')

        return plot

    @staticmethod
    def ph(flash: DARTSFlash, flash_results: xr.Dataset, composition: list, state: dict = None, plot_phase_fractions: bool = True,
           min_temp: float = None, max_temp: float = None, min_val: float = 0., max_val: float = 1., nlevels: int = 11,
           logT: bool = False, logP: bool = False, pt_props: xr.Dataset = None, cmap: str = 'winter'):
        # Slice dataset at current state
        comps = {comp: composition[i] for i, comp in enumerate(flash.components[:-1]) if comp in flash_results.dims}
        flash_at_z = flash_results.sel(comps, method='nearest').squeeze()
        np_max = max(int(np.nanmax(flash_at_z.np.values)), 2)
        state = state if state is not None else {"pressure": flash_results.pressure.values,
                                                 "enthalpy": flash_results.enthalpy.values}

        # Create Diagram object
        from dartsflash.diagram import Diagram
        nplots = 1 + (pt_props is not None)
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))
        # plot.add_attributes(suptitle="PH-diagram of " + flash.mixture_name)

        # Plot temperature
        data = np.array(flash_at_z.temp.T.values, copy=True)
        data[data <= flash.flash_params.T_min + 1e-2] = 750.
        data[data >= flash.flash_params.T_max - 1e-2] = 750.
        plot.draw_surf(x=state['enthalpy'], y=state['pressure'], data=data,
                       min_val=min_temp, max_val=max_temp, nlevels=nlevels, contour=True, fill_contour=True,
                       ax_labels=["enthalpy, kJ/mol", "pressure, bar"],
                       colours='winter', colorbar=True, colorbar_title="temperature, K"
                       )

        # Plot flash results at feed composition z
        if plot_phase_fractions:
            # Plot phase fractions for each phase
            for j in range(np_max - 1):
                data = np.array(flash_at_z.nu.isel(nu_array=j).transpose("enthalpy", "pressure").values, copy=True)
                min_val = np.nanmin(data) if min_val is None else min_val
                max_val = np.nanmax(data) if max_val is None else max_val

                data[data == 0.] = np.nan
                data[data >= 1.] = np.nan

                plot.draw_surf(x=state["enthalpy"], y=state["pressure"], data=data,
                               is_float=True, min_val=min_val, max_val=max_val, nlevels=nlevels,
                               colours='w', contour=True, fill_contour=False, contour_linestyle=plot.linestyles[j],
                               logy=logP)
                # plt.clabel(contours, inline=True, fontsize=8)
        else:
            # Plot np at feed composition z
            plot.subplot_idx = 0
            data = np.array(flash_at_z.np.transpose("enthalpy", "pressure").values, copy=True)
            min_val = np.nanmin(data) if min_val is None else min_val
            max_val = np.nanmax(data) if max_val is None else max_val

            plot.draw_surf(x=state["enthalpy"], y=state["pressure"], data=data, min_val=min_val, max_val=max_val,
                           is_float=False, colours='w', contour=True, fill_contour=False, logy=logP)

        if pt_props is not None:
            plot.subplot_idx += 1
            data = pt_props.H.values * R
            temps = pt_props.temperature.values
            state_spec = {"pressure": state['pressure'],
                          "temperature": np.linspace(np.nanmin(temps), np.nanmax(temps), 100)}

            plot.draw_surf(x=state_spec['temperature'], y=state_spec['pressure'], data=data,
                           colours=cmap, colorbar=True, contour=True, fill_contour=True, logx=logT, logy=logP,
                           ax_labels=["temperature, K", "pressure, bar"]
                           )
            plot.add_attributes(title="Enthalpy, kJ/mol")
        return plot

    @staticmethod
    def compositional(flash: DARTSFlash, flash_results: xr.Dataset, y_var: str, variable_comp_idx: int, dz: float,
                      min_z: list = None, max_z: list = None, state: dict = None, plot_phase_fractions: bool = True,
                      min_temp: float = None, max_temp: float = None, min_val: float = 0., max_val: float = 1., nlevels: int = 11, cmap: str = 'winter'):
        variable_comp = flash.components[variable_comp_idx]
        comp_str = flash.comp_data.comp_labels[variable_comp_idx]

        # Slice dataset at current state
        y = state[y_var]
        ylabel = flash.get_labels[y_var]
        min_z = [0.] if min_z is None else min_z
        max_z = [1.] if max_z is None else max_z
        z0 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)

        comps = {comp: flash_results.dims[comp] for i, comp in enumerate(flash.components[:-1])
                 if comp != variable_comp and comp in flash_results.dims}
        results_at_state = flash_results.sel(state, method='nearest').sel(comps, method='nearest').squeeze()
        np_max = int(np.nanmax(results_at_state.np.values))

        # Create TernaryDiagram object
        from dartsflash.diagram import Diagram
        plot = Diagram(figsize=(8, 5))
        # plot.add_attributes(suptitle="Hx-diagram of " + flash.mixture_name)

        # Plot temperature
        if y_var not in ["temperature", "pressure"]:
            data = results_at_state.temp.transpose(variable_comp, y_var).values
            data[data <= flash.flash_params.T_min + 1e-2] = flash.flash_params.T_min
            data[data >= flash.flash_params.T_max - 1e-2] = flash.flash_params.T_max
            plot.draw_surf(x=z0, y=y, data=data,
                           min_val=min_temp, max_val=max_temp, nlevels=nlevels, contour=True, fill_contour=True,
                           ax_labels=[comp_str, ylabel],
                           colours='winter', colorbar=True, colorbar_title="temperature, K"
                           )

        # Plot flash results at feed composition z
        if plot_phase_fractions:
            # Plot phase fractions for each phase
            for j in range(np_max - 1):
                data = results_at_state.nu.isel(nu_array=j).transpose(variable_comp, y_var).values
                min_val = np.nanmin(data) if min_val is None else min_val
                max_val = np.nanmax(data) if max_val is None else max_val

                data[data == 0.] = np.nan
                data[data >= 1.] = np.nan

                plot.draw_surf(x=z0, y=y, data=data,
                               is_float=True, min_val=min_val, max_val=max_val, nlevels=nlevels,
                               colours='w', contour=True, fill_contour=False, contour_linestyle=plot.linestyles[j], )
                # plt.clabel(contours, inline=True, fontsize=8)
        else:
            # Plot np at feed composition z
            plot.subplot_idx = 0
            phase_state_ids = results_at_state.phase_state_id.transpose(variable_comp, y_var).values
            plot.draw_contours(z0, y, phase_state_ids, colours='k' if y_var in ["temperature", "pressure"] else 'w')
            # plot.subplot_idx = 0
            # plot.draw_surf(x=z0, y=y, data=data,
            #                is_float=False, colours='w', contour=True, fill_contour=False,
            #                min_val=0, max_val=len(flash.flash_params.phase_states_str) - 2,
            #                # colorbar_labels=flash.flash_params.phase_states_str
            #                colorbar_labels=["Aq", "V", "L", "Aq-V", "Aq-L", "V-L", "Aq-V-L"],
            #                )

        return plot

    @staticmethod
    def ps(flash: DARTSFlash, flash_results: xr.Dataset, composition: list, state: dict = None, plot_phase_fractions: bool = True,
           min_temp: float = None, max_temp: float = None, min_val: float = 0., max_val: float = 1., nlevels: int = 11,
           logT: bool = False, logP: bool = False, pt_props: xr.Dataset = None, cmap: str = 'winter'):
        # Slice dataset at current state
        comps = {comp: composition[i] for i, comp in enumerate(flash.components[:-1]) if comp in flash_results.dims}
        flash_at_z = flash_results.sel(comps, method='nearest').squeeze()
        np_max = max(int(np.nanmax(flash_at_z.np.values)), 2)
        state = state if state is not None else {"pressure": flash_results.pressure.values,
                                                 "entropy": flash_results.entropy.values}

        # Create Diagram object
        from dartsflash.diagram import Diagram
        nplots = 1 + (pt_props is not None)
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))
        # plot.add_attributes(suptitle="PS-diagram of " + flash.mixture_name)

        # Plot temperature
        data = np.array(flash_at_z.temp.T.values, copy=True)
        data[data <= flash.flash_params.T_min+1e-2] = 750.
        data[data >= flash.flash_params.T_max-1e-2] = 750.
        plot.draw_surf(x=state['entropy'], y=state['pressure'], data=data, is_float=True,
                       min_val=min_temp, max_val=max_temp, nlevels=nlevels, contour=True, fill_contour=True,
                       ax_labels=["entropy, kJ/mol", "pressure, bar"],
                       colours='winter', colorbar=True, colorbar_title="temperature, K"
                       )

        # Plot flash results at feed composition z
        if plot_phase_fractions:
            # Plot phase fractions for each phase
            for j in range(np_max - 1):
                data = np.array(flash_at_z.nu.isel(nu_array=j).transpose("entropy", "pressure").values, copy=True)
                min_val = np.nanmin(data) if min_val is None else min_val
                max_val = np.nanmax(data) if max_val is None else max_val

                data[data == 0.] = np.nan
                data[data >= 1.] = np.nan

                plot.draw_surf(x=state["entropy"], y=state["pressure"], data=data,
                               is_float=True, min_val=min_val, max_val=max_val, nlevels=nlevels,
                               colours='w', contour=True, fill_contour=False, contour_linestyle=plot.linestyles[j],
                               logy=logP)
                # plt.clabel(contours, inline=True, fontsize=8)
        else:
            # Plot np at feed composition z
            plot.subplot_idx = 0
            data = np.array(flash_at_z.np.transpose("entropy", "pressure").values, copy=True)
            min_val = np.nanmin(data) if min_val is None else min_val
            max_val = np.nanmax(data) if max_val is None else max_val

            plot.draw_surf(x=state["entropy"], y=state["pressure"], data=data, min_val=min_val, max_val=max_val,
                           is_float=False, colours='w', contour=True, fill_contour=False)

        if pt_props is not None:
            plot.subplot_idx += 1
            data = pt_props.S.values * R
            temps = pt_props.temperature.values
            state_spec = {"pressure": state['pressure'],
                          "temperature": np.linspace(np.nanmin(temps), np.nanmax(temps), 100)}

            plot.draw_surf(x=state_spec['temperature'], y=state_spec['pressure'], data=data,
                           colours=cmap, colorbar=True, contour=True, fill_contour=True, logx=logT, logy=logP,
                           ax_labels=["temperature, K", "pressure, bar"]
                           )
            plot.add_attributes(title="Entropy, kJ/mol")
        return plot

    @staticmethod
    def binary_xz(flash: DARTSFlash, flash_results: xr.Dataset, state: dict, variable_comp_idx: int, dz: float,
                  min_z: list = None, max_z: list = None, plot_phase_fractions: bool = True,
                  min_temp: float = None, max_temp: float = None, min_val: float = 0., max_val: float = 1., nlevels: int = 11,
                  pt_props: xr.Dataset = None, cmap: str = 'winter'):
        variable_comp = flash.components[variable_comp_idx]
        comp_str = flash.comp_data.comp_labels[variable_comp_idx]

        # Slice dataset at current state
        y = state["enthalpy"]
        min_z = [0.] if min_z is None else min_z
        max_z = [1.] if max_z is None else max_z
        x = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)

        # Slice dataset at current state
        flash_at_state = flash_results.sel(state, method='nearest').transpose(variable_comp, "enthalpy", ...).squeeze()
        np_max = max(int(np.nanmax(flash_at_state.np.values)), 2)
        state = state if state is not None else {"enthalpy": flash_results.enthalpy.values}

        # Create Diagram object
        from dartsflash.diagram import Diagram
        nplots = 1 + (pt_props is not None)
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))
        plot.add_attributes(suptitle="Hx-diagram of " + flash.mixture_name)

        # Plot temperature
        plot.draw_surf(x=x, y=state['enthalpy'], data=flash_at_state.temp.values,
                       min_val=min_temp, max_val=max_temp, nlevels=nlevels, contour=True, fill_contour=True,
                       ax_labels=[comp_str, "enthalpy, kJ/mol"],
                       colours=cmap, colorbar=True, colorbar_title="temperature, K"
                       )

        # Plot flash results at feed composition z
        if plot_phase_fractions:
            # Plot phase fractions for each phase
            for j in range(np_max - 1):
                data = flash_at_state.nu.isel(nu_array=j).transpose(variable_comp, "enthalpy").values
                min_val = np.nanmin(data) if min_val is None else min_val
                max_val = np.nanmax(data) if max_val is None else max_val

                data[data == 0.] = np.nan
                data[data >= 1.] = np.nan

                plot.draw_surf(x=x, y=state["enthalpy"], data=data,
                               is_float=True, min_val=min_val, max_val=max_val, nlevels=nlevels,
                               colours='w', contour=True, fill_contour=False)
                # plt.clabel(contours, inline=True, fontsize=8)
        else:
            # Plot np at feed composition z
            plot.subplot_idx = 0
            data = flash_at_state.np.transpose(variable_comp, "enthalpy").values
            min_val = np.nanmin(data) if min_val is None else min_val
            max_val = np.nanmax(data) if max_val is None else max_val

            plot.draw_surf(x=x, y=state["enthalpy"], data=data, min_val=min_val, max_val=max_val,
                           is_float=False, colours='w', contour=True, fill_contour=False)

        # if pt_props is not None:
        #     plot.subplot_idx += 1
        #     data = pt_props.S.values * R
        #     temps = pt_props.temperature.values
        #     state_spec = {"pressure": state['pressure'],
        #                   "temperature": np.linspace(np.nanmin(temps), np.nanmax(temps), 100)}
        #
        #     plot.draw_surf(x=state_spec['temperature'], y=state_spec['pressure'], data=data,
        #                    colours=cmap, colorbar=True, contour=True, fill_contour=True,
        #                    ax_labels=["temperature, K", "pressure, bar"]
        #                    )
        #     plot.add_attributes(title="Entropy, kJ/mol")
        return plot


class PlotTPD:
    @staticmethod
    def binary_tpd(flash: DARTSFlash, state: dict, ref_composition: list, variable_comp_idx: int, dz: float,
                   min_z: list = None, max_z: list = None, lnphi_1p: xr.Dataset = None, flash_results: xr.Dataset = None,
                   sp_results: xr.Dataset = None, title: str = None, ax_label: str = None, datalabels: list = None):
        """
        Method to plot binary TPD surfaces.

        :param flash: DARTSFlash object
        :param state: Dictionary of states to plot surfaces for
        :param ref_composition: Reference composition
        :param variable_comp_idx: Index of variable component
        :param dz: Composition interval
        :param min_z: Minimum composition for composition variable, will default to 0
        :param max_z: Maximum composition for composition variable, will default to 1
        :param lnphi_1p: xarray.DataArray of 1P lnphi's
        :param flash_results: xarray.Dataset of flash results
        :param sp_results: xarray.Dataset of stationary points results
        :param title: Title of plot
        :param ax_label: Axes label
        :param datalabels: List of data labels
        """
        variable_comp = flash.components[variable_comp_idx]
        comp_str = flash.comp_data.comp_labels[variable_comp_idx]
        min_z = [0.] if min_z is None else min_z
        max_z = [1.] if max_z is None else max_z
        y1 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)
        state_len = [len(arr) if not np.isscalar(arr) else 1 for arr in state.values()]
        assert state_len[0] == 1 and state_len[1] == 1, "Can't specify array for state specifications"

        # Initialize plot
        from dartsflash.diagram import Diagram
        plot = Diagram(figsize=(8, 5))

        # Prepare tpd calculation: find reference state and y[i] * ln(y[i])
        gmin = None
        for j, eosname in enumerate(flash.eos.keys()):
            lnphi0 = flash.eos[eosname].lnphi(state["pressure"], state["temperature"], ref_composition)
            g = np.sum([ref_composition[i] * lnphi0[i] for i, comp in enumerate(flash.components)])
            if gmin is None or g < gmin:
                gmin = g
                h = np.array([np.log(ref_composition[i]) + lnphi0[i] for i, comp in enumerate(flash.components)])

        y2 = 1. - y1
        y2[y2 < -1e-15] = np.nan
        y2[y2 < 1e-15] = 0.
        ylny = y1 * np.log(y1) + y2 * np.log(y2)
        yh = y1 * h[0] + y2 * h[1]

        # Slice results at current state
        res_at_state = lnphi_1p.sel(state, method='nearest').transpose(..., variable_comp).squeeze()
        minY, maxY = None, None

        # Loop over EoS to plot 1P curves
        for j, eosname in enumerate(flash.eos.keys()):
            lnphi = eval("res_at_state.lnphi_" + eosname + ".values")
            tpd = ylny + y1 * lnphi[0, :] + y2 * lnphi[1, :] - yh

            plot.draw_line(X=y1, Y=tpd, colours=plot.colours[j],
                           datalabels=datalabels[j] if datalabels is not None else None)

            minY = np.nanmin(tpd) if minY is None or np.nanmin(tpd) < minY else minY
            maxY = np.nanmax(tpd) if maxY is None or np.nanmax(tpd) < maxY else maxY

        # Plot stationary points from stability analysis
        if sp_results is not None:
            # Slice results at current state
            sp_at_pt = sp_results.sel(state, method='nearest').squeeze()
            comps = {comp: ref_composition[i] for i, comp in enumerate(flash.components[:-1])}

            # Find stationary points at P,T,z
            sp_at_ptz = sp_at_pt.sel(comps, method='nearest').squeeze()
            Y = sp_at_ptz.y.values
            tot_sp = sp_at_ptz.tot_sp.values
            tpd = sp_at_ptz.tpd.values
            eos_idx = sp_at_ptz.eos_idx.values
            roots = sp_at_ptz.root_type.values

            Yj = np.array([Y[jj * flash.ns:(jj + 1) * flash.ns] for jj in range(tot_sp)])
            tpdj = []
            for j, yj in enumerate(Yj):
                if not np.isnan(yj[0]):
                    eosname = flash.flash_params.eos_order[eos_idx[j]]
                    # flash.eos[eosname].set_root_flag(roots[j])
                    lnphij = flash.eos[eosname].lnphi(state["pressure"], state["temperature"], yj)
                    tpdj.append(np.sum([yj[i] * (np.log(yj[i]) + lnphij[i] - h[i]) for i in range(2)], axis=0))
                else:
                    tpdj.append(np.nan)

            # Plot phase compositions xij at feed composition z
            plot.draw_point(X=Yj[:, 0], Y=tpdj, colours='k', markers="*")

        # Plot equilibrium compositions
        if flash_results is not None:
            # Slice flash results at current state
            flash_at_pt = flash_results.sel(state, method='nearest').squeeze()
            comps = {comp: ref_composition[i] for i, comp in enumerate(flash.components[:-1])}

            # Find flash results at P,T,z
            flash_at_ptz = flash_at_pt.sel(comps, method='nearest').squeeze()
            X = flash_at_ptz.X.values
            eos_idx = flash_at_ptz.eos_idx.values
            roots = flash_at_ptz.root_type.values
            Xj = np.array([X[jj * flash.ns:(jj + 1) * flash.ns] for jj in range(flash.np_max)])

            # Plot phase compositions xij at feed composition z
            yj = []
            for j, xj in enumerate(Xj):
                eosname = flash.flash_params.eos_order[eos_idx[j]]
                # flash.eos[eosname].set_root_flag(roots[j])
                lnphij = flash.eos[eosname].lnphi(state["pressure"], state["temperature"], xj)
                yj.append(np.sum([xj[i] * (np.log(xj[i]) + lnphij[i] - h[i]) for i in range(2)], axis=0))

            plot.draw_point(X=Xj[:, 0], Y=yj, colours='r')
            plot.draw_line(X=Xj[:, 0], Y=yj, colours='r')

        plot.add_attributes(suptitle=title, ax_labels=[comp_str, ax_label], grid=True,
                            legend=(datalabels is not None))
        plot.ax[plot.subplot_idx].set(xlim=[min_z[0], max_z[0]], ylim=[minY, maxY])

        return plot

    @staticmethod
    def ternary_tpd(flash: DARTSFlash, state: dict, ref_composition: list, variable_comp_idx: list, dz: float,
                    min_z: list = None, max_z: list = None, lnphi_1p: xr.Dataset = None, flash_results: xr.Dataset = None,
                    sp_results: xr.Dataset = None, title: str = None):
        """
        Method to plot ternary TPD surfaces.

        :param flash: DARTSFlash object
        :param state: Dictionary of states to plot surfaces for
        :param ref_composition: Reference composition
        :param variable_comp_idx: Indices of variable components
        :param dz: Composition interval
        :param min_z: Minimum composition for composition variables, will default to 0 for each
        :param max_z: Maximum composition for composition variables, will default to 1 for each
        :param lnphi_1p: xarray.DataArray of properties
        :param flash_results: xarray.Dataset of flash results
        :param sp_results: xarray.Dataset of stationary points results
        :param title: Title of plot
        :param cmap: plt.Colormap for plotting
        """
        variable_comps = [flash.components[idx] for idx in variable_comp_idx]
        min_z = [0., 0.] if min_z is None else min_z
        max_z = [1., 1.] if max_z is None else max_z
        x0 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)
        x1 = np.arange(min_z[1], max_z[1] + dz * 0.1, dz)
        state_len = [len(arr) if not np.isscalar(arr) else 1 for arr in state.values()]
        assert state_len[0] == 1 and state_len[1] == 1, "Can't specify array for state specifications"

        # Create TernaryDiagram object
        from dartsflash.diagram import TernaryDiagram
        nplots = len(flash.flash_params.eos_order)
        plot = TernaryDiagram(ncols=nplots, figsize=(nplots * 5 + 3, 5), dz=dz)

        # Prepare tpd calculation: find reference state and y[i] * ln(y[i])
        gmin = None
        for j, eosname in enumerate(flash.eos.keys()):
            lnphi0 = flash.eos[eosname].lnphi(state["pressure"], state["temperature"], ref_composition)
            g = np.sum([ref_composition[i] * lnphi0[i] for i, comp in enumerate(flash.components)])
            if gmin is None or g < gmin:
                gmin = g
                h = np.array([np.log(ref_composition[i]) + lnphi0[i] for i, comp in enumerate(flash.components)])

        y1, y2 = np.meshgrid(x0, x1)
        y3 = 1. - y1 - y2
        y3[y3 < -1e-15] = np.nan
        y3[y3 < 1e-15] = 0.
        y1[y3 < 0.] = np.nan
        y2[y3 < 0.] = np.nan
        y3[y3 < 0.] = np.nan

        ylny = y1 * np.log(y1) + y2 * np.log(y2) + y3 * np.log(y3)
        yh = y1 * h[0] + y2 * h[1] + y3 * h[2]

        # Slice results at current state
        res_at_state = lnphi_1p.sel(state, method='nearest').transpose(..., variable_comps[0], variable_comps[1]).squeeze()

        # Loop over EoS to plot 1P curves
        for j, eosname in enumerate(flash.flash_params.eos_order):
            lnphi = eval("res_at_state.lnphi_" + eosname + ".values")
            tpd = ylny + y1 * lnphi[0, :, :] + y2 * lnphi[1, :, :] + y3 * lnphi[2, :, :] - yh

            plot.subplot_idx = j
            plot.draw_surf(X1=x0, X2=x1, data=tpd, is_float=True, nlevels=10,
                           colours='winter', colorbar=True, contour=True, fill_contour=True,
                           corner_labels=flash.comp_data.comp_labels
                           )
            plot.add_attributes(title=eosname)

        # Plot stationary points from stability analysis
        if sp_results is not None:
            # Slice results at current state
            sp_at_pt = sp_results.sel(state, method='nearest').squeeze()
            comps = {comp: ref_composition[i] for i, comp in enumerate(flash.components[:-1])}

            # Find flash results at P,T,z
            sp_at_ptz = sp_at_pt.sel(comps, method='nearest').squeeze()
            Y = sp_at_ptz.y.values
            tot_sp = sp_at_ptz.tot_sp.values
            tpd = sp_at_ptz.tpd.values
            eos_idx = sp_at_ptz.eos_idx.values
            roots = sp_at_ptz.root_type.values

            Yj = np.array([Y[jj * flash.ns:(jj + 1) * flash.ns] for jj in range(tot_sp)])
            tpdj = []
            for j, yj in enumerate(Yj):
                if not np.isnan(yj[0]):
                    eosname = flash.flash_params.eos_order[eos_idx[j]]
                    # flash.eos[eosname].set_root_flag(roots[j])
                    lnphij = flash.eos[eosname].lnphi(state["pressure"], state["temperature"], yj)
                    tpdj.append(np.sum([yj[i] * (np.log(yj[i]) + lnphij[i] - h[i]) for i in range(2)], axis=0))
                else:
                    tpdj.append(np.nan)

            # Plot phase compositions
            if not np.all(np.isnan(tpdj)):
                plot.draw_compositions(compositions=Yj, colours='g', connect_compositions=False)
            else:
                # For single phase conditions, skip
                pass

        # Plot flash results at specified composition
        if flash_results is not None:
            flash_at_pt = flash_results.sel(state, method='nearest').squeeze()

            # Plot phase compositions xij at feed composition z
            comps = {comp: ref_composition[i] for i, comp in enumerate(flash.components[:-1])}
            plot.draw_compositions(compositions=ref_composition, colours='k')

            # Find flash results at P,T,z
            flash_at_ptz = flash_at_pt.sel(comps, method='nearest').squeeze()
            X = flash_at_ptz.X.values
            Xj = [X[jj * flash.ns:(jj + 1) * flash.ns] for jj in range(flash.np_max)]

            # Plot phase compositions
            if flash_at_ptz.np.values > 1:
                plot.draw_compositions(compositions=Xj, colours='r', connect_compositions=True)
            else:
                # For single phase conditions, skip
                pass

        plot.add_attributes(suptitle=title)

        return plot

    @staticmethod
    def binary_sp(flash: DARTSFlash, tpd_results: xr.Dataset, variable_comp_idx: int, dz: float, state: dict,
                  min_z: list = None, max_z: list = None, cmap: str = 'RdBu'):
        """
        Method to plot P-x and T-x diagrams

        :param flash: DARTSFlash object
        :param flash_results: xarray.DataArray
        :param state: State to plot GE surfaces for
        :param variable_comp_idx: Index of variable component
        """
        variable_comp = flash.components[variable_comp_idx]
        comp_str = flash.comp_data.comp_labels[variable_comp_idx]

        # Slice dataset at current state
        px = hasattr(state["pressure"], "__len__")
        y = state["pressure"] if px else state["temperature"]
        min_z = [0.] if min_z is None else min_z
        max_z = [1.] if max_z is None else max_z
        x0 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)
        zi = tpd_results.variables[variable_comp].values

        comps = {comp: tpd_results.dims[comp] for i, comp in enumerate(flash.components[:-1])
                 if comp != variable_comp and comp in tpd_results.dims}
        results_at_state = tpd_results.sel(state, method='nearest').sel(comps, method='nearest').squeeze()

        # Create TernaryDiagram object
        from dartsflash.diagram import Diagram
        nplots = 1
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))

        # Plot np at feed composition z
        ntpd = results_at_state.neg_sp.values
        plot.draw_surf(zi, y, data=ntpd, is_float=False, colours=cmap, colorbar=True,
                       contour=True, fill_contour=True, min_val=np.nanmin(ntpd), max_val=np.nanmax(ntpd),
                       )
        plot.add_attributes(ax_labels=[comp_str, "pressure, bar" if px else "temperature, K"],
                            suptitle="Number of negative TPD for " + flash.mixture_name +
                                     (" at T = {} K".format(state["temperature"]) if px else " at P = {} bar".format(state["pressure"])))

        return plot

    @staticmethod
    def ternary_sp(flash: DARTSFlash, tpd_results: xr.Dataset, dz: float, state: dict, min_z: list = None, max_z: list = None,
                   composition_to_plot: list = None, cmap: str = 'RdBu'):
        """
        Method to plot TPD

        :param flash: DARTSFlash object
        :param tpd_results:
        :param state:
        :param composition:
        """
        # Slice dataset at current state
        tpd_at_pt = tpd_results.sel(state, method='nearest').squeeze()

        min_z = [0., 0.] if min_z is None else min_z
        max_z = [1., 1.] if max_z is None else max_z
        x0 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)
        x1 = np.arange(min_z[1], max_z[1] + dz * 0.1, dz)

        # Create TernaryDiagram object
        from dartsflash.diagram import TernaryDiagram
        nplots = 1
        plot = TernaryDiagram(ncols=nplots, figsize=(nplots * 5 + 3, 5), dz=dz)
        plot.add_attributes(suptitle="Stationary points for " + flash.mixture_name + " at P = {} bar and T = {} K"
                            .format(state["pressure"], state["temperature"]))

        # Plot number of negative TPD
        ntpd = tpd_at_pt.neg_sp.values
        plot.draw_surf(X1=x0, X2=x1, data=ntpd, is_float=False, colours=cmap, colorbar=True,
                       contour=True, fill_contour=True, min_val=np.nanmin(ntpd), max_val=np.nanmax(ntpd),
                       corner_labels=flash.comp_data.comp_labels
                       )

        # Plot TPD at composition
        if composition_to_plot is not None:
            assert len(composition_to_plot) == flash.ns
            comps = {comp: composition_to_plot[i] for i, comp in enumerate(flash.components[:-1])}

            Y = tpd_at_pt.sel(comps, method='nearest').squeeze().y.values
            Ysp = [Y[j * flash.ns:(j + 1) * flash.ns] for j in range(flash.np_max)]
            plot.draw_compositions(compositions=composition_to_plot, colours='k')
            plot.draw_compositions(compositions=Ysp, colours='r', connect_compositions=True)

        return plot

    @staticmethod
    def pt():
        pass


from dartsflash.hyflash import HyFlash
class PlotHydrate:
    @staticmethod
    def pt(flash: HyFlash, flash_results: xr.Dataset, compositions_to_plot: list, state: dict = None,
           concentrations: list = None, labels: list = None, ref_t: list = None, ref_p: list = None,
           xlim: list = None, ylim: list = None, logx: bool = False, logy: bool = False,
           props: xr.Dataset = None, legend_loc: str = 'upper right'):
        # Slice Dataset at state and composition
        state = state if state is not None else {"pressure": flash_results.pressure.values,
                                                 "temperature": flash_results.temperature.values}
        results_at_state = flash_results.sel(state, method='nearest').squeeze()

        # Initialize Plot object
        from dartsflash.diagram import Diagram
        plot = Diagram(figsize=(8, 5))
        plot.add_attributes(#suptitle=flash.mixture_name + "-hydrate",
                            ax_labels=["temperature, K", "pressure, bar"])

        # Loop over compositions
        comps = {comp: compositions_to_plot[i] for i, comp in enumerate(flash.components[:-1]) if comp in flash_results.dims}
        results_at_comp = results_at_state.sel(comps, method='nearest').squeeze()

        # If multiple salt concentrations have been specified, concatenate
        if not results_at_comp.pressure.values:
            if concentrations is not None:
                X_at_state = [results_at_comp.isel(concentrations=i).pres.values
                              for i, _ in enumerate(concentrations)]
            else:
                X_at_state = results_at_comp.pres.values
        else:
            if concentrations is not None:
                X_at_state = [results_at_comp.isel(concentrations=i).temp.T.values
                              for i, _ in enumerate(concentrations)]
            else:
                X_at_state = results_at_comp.temp.values

        # Plot equilibrium curve data
        if not results_at_comp.pressure.values:
            pressure = X_at_state
            temperature = results_at_comp.temperature.values
        else:
            pressure = results_at_comp.pressure.values
            temperature = X_at_state

        plot.draw_line(X=temperature, Y=pressure, datalabels=labels)
        plot.set_axes(xlim=xlim, ylim=ylim, logx=logx, logy=logy)
        if ref_t is not None or ref_p is not None:
            plot.draw_point(X=ref_t, Y=ref_p)
        plot.add_attributes(legend=labels is not None, legend_loc=legend_loc, grid=True)

        return plot

    @staticmethod
    def binary(flash: HyFlash, flash_results: xr.Dataset, variable_comp_idx: int, dz: float, state: dict,
               min_z: list = None, max_z: list = None, logy: bool = False):
        """
        Method to plot P-x and T-x diagrams

        :param flash: DARTSFlash object
        :param flash_results: xarray.DataArray
        :param state: State to plot GE surfaces for
        :param variable_comp_idx: Index of variable component
        """
        variable_comp = flash.components[variable_comp_idx]
        comp_str = flash.comp_data.comp_labels[variable_comp_idx]

        # Slice dataset at current state
        px = state["pressure"] is None
        y_var = ("pressure" if px else "temperature")
        y = state[y_var]
        min_z = [0.] if min_z is None else min_z
        max_z = [1.] if max_z is None else max_z
        z0 = np.arange(min_z[0], max_z[0] + dz * 0.1, dz)

        comps = {comp: flash_results.dims[comp] for i, comp in enumerate(flash.components[:-1])
                 if comp != variable_comp and comp in flash_results.dims}
        results_at_state = flash_results.sel(state, method='nearest').sel(comps, method='nearest').squeeze()

        # Create TernaryDiagram object
        from dartsflash.diagram import Diagram
        nplots = 1
        plot = Diagram(ncols=nplots, figsize=(nplots * 5 + 3, 5))
        plot.add_attributes(ax_labels=[comp_str, "pressure, bar" if px else "temperature, K"],
                            suptitle="Hydrate equilibrium " + y_var + " for " + flash.mixture_name)

        # Plot phase compositions xij at feed composition z
        if px:
            labels = ["T = {} K".format(t) for t in state["temperature"]]
            data = results_at_state.pres.transpose("temperature", variable_comp).values
        else:
            labels = ["P = {} bar".format(p) for p in state["pressure"]]
            data = results_at_state.temp.transpose("pressure", variable_comp).values

        plot.draw_line(X=z0, Y=data, )
        plot.set_axes(logy=logy, xlim=[min_z[0], max_z[0]])

        return plot
