import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
import matplotlib.tri as tri


class Diagram:
    """
    This is a base class for construction of diagrams.

    :ivar colours: Predefined set of colours
    :type colours: list[str]
    :ivar markers: Predefined set of markers
    :type markers: list[str]
    :ivar linestyles: Predefined set of linestyles
    :type linestyles: list[str]
    :ivar ax_labels: Axis labels
    :type ax_labels: list[str]
    """
    cmap = "winter"
    colours = ["blue", "lightskyblue", "mediumseagreen", "orchid", "dodgerblue", "darkcyan"]
    markers = [None, "--", "o", "v"]
    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    fontsizes = {'suptitle': 20, 'title': 16, 'axlabel': 12, 'axes': 10, 'legend': 12}

    def __init__(self, nrows: int = 1, ncols: int = 1, figsize: tuple = (8, 6), sharex: bool = False,
                 sharey: bool = False):
        """
        Constructor for Diagram base class

        :param nrows, ncols: Number of rows/columns for subplots
        :type nrows, ncols: int
        :param figsize: Size of figure object
        :type figsize: tuple[float]
        :param sharex, sharey: Share axes
        :type sharex, sharey: bool
        """
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey)
        self.ax = [self.ax] if (nrows == 1 and ncols == 1) else self.ax  # make sure we can index self.ax[subplot_idx]
        self.subplot_idx = 0 if (nrows == 1 or ncols == 1) else (0, 0)

        self.im = []

    def get_levels(self, data: np.ndarray, is_float: bool, nlevels: int, min_val: float = None, max_val: float = None):
        # Define levels of contours/patches
        amin = np.nanmin(data) if min_val is None else min_val
        amax = np.nanmax(data) if max_val is None else max_val
        if is_float:
            levels = np.linspace(amin, amax, nlevels)
            ticks = np.linspace(amin, amax, nlevels if nlevels <= 11 else 11)
        else:
            levels = np.linspace(amin - 0.5, amax + .5, amax-amin+2)
            ticks = np.linspace(amin, amax, len(levels)-1)

        return levels, ticks

    def get_cmap(self, levels: np.ndarray, colours: list = None):
        # Get colormap and set discrete colorbar levels
        if isinstance(colours, (list, np.ndarray)):
            cmap = colors.ListedColormap(colours[:len(levels)])
        else:
            try:
                # Specified cmap
                cmap = plt.get_cmap(colours if colours is not None else self.cmap, len(levels))
            except ValueError:
                # Single colour
                cmap = colors.ListedColormap([colours for _ in range(len(levels))])

        level_diff = levels[1] - levels[0] if len(levels) > 1 else 1.
        bounds = np.linspace(levels[0], levels[-1] + level_diff, len(levels) + 1)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        return cmap, norm

    def draw_surf(self, x, y, data: np.ndarray, xlim: list = None, ylim: list = None, ax_labels: list = None,
                  is_float: bool = True, nlevels: int = 10, min_val: float = None, max_val: float = None,
                  colours: list = None, colorbar: bool = False, colorbar_labels: list = None, colorbar_title: str = None,
                  contour: bool = False, fill_contour: bool = False, contour_linestyle: str = None,
                  logx: bool = False, logy: bool = False):
        """
        Function to draw 2D pcolormesh/contourplot.

        :param x: Grid points on x-axis
        :type x: list
        :param y: Grid points on y-axis
        :type y: list
        :param data: :class:`np.ndarray` of data for plotting
        :param xlim, ylim: Limits for axes
        :type xlim, ylim: list
        :param ax_labels: Axis labels
        :param is_float: Type of data is float/integer
        :param nlevels: Number of contour levels for legend
        :type nlevels: int
        :param min_val: Minimum value of contour levels
        :type min_val: float
        :param max_val: Maximum value of contour levels
        :type max_val: float
        :param colours: Colours for ListedColormap
        :type colours: list[str]
        :param colorbar: Switch to add colorbar
        :type colorbar: bool
        :param colorbar_labels: Optional labels for colorbar
        :param colorbar_title: Optional colorbar title
        :type colorbar_title: str
        :param contour: Switch for contour or surface plot
        :type contour: bool
        :param fill_contour: Switch for filled contour plot
        :type fill_contour: bool
        """
        # Define levels of contours/patches and cmap + norm
        levels, ticks = self.get_levels(data, is_float, nlevels, min_val=min_val, max_val=max_val)
        cmap, norm = self.get_cmap(levels, colours)

        nx, ny = len(x), len(y)
        if contour:
            xgrid = np.logspace(np.log10(x[0]), np.log10(x[-1]), nx) if logx else np.linspace(x[0], x[-1], nx)
            ygrid = np.logspace(np.log10(y[0]), np.log10(y[-1]), ny) if logy else np.linspace(y[0], y[-1], ny)
            X, Y = np.meshgrid(xgrid, ygrid)
            z = np.swapaxes(data, 0, 1)

            mask = np.where(np.isnan(z), 1, 0)
            z_ma = np.ma.array(z, mask=mask)

            if fill_contour:
                self.im = self.ax[self.subplot_idx].contourf(X, Y, z_ma, cmap=cmap, norm=norm, levels=levels, vmin=levels[0], vmax=levels[-1], extend='both')
                # if np.any(mask):
                #     self.ax[self.subplot_idx].contourf(X, Y, mask, colors=cmap.colors[0], levels=[0., 1.])
            else:
                self.ax[self.subplot_idx].contour(X, Y, z_ma, cmap=cmap, linestyles=contour_linestyle,
                                                  norm=norm, levels=levels, vmin=levels[0], vmax=levels[-1], extend='both')
                if np.any(mask):
                    self.ax[self.subplot_idx].contour(X, Y, mask, colors=cmap.colors[0], linestyles=contour_linestyle, levels=[0., 1.])
        else:
            dx, dy = x[1] - x[0], y[1] - y[0]
            xgrid = np.linspace(x[0] - 0.5 * dx, x[-1] + 0.5 * dx, nx + 1)
            ygrid = np.linspace(y[0] - 0.5 * dy, y[-1] + 0.5 * dy, ny + 1)
            X, Y = np.meshgrid(xgrid, ygrid)
            z = np.swapaxes(data, 0, 1)
            self.im = self.ax[self.subplot_idx].pcolormesh(X, Y, z, shading='flat', cmap=cmap, norm=norm)

        self.ax[self.subplot_idx].set(xlim=xlim if xlim is not None else [x[0], x[-1]],
                                      ylim=ylim if ylim is not None else [y[0], y[-1]],
                                      xscale='log' if logx else 'linear',
                                      yscale='log' if logy else 'linear',)
        self.add_attributes(ax_labels=ax_labels)
        self.ax[self.subplot_idx].tick_params(axis='both', which='major', labelsize=self.fontsizes['axes'])

        if colorbar:
            cbar = self.fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=self.ax[self.subplot_idx], ticks=ticks)

            if colorbar_labels is not None:
                yticks = np.linspace(*cbar.ax.get_ylim(), cmap.N + 1)[:-1]
                yticks += (yticks[1] - yticks[0]) / 2
                cbar.set_ticks(yticks, labels=colorbar_labels)
                cbar.ax.tick_params(length=0)
            cbar.ax.tick_params(labelsize=self.fontsizes['axes'])

            if colorbar_title is not None:
                cbar.set_label(colorbar_title, size=self.fontsizes['axlabel'])

        return

    def draw_line(self, X: np.ndarray, Y: np.ndarray, colours: list = None, styles: list = None, widths: float = 2,
                  datalabels: list = None):
        """
        Function to draw line with coordinates X-Y(-Z)

        :param X, Y: x- and y-axis values. N-D arrays must be provided as [ith_curve, values]
        :param colours: Colour or list of colours
        :param styles: Line-/Markerstyle or list of styles for line/scatter plot
        :param widths: Line-/Markerwidth or list of widths for line/scatter plot
        :param datalabels: List of datalabels
        """
        X, Y = np.array(X), np.array(Y)
        number_of_curves = np.shape(Y)[0] if (not np.isscalar(Y) and Y.ndim > 1) else np.shape(X)[0] if (not np.isscalar(X) and X.ndim > 1) else 1
        colours = [colours for i in range(number_of_curves)] if not isinstance(colours, (list, np.ndarray, type(None))) else colours
        styles = [styles for i in range(number_of_curves)] if not isinstance(styles, (list, np.ndarray, type(None))) else styles
        widths = [widths for i in range(number_of_curves)] if not isinstance(widths, (list, np.ndarray, type(None))) else widths
        datalabels = [datalabels for i in range(number_of_curves)] if not isinstance(datalabels, (list, np.ndarray, type(None))) else datalabels

        if X.ndim == Y.ndim:
            Y = np.tile(Y, (number_of_curves, 1)) if (not np.isscalar(Y) or not isinstance(Y[0], (list, np.ndarray))) else Y
        X = np.tile(X, (number_of_curves, 1)) if (not np.isscalar(X) or not isinstance(X[0], (list, np.ndarray))) else X

        for i in range(number_of_curves):
            self.ax[self.subplot_idx].plot(X[i][:], Y[i][:],
                                           c=colours[i] if colours is not None else self.colours[int(i % len(self.colours))],
                                           linestyle=styles[i] if styles is not None else self.linestyles[0],
                                           linewidth=widths[i] if widths is not None else None,
                                           label=datalabels[i] if datalabels is not None else None
                                           )
        return

    def draw_point(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray = None, colours: list = None, markers: list = None,
                   widths: list = None, datalabels: list = None):
        """
        Function to draw points with coordinates X-Y(-Z)

        :param X: List of X-coordinates for points
        :param Y: List of Y-coordinates for points
        :param Z: List of Z-coordinates for points, optional
        :param colours: Colour or list of colours
        :param styles: Markerstyle or list of styles for scatter plot
        :param widths: Markerwidth or list of widths for scatter plot
        :param datalabels: List of data labels
        """
        X, Y = np.array(X), np.array(Y)
        number_of_curves = np.shape(Y)[0] if (not np.isscalar(Y) and Y.ndim > 1) else np.shape(X)[0] if (not np.isscalar(X) and X.ndim > 1) else 1
        colours = [colours for i in range(number_of_curves)] if not isinstance(colours, (list, np.ndarray, type(None))) else colours
        markers = [markers for i in range(number_of_curves)] if not isinstance(markers, (list, np.ndarray, type(None))) else markers
        widths = [widths for i in range(number_of_curves)] if not isinstance(widths, (list, np.ndarray, type(None))) else widths
        datalabels = [datalabels for i in range(number_of_curves)] if not isinstance(datalabels, (list, np.ndarray, type(None))) else datalabels

        X = np.array([X]) if np.isscalar(X) else X
        Y = np.array([Y]) if np.isscalar(Y) else Y
        X = np.tile(X, (number_of_curves, 1)) if (not np.isscalar(X) or not isinstance(X[0], (list, np.ndarray))) else X
        Y = np.tile(Y, (number_of_curves, 1)) if (not np.isscalar(Y) or not isinstance(Y[0], (list, np.ndarray))) else Y
        if Z is not None:
            Z = np.array([Z]) if np.isscalar(Z) else np.array(Z)
            Z = np.tile(Z, (number_of_curves, 1)) if (not np.isscalar(Z) or not isinstance(Z[0], (list, np.ndarray))) else Z

        for i in range(number_of_curves):
            self.ax[self.subplot_idx].scatter(X[i][:], Y[i][:],
                                              c=colours[i] if colours is not None else self.colours[int(i % len(self.colours))],
                                              marker=markers[i] if markers is not None else self.markers[0],
                                              s=widths[i] if widths is not None else None,
                                              label=datalabels[i] if datalabels is not None else None)
        return

    def get_contours(self, x, y, data: np.ndarray, mask: float = None):
        # Find boundaries between discrete levels
        contours = {}
        levels = {}
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                if i < len(x) - 1 and data[i, j] != data[i + 1, j]:
                    pair = (min(data[i, j], data[i + 1, j]), max(data[i, j], data[i + 1, j]))
                    if mask is not None and pair[0] <= mask:
                        break

                    key = 0
                    for level in levels.values():
                        if level == pair:
                            break
                        key += 1
                    levels[key] = pair

                    if key in contours.keys():
                        contours[key] += [[(i + 1, i + 1), (j, j + 1)]]
                    else:
                        contours[key] = [[(i + 1, i + 1), (j, j + 1)]]
                if j < len(y) - 1 and data[i, j] != data[i, j + 1]:
                    pair = (min(data[i, j], data[i, j + 1]), max(data[i, j], data[i, j + 1]))
                    if mask is not None and pair[0] <= mask:
                        break

                    key = 0
                    for level in levels.values():
                        if level == pair:
                            break
                        key += 1
                    levels[key] = pair

                    if key in contours.keys():
                        contours[key] += [[(i, i + 1), (j + 1, j + 1)]]
                    else:
                        contours[key] = [[(i, i + 1), (j + 1, j + 1)]]
        return contours, levels

    def draw_contours(self, x, y, data: np.ndarray, mask: float = None, colours: str = None, linewidth: float = 1.):
        """
        Function to draw contour lines between discrete levels

        :param x: Grid points on x-axis
        :type x: list
        :param y: Grid points on y-axis
        :type y: list
        :param data: :class:`np.ndarray` of data for plotting
        :param colours: Colours for contourlines
        :type colours: str
        :param linewidth: Line width
        :type linewidth: float
        """
        # Find boundaries between discrete levels
        contours, levels = self.get_contours(x, y, data, mask=mask)

        # Plot lines at the boundaries
        dx, dy = x[1] - x[0], y[1] - y[0]
        xgrid = np.linspace(x[0] - dx * 0.5, x[-1] + dx * 0.5, len(x) + 1)
        ygrid = np.linspace(y[0] - dy * 0.5, y[-1] + dy * 0.5, len(y) + 1)

        colours = colours if colours is not None else self.colours
        for ith_contour, (level, lines) in enumerate(contours.items()):
            colour = colours if isinstance(colours, str) else colours[ith_contour]
            for line in lines:
                xx = [xgrid[line[0][0]], xgrid[line[0][1]]]
                yy = [ygrid[line[1][0]], ygrid[line[1][1]]]
                self.ax[self.subplot_idx].plot(xx, yy, c=colour, linewidth=linewidth)

        self.set_axes(xlim=[x[0], x[-1]], ylim=[y[0], y[-1]])

        return

    def set_axes(self, xlim: list = None, ylim: list = None, logx: bool = False, logy: bool = False):
        """
        Function to set axes limits and scale

        :param xlim, ylim: Limits of x- and y-axes, default is None
        :param logx, logy: Option to set x- or y-axis to logscale, default is False
        """
        # Set log scale and limits
        if logx:
            self.ax[self.subplot_idx].set_xscale("log")
        if logy:
            self.ax[self.subplot_idx].set_yscale("log")

        self.ax[self.subplot_idx].set_xlim(xlim)
        self.ax[self.subplot_idx].set_ylim(ylim)
        return

    def add_attributes(self, suptitle: str = None, title: str = None, ax_labels: list = None,
                       legend: bool = False, legend_loc: str = 'upper right', grid: bool = False):
        """
        Function to add attributes to diagram.

        :param title: Figure title
        :type title: str
        :param ax_labels: Axes labels
        :type ax_labels: list[str]
        :param legend: Switch to add legend for lines/points
        :type legend: bool
        """
        # add title, axlabels, legend if not None
        if suptitle is not None:
            self.fig.suptitle(suptitle, fontsize=self.fontsizes['suptitle'])

        if title is not None:
            self.ax[self.subplot_idx].set_title(title, fontsize=self.fontsizes['title'])

        if ax_labels is not None:
            self.ax[self.subplot_idx].set_xlabel(ax_labels[0], fontsize=self.fontsizes['axlabel'])
            self.ax[self.subplot_idx].set_ylabel(ax_labels[1], fontsize=self.fontsizes['axlabel'])

        if legend:
            self.ax[self.subplot_idx].legend(loc=legend_loc, fontsize=self.fontsizes['legend'])

        if grid:
            self.ax[self.subplot_idx].grid(True, which='both', linestyle='-.')
            self.ax[self.subplot_idx].tick_params(direction='in', length=1, width=1, colors='k',
                                                  grid_color='k', grid_alpha=0.2, labelsize=self.fontsizes['axes'])

    def add_text(self, text: str, xloc: float, yloc: float, fontsize: float = 8, colours: str = 'k',
                 box_colour: str = 'none'):
        """
        Function to add text to diagram.
        """
        ax = self.ax[self.subplot_idx]
        ax.text(xloc, yloc, text, fontsize=fontsize, transform=ax.transAxes, c=colours,
                bbox=dict(facecolor='none', edgecolor=box_colour, boxstyle='square', mutation_aspect=1.5))


class NCompDiagram(Diagram):
    """
    This is a base class for construction of N-component diagrams.
    """

    def __init__(self, nc: int, dz: float, min_z: list = None, max_z: list = None,
                 nrows: int = 1, ncols: int = 1, figsize: tuple = (10, 10)):
        """
        The constructor will find the set of physical compositions.

        :param nc: Number of components
        :type nc: int
        :param dz: Mesh size of compositions
        :type dz: float
        :param min_z: Minimum composition of each component (i = 1,...,nc-1), optional
        :type min_z: list[float]
        :param max_z: Maximum composition of each component (i = 1,...,nc-1), optional
        :type max_z: list[float]
        :param nrows: Number of rows for subplots
        :type nrows: int
        :param ncols: Number of columns for subplots
        :type ncols: int
        :param figsize: Size of figure object
        :type figsize: tuple[float]
        """
        super().__init__(nrows, ncols, figsize)

        self.min_z = min_z if min_z is not None else [0. for _ in range(nc - 1)]
        self.max_z = max_z if max_z is not None else [1. for _ in range(nc - 1)]
        self.min_z += [max(1.-sum(self.max_z), 0.)]
        self.max_z += [max(1.-sum(self.min_z), 0.)]
        n_points = [int(np.round(((self.max_z[i] - self.min_z[i]) / dz))) + 1 for i in range(nc - 1)]

        comp_bound = np.array([[self.min_z[i], self.max_z[i]] for i in range(nc - 1)])
        comp_vec = [np.linspace(comp_bound[i, 0], comp_bound[i, 1], n_points[i]) for i in range(nc - 1)]
        composition = np.zeros((np.prod(n_points), nc))

        if nc == 2:
            composition[:, 0] = comp_vec[0][:]
        elif nc == 3:
            for ii in range(n_points[0]):
                composition[ii * n_points[1]:(ii + 1) * n_points[1], 0] = comp_vec[0][ii]
                for jj in range(n_points[1]):
                    composition[ii * n_points[1] + jj, 1] = comp_vec[1][jj]
        elif nc == 4:
            for ii in range(n_points[0]):
                composition[ii * n_points[1] * n_points[2]:(ii + 1) * n_points[1] * n_points[2], 0] = comp_vec[0][ii]
                for jj in range(n_points[1]):
                    composition[
                    ii * n_points[1] * n_points[2] + jj * n_points[2]:ii * n_points[1] * n_points[2] + (jj + 1) *
                                                                      n_points[2], 1] = comp_vec[1][jj]
                    for kk in range(n_points[2]):
                        composition[ii * n_points[1] * n_points[2] + jj * n_points[2] + kk, 2] = comp_vec[2][kk]

        composition[:, -1] = 1. - np.sum(composition, 1)
        self.comp_physical = composition[(composition[:, -1] >= -1e-14) * (composition[:, -1] <= 1.+1e-14)]


class TernaryDiagram(NCompDiagram):
    """
    This class can construct ternary diagrams.
    """

    def __init__(self, dz: float, min_z: list = None, max_z: list = None,
                 nrows: int = 1, ncols: int = 1, figsize: tuple = (10, 10)):
        """
        The constructor will find the set of physical compositions for nc=3.

        :param dz: Mesh size of compositions
        :type dz: float
        :param min_z: Minimum composition of each component (i = 1,...,nc-1), optional
        :type min_z: list[float]
        :param max_z: Maximum composition of each component (i = 1,...,nc-1), optional
        :type max_z: list[float]
        :param nrows: Number of rows for subplots
        :type nrows: int
        :param ncols: Number of columns for subplots
        :type ncols: int
        :param figsize: Size of figure object
        :type figsize: tuple[float]
        """
        super().__init__(nc=3, dz=dz, min_z=min_z, max_z=max_z, nrows=nrows, ncols=ncols, figsize=figsize)

        # barycentric coords: (a,b,c)
        self.a = self.comp_physical[:, 0]
        self.b = self.comp_physical[:, 1]
        self.c = self.comp_physical[:, 2]
        self.n_data_points = self.a.shape[0]

    def triangulation(self, X1: np.ndarray, X2: np.ndarray, corner_labels: list = None):
        """
        Function to construct triangular grid and axis.

        :param X1, X2: Composition of 1st and second component
        :type X1, X2: list
        :param corner_labels: Labels at corners

        :returns: Triangular grid and Axes
        :rtype: :class:`matplotlib.tri.Triangulation`, :class:`matplotlib.pyplot.Axes`
        """
        # calculate corner points
        x3min, x3max = max(1.-X1[-1]-X2[-1], 0.), min(1.-X1[0]-X2[0], 1.)
        ymin = max(x3min * np.sqrt(3.)/2., 0.)  # sin(pi/3) = sqrt(3)/2
        ymax = min(x3max * np.sqrt(3.)/2., np.sqrt(3)/2.)
        xmin = max((1.-X1[-1]) - ymin / np.sqrt(3.), 0.)  # tan(pi/3) = sqrt(3)
        xmax = min((1.-X1[0]) - ymin / np.sqrt(3.), 1.)
        xmid = (xmin + xmax) / 2.
        self.corners = np.array([[xmin, ymin], [xmax, ymin], [xmid, ymax]])
        triangle = tri.Triangulation(self.corners[:, 0], self.corners[:, 1])

        # plotting the mesh
        self.ax[self.subplot_idx].triplot(triangle, color='k', linewidth=0.5)
        self.ax[self.subplot_idx].set_ylim([ymin, ymax*1.1])
        self.ax[self.subplot_idx].axis('off')
        self.ax[self.subplot_idx].set_aspect('equal')

        # translate the data to cartesian coords
        self.data_idxs = [[(x1 + x2 <= 1. + 1e-14) for x2 in X2] for x1 in X1]
        self.x = 0.5 * (2. * self.b + self.c) / (self.a + self.b + self.c)
        self.y = 0.5 * np.sqrt(3) * self.c / (self.a + self.b + self.c)

        # create a triangulation out of these points
        T = tri.Triangulation(self.x, self.y)

        # labels at corner points
        if corner_labels is not None:
            self.ax[self.subplot_idx].text(self.corners[0, 0] - 0.015 * (xmax-xmin), self.corners[0, 1]-0.025 * (ymax-ymin),
                                           corner_labels[0], fontsize=self.fontsizes['axlabel'], horizontalalignment='right')
            self.ax[self.subplot_idx].text(self.corners[1, 0] + 0.015 * (xmax-xmin), self.corners[1, 1]-0.025 * (ymax-ymin),
                                           corner_labels[1], fontsize=self.fontsizes['axlabel'], horizontalalignment='left')
            self.ax[self.subplot_idx].text(self.corners[2, 0], self.corners[2, 1] + (ymax-ymin)*0.03,
                                           corner_labels[2], fontsize=self.fontsizes['axlabel'], horizontalalignment='center')

        return T

    def draw_surf(self, X1, X2, data: np.ndarray, corner_labels: list = None, xlim: list = None, ylim: list = None,
                  is_float: bool = True, nlevels: int = 10, min_val: float = None, max_val: float = None,
                  colours: list = None, colorbar: bool = False, colorbar_labels: list = None, colorbar_title: str = None,
                  contour: bool = False, fill_contour: bool = False):
        """
        Function to draw ternary pcolormesh/contourplot.

        :param X1, X2: Compositions of first and second components
        :type X1, X2: list
        :param data: :class:`np.ndarray` of data for plotting
        :param corner_labels: Labels of ternary diagram corners
        :param xlim, ylim: Limits for axes
        :type xlim, ylim: list
        :param ax_labels: Axis labels
        :param is_float: Type of data is float/integer
        :param nlevels: Number of contour levels for legend
        :type nlevels: int
        :param min_val: Minimum value of contour levels
        :type min_val: float
        :param max_val: Maximum value of contour levels
        :type max_val: float
        :param colours: Colours for ListedColormap
        :type colours: list[str]
        :param colorbar: Switch to add colorbar
        :type colorbar: bool
        :param colorbar_labels: Labels for colorbar
        :param colorbar_title: Optional colorbar title
        :type colorbar_title: str
        :param contour: Switch for contour or surface plot
        :type contour: bool
        :param fill_contour: Switch for filled contour plot
        :type fill_contour: bool
        """
        # Define levels of contours/patches and cmap + norm
        levels, ticks = self.get_levels(data, is_float, nlevels, min_val, max_val)
        cmap, norm = self.get_cmap(levels, colours)

        # Create triangular
        T = self.triangulation(X1, X2, corner_labels)

        # plot the contour, mask the nan or inf data points
        plot_data = data[self.data_idxs].flatten()
        point_mask = ~np.isfinite(plot_data)  # Points to mask out.
        tri_mask = np.any(point_mask[T.triangles], axis=1)  # Triangles to mask out.
        T.set_mask(tri_mask)
        if contour:
            plot_method = self.ax[self.subplot_idx].tricontourf if fill_contour else self.ax[self.subplot_idx].tricontour
            self.im = plot_method(self.x, self.y, T.triangles, plot_data, mask=tri_mask,
                                  cmap=cmap, norm=norm, vmin=levels[0], vmax=levels[-1], extend="both")  # extend='max'
        else:
            self.im = self.ax[self.subplot_idx].tripcolor(self.x, self.y, T.triangles, plot_data, mask=tri_mask,
                                                          shading='flat', cmap=cmap, norm=norm)

        if colorbar:
            cax = self.ax[self.subplot_idx].inset_axes([0.85, 0.5, 0.055, 0.3])
            cbar = self.fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=self.ax[self.subplot_idx], cax=cax)

            if colorbar_labels is not None:
                yticks = np.linspace(*cbar.ax.get_ylim(), cmap.N + 1)[:-1]
                yticks += (yticks[1] - yticks[0]) / 2
                cbar.set_ticks(yticks, labels=colorbar_labels)
                cbar.ax.tick_params(length=0)
            cbar.ax.tick_params(labelsize=self.fontsizes['axes'])

            if colorbar_title is not None:
                cbar.set_label(colorbar_title, size=self.fontsizes['axlabel'])

        return

    def draw_compositions(self, compositions: list, axes: list = None, colours: str = None, markerstyle: str = None,
                          linestyle: str = None, connect_compositions: bool = False):
        """
        Function to draw compositions in ternary plot.

        :param compositions: Compositions of end points [[x0, y0, z0], [x1, y1, z1], ...]
        :param colours: Marker/line colour, optional
        :param markerstyle: Point markerstyle, optional
        :param linestyle: Linestyle, optional
        :param connect_compositions: Switch to connect compositions with a line
        """
        # Calculate mole fractions
        compositions = np.array([compositions]) if not hasattr(compositions[0], "__len__") else np.array(compositions)
        compositions = np.array([comp / np.sum(comp) if np.nansum(comp) else np.ones(3) * np.nan for comp in compositions])
        compositions_scaled = (compositions - np.array(self.min_z)) / (np.array(self.max_z) - np.array(self.min_z))
        inside_ternary = np.all(compositions_scaled >= 0., axis=1) * np.all(compositions_scaled <= 1., axis=1)

        # translate the data to cords
        y = compositions_scaled[:, 2] * np.sqrt(3.) / 2.  # sin(pi/3) = sqrt(3)/2
        x = (1. - compositions_scaled[:, 0]) - y / np.sqrt(3.)  # tan(pi/3) = sqrt(3)
        for j, xj in enumerate(compositions):
            axj = axes[j] if axes is not None else 0
            self.ax[axj].scatter(x[j], y[j], color=colours, marker=markerstyle)

        if connect_compositions:
            # Loop over compositions
            # Find number of equilibrium phases, max number is nc
            ph_idxs = [j for j, comp in enumerate(compositions) if not np.isnan(comp[0])]
            ncomp = len(ph_idxs)

            # If both within ternary bounds, plot both directly
            # If one outside, find intersection with edge of (scaled) ternary diagram
            # If both outside, pass
            for i, (i1, i2) in enumerate([(ph_idxs[ii], ph_idxs[ii+1]) for ii in range(ncomp-1)] + [(ph_idxs[-1], ph_idxs[0])]):
                if inside_ternary[i1] and inside_ternary[i2]:
                    for ax in self.ax:
                        ax.plot([x[i1], x[i2]], [y[i1], y[i2]], color=colours, linestyle=linestyle)
                elif inside_ternary[i1] or inside_ternary[i2]:
                    # Find intersection with edge of the (scaled) ternary diagram
                    (in_idx, out_idx) = (i1, i2) if inside_ternary[i1] else (i2, i1)
                    if not np.isnan(compositions[out_idx, 0]):
                        zero_comp = np.where(compositions_scaled[out_idx, :] < 0.)[0][0]
                        frac = (compositions_scaled[in_idx, zero_comp] /
                                (compositions_scaled[in_idx, zero_comp] - compositions_scaled[out_idx, zero_comp]))
                        x_out, y_out = x[in_idx] + frac * (x[out_idx]-x[in_idx]), y[in_idx] + frac * (y[out_idx]-y[in_idx])
                        for ax in self.ax:
                            ax.plot([x[in_idx], x_out], [y[in_idx], y_out], color=colours, linestyle=linestyle)
                else:
                    pass

            # # Connect last composition with the first
            # if inside_ternary[0] and inside_ternary[-1]:
            #     self.ax[self.subplot_idx].plot([x[0], x[-1]], [y[0], y[-1]], color=colours, linestyle=linestyle)
            # elif inside_ternary[0] or inside_ternary[-1]:
            #     # Find intersection with edge of the (scaled) ternary diagram
            #     (in_idx, out_idx) = (0, -1) if inside_ternary[0] else (-1, 0)
            #     if not np.isnan(compositions[out_idx, 0]):
            #         zero_comp = np.where(compositions_scaled[out_idx, :] < 0.)[0][0]
            #         frac = (compositions_scaled[in_idx, zero_comp] /
            #                 (compositions_scaled[in_idx, zero_comp] - compositions_scaled[out_idx, zero_comp]))
            #         x_out, y_out = x[in_idx] + frac * (x[out_idx] - x[in_idx]), y[in_idx] + frac * (
            #                     y[out_idx] - y[in_idx])
            #         self.ax[self.subplot_idx].plot([x[in_idx], x_out], [y[in_idx], y_out], color=colours,
            #                                        linestyle=linestyle)

        return

    def draw_contours(self, X1, X2, data: np.ndarray, mask: float = None, xlim: list = None, ylim: list = None,
                      corner_labels: list = None, colours: str = None, linewidth: float = 1.):
        """
        Function to draw contour lines between discrete levels

        :param X1: Grid points on x-axis
        :type X1: list
        :param X2: Grid points on y-axis
        :type X2: list
        :param data: :class:`np.ndarray` of data for plotting
        :param colours: Colours for contourlines
        :type colours: str
        :param linewidth: Line width
        :type linewidth: float
        """
        # Find boundaries between discrete levels
        contours, levels = self.get_contours(X1, X2, data, mask=mask)

        # translate the data to coords
        sqrt3 = np.sqrt(3.)
        nx, ny = len(X1)+1, len(X2)+1
        dx, dy = 1./nx, 1./ny
        # X1[1]-X1[0], X2[1]-X2[0]

        # Create triangular
        X = np.linspace(0., 1., nx-1)
        Y = np.linspace(0., 1., ny-1)
        T = self.triangulation(X, Y, corner_labels)

        x = np.linspace(- dx * 0.5, 1. + dx * 0.5, nx)
        y = np.linspace(- dy * 0.5, 1. + dy * 0.5, ny)

        # Plot lines at the boundaries
        colours = colours if colours is not None else self.colours
        for ith_contour, (level, lines) in enumerate(contours.items()):
            colour = colours if isinstance(colours, str) else colours[ith_contour]
            for line in lines:
                # Plot line segments and translate them to triangular space
                xx = np.array([x[line[0][0]], x[line[0][1]]])
                yy = np.array([y[line[1][0]], y[line[1][1]]])

                yy = (1 - xx - yy) * sqrt3 / 2.  # sin(pi/3) = sqrt(3)/2
                xx = 1. - xx - yy / sqrt3  # tan(pi/3) = sqrt(3)

                self.ax[self.subplot_idx].plot(xx, yy, c=colour, linewidth=linewidth)

        self.set_axes(xlim=[X[0], X[-1]], ylim=[Y[0], Y[-1]])

        return


class QuaternaryDiagram(NCompDiagram):
    """
    This class can construct quaternary diagrams.
    """

    def __init__(self, dz, min_z: list = None, max_z: list = None,
                 nrows: int = 1, ncols: int = 1, figsize: tuple = (10, 10)):
        """
        The constructor will find the set of physical compositions for nc=4.

        :param dz: Mesh size of compositions
        :type dz: float
        :param min_z: Minimum composition of each component (i = 1,...,nc-1), optional
        :type min_z: list[float]
        :param max_z: Maximum composition of each component (i = 1,...,nc-1), optional
        :type max_z: list[float]
        :param nrows: Number of rows for subplots
        :type nrows: int
        :param ncols: Number of columns for subplots
        :type ncols: int
        :param figsize: Size of figure object
        :type figsize: tuple[float]
        """
        super().__init__(nc=4, dz=dz, min_z=min_z, max_z=max_z, nrows=nrows, ncols=ncols, figsize=figsize)

        # barycentric coords: (a,b,c)
        self.a = self.comp_physical[:, 0]
        self.b = self.comp_physical[:, 1]
        self.c = self.comp_physical[:, 2]
        self.d = self.comp_physical[:, 3]
        self.n_data_points = self.a.shape[0]

    def quaternary(self):
        return
