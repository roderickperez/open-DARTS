"""
@author: adaniilidis

Economic model with pandas accessor for coupling with output from DARTS simulator

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

component = ['Pump (injection or production)', 'VSD', 'christmas tree', 'degasser', 'filter (candle + bag)', 'Heat exchanger', 'CHP']
capex_components = [350000, 150000, 125000, 500000, 2250, 1000000, 800000]
replace = [4., 0., 0., 0., 0.16666667,0., 0.]


@pd.api.extensions.register_dataframe_accessor("gte")
class GTEcon:
    def __init__(self, pandas_obj):
        # self._validate(pandas_obj)
        self._obj = pandas_obj



    @staticmethod
    def _validate(obj):
        """Verify the needed columns are present in the dataframe"""
        for col in ['Time (yrs)']:

            if col not in obj.columns:
                raise AttributeError('\nThe dataframe does not include collumn: %s\nThis does not allow further analysis.'%col )

    def init(self, wells_names, wells_top_tvd, wells_mid_tvd, wells_bottom_tvd):
        """
        Initialize instance py providing well names and TVD
        """
        self.wells_names = wells_names
        self.wells_top_tvd = wells_top_tvd
        self.wells_mid_tvd = wells_mid_tvd
        self.wells_bottom_tvd = wells_bottom_tvd


    def regularize(self, set_interval='200D'):
        """
        Regularize irregular time axis for economic assessments

        check also:
        https://stackoverflow.com/questions/49191998/pandas-dataframe-resample-from-irregular-timeseries-index?noredirect=1&lq=1
        """
        # assing datetime index
        self._obj['Datetime'] = pd.to_datetime('1/1/2000') + pd.to_timedelta(self._obj['Time (yrs)']*pd.Timedelta(('365.25 days')), unit='s')
        self._obj.set_index('Datetime', inplace=True)

        # generate the new index on which the data should correspond based on original index range
        # resample_index = pd.date_range(self._obj.index[0]-pd.Timedelta(set_interval*5), self._obj.index[-1]+pd.Timedelta(set_interval*5), freq=set_interval, normalize=True, closed='right')
        resample_index = pd.date_range(self._obj.index[0] - pd.Timedelta(set_interval),
                                       self._obj.index[-1] + pd.Timedelta(set_interval) , freq=set_interval, normalize=True, closed='right')

        # create a new index as the union of the old and new index and interpolate on the combined index
        # then reindex the dataframe to the new index
        temp = self._obj.reindex(self._obj.index.union(resample_index)).interpolate('index').reindex(resample_index)

        # maintain initial values of original dataset in the newly indexed one
        temp.iloc[0] = self._obj.iloc[0]
        temp.iloc[-1] = self._obj.iloc[-1]

        # add the number of periods for further economic assessment
        temp['econ_periods'] = np.arange(len(temp))

        # recompute time columns based on index
        temp['Time (yrs)'] = pd.to_timedelta(temp.index- temp.index[0]).total_seconds() / (365.25 * 24 * 60 * 60)

        # reset index to time delta
        temp.set_index(pd.to_timedelta(temp['Time (yrs)']*pd.Timedelta(('365.25 days')), unit='s'), inplace=True)
        temp.drop(columns='time', inplace=True)

        return(temp)

    def power(self):#, exploration_wells = ['E01', 'E02', 'E03', 'E04', 'E05', 'E06'], remove_exploration_wells=False):

        """
        Calculate the power of the system
        :return: power in MW
        """
        kJ_day_MWh_day = 1/3.6e6
        
        # # make a copy of the df and remove the exploration wells
        temp = self._obj.copy()
        # if remove_exploration_wells:
        #     drop_cols = [x + ' : energy (KJ/day)' for x in exploration_wells]
        #     temp.drop(drop_cols, axis=1, inplace=True)
        energy_cols_names = temp.filter(like='KJ/day').columns.tolist()
        # energy_cols = temp.filter(like='KJ/day')
        # inj_energy = [x for x in energy_cols.columns.tolist() if energy_cols[x].mean()>0]
        # prod_energy = [x for x in energy_cols.columns.tolist() if energy_cols[x].mean()<0]


        self._obj['Deltahours'] = self._obj.index.to_series().diff(1) / pd.Timedelta('1 hour')

        # self._obj['Power injection (MW)'] = self._obj[inj_energy].sum(axis=1) * kJ_day_MWh_day / 24
        # self._obj['Power production (MW)'] = - self._obj[prod_energy].sum(axis=1) * kJ_day_MWh_day / 24
        # self._obj['Power net (MW)'] = self._obj['Power production (MW)'] - self._obj['Power injection (MW)']
        self._obj['Power net (MW)'] = abs(self._obj[energy_cols_names].sum(axis=1)) * kJ_day_MWh_day / 24

        # assign initial Power to nan
        self._obj.loc[self._obj.index == self._obj.index[0], 'Power net (MW)'] = np.nan

    def pump_power(self, operated_wells_names=['I1', 'I2', 'I3', 'P1', 'P2', 'P3'],
                   operated_wells_mid_tvd = [2300, 2300, 2300, 2300, 2300, 2300],
                   pressure_grad_MPa = 10,
                   pump_efficiency = 0.5):
        """
        Compute the pumping power required based for each operated well
        """
        self.operated_wells = len(operated_wells_names)
        self.operated_well_names = operated_wells_names
        self.operated_well_mid_tvd = operated_wells_mid_tvd
        m3_day_m3_sec = 1 / (24 * 60 * 60)

        # Calculate the required pumping power for the system
        # if the inputs are in MPa the result is also in MW (1e6 factor)
        # self._obj['Pump power total (MW)'] = 0
        for i, well in enumerate(operated_wells_names):
            self._obj['%s : Pump dp (MPa)'%well] = abs(self._obj['%s : BHP (bar)'%well] * 0.1 - operated_wells_mid_tvd[i] / 1000 * pressure_grad_MPa + 1)
            self._obj['%s : Pump power (MW)' % well] = abs(self._obj['%s : Pump dp (MPa)'% well]) * \
                             abs(self._obj['%s : water rate (m3/day)'% well] * m3_day_m3_sec * pump_efficiency)

            # self._obj['Pump power total (MW)'] += abs(self._obj['%s : Pump power (MW)' % well])
            # self._obj['COP (-)'] = self._obj['Power net (MW)'] / self._obj['Pump power total (MW)']

        self._obj['Pump power total (MW)'] = self._obj[self._obj.filter(like=': Pump power (MW)').columns.tolist()].sum(axis=1)
        self._obj['COP (-)'] = self._obj['Power net (MW)'] / self._obj['Pump power total (MW)']


    #@property
    def lcoh(self, drilled_wells_depths=[2300, 2300, 2300, 2300, 2300, 2300],
             operated_wells_names = None,
             well_connection_dist_dict = None,
             heat_price = 50, electricity_price = 100,
             pump_cost = 5e5, pump_replace = 5, annual_OpEx_percent_CapEx = 0.05,
             annual_discount_rate = 0.05,
             surface_piping_cost_per_m = 600,
             drop_npv=True,
             verbose=False):

        """
        :param wells_depths: list of floats
            well depths to be used for the computation of drilling costs
        :param electricity_price: float
            price of electricity to run the pumps in Euro/MWh
        :param pump_replace: float
            frequency of replacing pumps in years
        :param pump_cost: float
            cost of purchasing each pump in euros
        :param heat_price: float
            price for produced heat in Euro/MWh
        :return:
        """
        self.drilled_wells = len(drilled_wells_depths)
        if not operated_wells_names:
            operated_wells_names = drilled_wells_depths

        # Calculate appropriate disount rate and OpEx rate for period
        interval = self._obj.index.to_series().diff(1).mode()[0]
        ratio_year = pd.Timedelta(interval)/pd.Timedelta('365.25 days')
        periodic_discount_rate = round(((1+annual_discount_rate)**ratio_year)-1, 12)
        periodic_OpEx_rate = round(((1+annual_OpEx_percent_CapEx)**ratio_year)-1, 12)

        # Calculate the deltahours intervals
        self._obj['Deltahours'] = self._obj.index.to_series().diff(1) /  pd.Timedelta('1 hour')

        # Compute the produced energy
        self._obj['Produced Energy (MWh)'] = (self._obj['Deltahours'] * self._obj['Power net (MW)']).fillna(0)

        # Compute the cumulative produced energy
        self._obj[r'CumProduced Energy (MWh) $\times$ $10^6$'] = (self._obj['Produced Energy (MWh)'].cumsum())*1e-6
        # print(self._obj[r'CumProduced Energy (MWh) $\times$ $10^6$'].iloc[-1])

        # Compute the generated income
        self._obj['Income (\u20ac)'] = self._obj['Produced Energy (MWh)'] * heat_price

        # self._obj['Test'] = 0

        # Compute the CapEX
        wellcosts = sum([drillingcostnl(depth) for depth in drilled_wells_depths])
        pumpcosts = pump_cost*len(operated_wells_names)

        # # find the nearest time position in days for the initial and recuring costs
        # capex_re_time = self._obj.index.get_loc(pd.Timedelta(365.25 * pump_replace, unit='d'), method='nearest')

        # assign initial CapEx
        self._obj['CapEx (\u20ac)'] = 0
        self._obj.loc[self._obj.index == self._obj.index[0], 'CapEx (\u20ac)'] = wellcosts

        # assign equipment CapEx to initial and recurring moments
        for k in range(len(operated_wells_names)):
            if verbose:
                print('\nFor operated well %s:'%operated_wells_names[k])
            for i in range(len(capex_components)):
                self._obj.gte.recurring_costs(recurring_cost=capex_components[i],
                                              recurring_frequency=replace[i])
                if verbose:
                    print('\t added cost component: %s \n\t\tcost %s \n\t\trecurring every %s years' % (component[i], capex_components[i], round(replace[i], 2)))

            self._obj.gte.recurring_costs(recurring_cost=well_connection_dist_dict.get(operated_wells_names[k]) * surface_piping_cost_per_m,
                                          recurring_frequency=0)
            if verbose:
                print('\t added cost component: %s m surface piping to demand site \n\t\tcost %s \n\t\trecurring every %s years' % (
            well_connection_dist_dict.get(operated_wells_names[k]),
            well_connection_dist_dict.get(operated_wells_names[k]) * surface_piping_cost_per_m, 0 ))

        # calculate OpEx costs from CapEx
        self._obj['OpEx (\u20ac)'] = self._obj['CapEx (\u20ac)'].cumsum() * periodic_OpEx_rate

        # calculate OpEx costs for pumping
        self._obj['OpEx_pump (\u20ac)'] = self._obj['Pump power total (MW)'] * self._obj['Deltahours'] * electricity_price

        # calculate CashFlow
        self._obj['CF (\u20ac)'] = - self._obj['CapEx (\u20ac)'].fillna(0) \
                                   - self._obj['OpEx (\u20ac)'].fillna(0) \
                                   - self._obj['OpEx_pump (\u20ac)'].fillna(0) \
                                   + self._obj['Income (\u20ac)'].fillna(0)

        # calculate NPV
        NPVmulti = '1e-6'

        self._obj[r'NPV (€) $\times$ $10^%s$'%NPVmulti[-1]] = (self._obj['CF (\u20ac)'] /
                                     (1+periodic_discount_rate)**self._obj['econ_periods']).cumsum() * float(NPVmulti)
        if drop_npv:
            self._obj.drop(columns=[r'NPV (€) $\times$ $10^%s$'%NPVmulti[-1]], inplace=True)

        # calculate LCOH
        self._obj['LCOH costs'] = self._obj['CapEx (\u20ac)'].fillna(0) \
                                  + self._obj['OpEx (\u20ac)'].fillna(0) \
                                  + self._obj['OpEx_pump (\u20ac)'].fillna(0)

        self._obj['discounted LCOH costs'] = (self._obj['LCOH costs'] /
                                              (1 + periodic_discount_rate) ** self._obj[
                                                  'Time (yrs)']).cumsum()

        self._obj['discounted  LCOH energy'] = (self._obj['Produced Energy (MWh)'].cumsum() /
                                                (1 + periodic_discount_rate) ** self._obj[
                                                    'econ_periods']).cumsum()

        self._obj[r'LCOH (€/MWh)'] = self._obj['discounted LCOH costs'] / self._obj['discounted  LCOH energy']


    def overview_plot(self):
        """
        Plot some model results for a quick overview
        """


        fig, ax = plt.subplots(2,3, figsize=(14,10), dpi=250, sharex=True)
        ax_list = fig.axes

        for i, key in enumerate(['temperature', 'BHP','Pump power', 'Power', 'COP',  'LCOH (€/MWh)']):
            # injwells = [x for x in econ.filter(like=key).columns.tolist() if 'I' in x]
            # prodwells = [x for x in econ.filter(like=key).columns.tolist() if 'P' in x]
            self._obj.plot(x='Time (yrs)',y=sorted(self._obj.filter(like=key).columns.tolist()), ax=ax_list[i], legend=True)
            ax_list[i].set_ylabel('%s %s'%(key, self._obj.filter(like=key).columns.tolist()[0].split(' ')[-1]))
            ax_list[i].grid(alpha=0.3)
            ax_list[i].legend(labels=[lab.split(':')[0].split('(')[0] for lab in ax_list[i].get_legend_handles_labels()[1]],
                      frameon=False, fontsize=5, ncol=2)

            if key == 'LCOH (€/MWh)':
                ax_list[i].annotate('%0.2f @ time: %.2f yrs' % (self._obj[key].iloc[-1],self._obj['Time (yrs)'].iloc[-1]),
                                    xy=(0.5, 0.5), va='center',ha='center',#xytext=(0.5, 0.5),
                                    xycoords=('axes fraction'))#, textcoords='offset points')

                ax_list[i].annotate('wells\n drilled: %.f\n operated: %.f' % (self.drilled_wells, self.operated_wells),
                                    xy=(0.95, 0.75), va='center', ha='right',  # xytext=(0.5, 0.5),
                                    xycoords=('axes fraction'), fontsize=6)  # , textcoords='offset points')
        [ax_list[k].tick_params(axis="both", which="both", bottom=False, top=False,
                                labelbottom=True, left=False, right=False, labelleft=True) for k in
         range(len(ax_list))]

        [ax_list[k].spines[location].set_linewidth(0) for location in ['top', 'bottom', 'left', 'right'] for k in
         range(len(ax_list))]
        plt.show()

    def recurring_costs(self, recurring_cost = 1000,recurring_frequency=2):#, ,recurring_frequency='365D'):
        """
        Add CapEx costs to cash flow with a recurring interval

        recurring_cost: float
            component cost in euro

        recurring_frequency: float
            interval of recurrence in years

        :return: series
            costs in euros

        """

        # find the nearest time position in days for the recuring costs
        # recurrence_index_time = self._obj.index.get_loc(pd.Timedelta(recurring_frequency), method='nearest')
        recurrence_index_time = self._obj.index.get_loc(pd.Timedelta(365.25 * recurring_frequency, unit='d'), method='nearest')
        # print(recurrence_index_time)


        # assign CapEx to inital cashflow
        self._obj.loc[self._obj.index == self._obj.index[0], 'CapEx (\u20ac)'] += recurring_cost

        # if recurring assign recurring costs
        if recurring_frequency != 0 :
            self._obj.loc[::recurrence_index_time, 'CapEx (\u20ac)'] += recurring_cost



def drillingcostnl(depth):
    """
    Calculate the cost of drilling as a function of depth
    Reference source:
        https://www.thermogis.nl/en/economic-model

    :param depth: float
        measured depth along hole in meters

    :return: float
        costs in euros
    """
    drilling_cost_nl = 375000 + 1150 * depth + 0.3 * depth ** 2
    return(drilling_cost_nl)

