import numpy as np
from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer
from darts.engines import value_vector, sim_params, well_control_iface


#from re import L
#from darts.models.cicd_model import CICDModel
#from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
#from darts.tools.keyword_file_tools import load_single_keyword
#from darts.physics.properties.iapws.iapws_property import *
#from darts.physics.properties.basic import ConstFunc





class Model(DartsModel):

    def __init__(self, n_points=64):


        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.set_reservoir()         # Se llama la funcion "set_reservoir" (definida mas abajo)
                                     # con la cual se establecen las caracteristicas el yacimiento
        self.set_physics(n_points)   # Se llama la funcion "set_physics" (definida mas abajo)
                                     # con la cual se establece la fisica del problema

                                  # 1 [s]= 1.1574e-5 [days]
        self.set_sim_params(first_ts=1.1574e-5, mult_ts=2, max_ts=100, runtime=3650, tol_newton=1e-2, tol_linear=1e-4,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))

        self.timer.node["initialization"].stop()

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal

        print('Reservoir physics: Geothermal')

        self.min_p=0.                #  bar
        self.max_p=500.              #  bar
        self.min_e= 50.*18.01528      # KJ/kg ---- KJ/Kmol
        self.max_e= 3200.*18.01528   # KJ/kg ---- KJ/Kmol

        self.physics = Geothermal(self.timer, n_points=n_points,
                                  min_p=self.min_p, max_p=self.max_p,
                                  min_e=self.min_e, max_e=self.max_e, cache=False)

        print('\n')


        property_container = PropertyContainer()

        property_container.output_props = {'temp [K]': lambda: property_container.temperature,
                                           'temp [°C]': lambda: property_container.temperature-273.15,
                                           'sat_W': lambda: property_container.saturation[0],
                                           'sat_S': lambda: property_container.saturation[1],
                                        #     'Enthalpy_W [KJ/kmol]': lambda: property_container.enthalpy[0],
                                        #     'Enthalpy_S [KJ/kmol]': lambda: property_container.enthalpy[1],
                                        #     'Enthalpy_W [KJ/kg]': lambda: property_container.enthalpy[0]/18.01528,
                                        #     'Enthalpy_S [KJ/kg]': lambda: property_container.enthalpy[1]/18.01528,
                                        #     'Dens_W': lambda: property_container.dens[0],
                                        #     'Dens_S': lambda: property_container.dens[1]
                                            }

        self.physics.add_property_region(property_container)

    def set_initial_conditions(self):

        # initialization with constant pressure and temperature
        self.input_distribution = {"pressure": 100.,              # [bar]
                              "temperature": 300. + 273.15  # [c]-----[k]
                             }
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                        input_distribution=self.input_distribution)


        state_init = value_vector([self.input_distribution['pressure']])               # Initial presion   (bar)

        self.enth_init = self.physics.property_containers[0].enthalpy_ev['total'].evaluate(state_init,
                                                                                    self.input_distribution['temperature'] )

        estado=np.array([state_init[0],self.enth_init ])
        Sw_init=self.physics.property_containers[0].saturation_ev['water'].evaluate(estado)


        print('\n----------------INITIAL CONDITIONS :')
        print('\n')
        print('Initial pressure (bar)=',self.input_distribution['pressure'])
        print('Initial temperature (C)=',self.input_distribution['temperature']-273.15)
        print('Initial enthalpy ([kJ/kg])=',self.enth_init/18.01528)
        print('Initial enthalpy ([kJ/kmol])=',self.enth_init)
        print('Initial Saturation (water)=',Sw_init)
        print('\n')

    def set_reservoir(self):


        print('\n----------------DEFINE RESERVOIR :')
        print('\n')

        self.Long=1500.   # [m]

        (self.nx, self.ny, self.nz) = (500, 1, 1)
        dx=self.Long/self.nx

        self.nb = self.nx * self.ny * self.nz
        perm = np.ones(self.nb) * 1.
        perm[:self.nx*self.ny] =5.   #  [mD]

        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=dx, dy=100., dz=100.,
                                         permx=perm, permy=perm, permz=perm, poro=.01, depth=1.,
                                         hcap=2470., rcond= 172.8 )
                                        #2650   ,   181.44

        print('Reservoir type= Structured')
        print('Reservoir dimension=',(self.nx, self.ny, self.nz))
        print('Number of blocks=',self.nb)

    def set_wells(self):


        # add well
        self.reservoir.add_well("INJ", wellbore_diameter=0.3048)
        self.reservoir.add_perforation("INJ", cell_index=(1, 1, 1),  skin=0.0,
                                           well_radius=0.1524, multi_segment=False,well_index=100)

        # add well
        self.reservoir.add_well("PRD", wellbore_diameter=0.3048)
        self.reservoir.add_perforation("PRD", cell_index=(self.nb, 1, 1),  skin=0.0,
                                           well_radius=0.1524, multi_segment=False,well_index=100)

                                          # well_indexD - thermal well index (for heat loss through the wellbore)
                                              # if None, use computed value based on cell geometry;
                                              # if 0 - no heat losses

        # ( ?? , reservoir block, well_index, well_indexD)


        print('pozo 1, block:', self.reservoir.wells[0].perforations[0][1])
        print('pozo 1, well_index:', self.reservoir.wells[0].perforations[0][2])
        print('pozo 1, well_indexD:', self.reservoir.wells[0].perforations[0][3])

        print('pozo 2, block:', self.reservoir.wells[1].perforations[0][1])
        print('pozo 2, well_index:', self.reservoir.wells[1].perforations[0][2])
        print('pozo 2, well_indexD:', self.reservoir.wells[1].perforations[0][3])

    def set_well_controls(self):


        #   presion constante
        
        self.P_iny=150                      #     [bar]
        self.inj_temp= 100                    #     [c]
        print('\n---------Injector :')
        print('INJ pressure [bar]=',self.P_iny)
        print('INJ  temp. [c]=',self.inj_temp)
    

        self.P_prod=50                        #     [bar]

        print('\n---------Producer :')
        print('PRD pressure [bar]=',self.P_prod)
        print('\n---------------------------')


        for i, w in enumerate(self.reservoir.wells):
            if 'INJ' in w.name:
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=well_control_iface.BHP,
                                               is_inj=True,
                                               target=self.P_iny,
                                               phase_name='water', 
                                               inj_temp=self.inj_temp + 273.15   #  [c]-----[k] 
                                               )
            else:
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=well_control_iface.BHP,
                                               is_inj=False,
                                               target=self.P_prod
                                               )

    def plot_init_phase_with_p_t(self,P_ini,T_ini,P_iny,T_iny,P_prod,T_prod):

        import numpy as np
        import matplotlib.pyplot as plt
        from CoolProp.CoolProp import PropsSI

        P_min = 1      #  bar
        P_max = 400     #  bar

        P_min = P_min *100000   #  kPa
        P_max = P_max *100000   #  kPa
        P_range = np.logspace(np.log10(P_min), np.log10(P_max), 300)

        # Saturación: líquido y vapor
        h_liq = []
        h_vap = []
        for P in P_range:
            try:
                h_liq.append(PropsSI('H','P',P,'Q',0,'Water'))
                h_vap.append(PropsSI('H','P',P,'Q',1,'Water'))
            except:
                h_liq.append(np.nan)
                h_vap.append(np.nan)

        P_bar = P_range / 1e5
        h_liq = np.array(h_liq)/1000
        h_vap = np.array(h_vap)/1000

        plt.figure(figsize=(8,6))
        plt.semilogy(h_liq, P_bar, 'b-', label='Líquido saturado')
        plt.semilogy(h_vap, P_bar, 'r-', label='Vapor saturado')


        # === Líneas de temperatura constante ===
        T_list = [T_ini, T_iny]  # en K
        colors = ['k', 'b']      # 'k' = negro, 'b' = azul

        for T, color in zip(T_list, colors):
            h_line = []
            P_line = []
            for P in P_range:
                try:
                    h = PropsSI('H', 'P', P, 'T', T, 'Water')
                    h_line.append(h/1000)
                    P_line.append(P/1e5)
                except:
                    h_line.append(np.nan)
                    P_line.append(np.nan)
            plt.semilogy(h_line, P_line, '--', color=color, label=f'T={T-273.15:.0f}°C')

        # === Ubicar un punto dado por presión y entalpia ===

        # p = 5e5       # Pa (5 bar)
        # h = 2100e3    # J/kg (2100 kJ/kg)
        # plt.plot(h/1000, p/1e5, 'ko')
        # plt.text(h/1000, p/1e5*1.1, f"P={p/1e5:.2f} bar\nH={h/1000:.1f} kJ/kg",
        #         ha='center', fontsize=9)


        ##############               Inicial
        #
        # # === Ubicar un punto dado por presión y temperatura ===
        P_punto = P_ini*100000    #  kPa
        T_punto = T_ini           #  K
        # Calcular entalpía en ese punto
        h_punto = PropsSI('H','P',P_punto,'T',T_punto,'Water')  # J/kg
        # Dibujar el punto
        plt.plot(h_punto/1000, P_punto/1e5, 'ko',label='Inicial')
        plt.text(h_punto/1000*1.2, P_punto/1e5,
                f"P={P_punto/1e5:.2f} bar\nT={T_punto-273.15:.1f}°C\nH={h_punto/1000:.1f} kJ/kg",
                ha='center', fontsize=9)

        ##############               Inyeccion
        #
        # # === Ubicar un punto dado por presión y temperatura ===
        P_punto = P_iny*100000    #  kPa
        T_punto = T_iny           #  K
        # Calcular entalpía en ese punto
        h_punto = PropsSI('H','P',P_punto,'T',T_punto,'Water')  # J/kg
        # Dibujar el punto
        plt.plot(h_punto/1000, P_punto/1e5, 'bo',label='Inyeccion', )
        plt.text(h_punto/1000, P_punto/1e5*1.1,
                f"T={T_punto-273.15:.1f}°C",
                ha='center', fontsize=9)


        ##############               Produccion
        #
        # === Ubicar una línea horizontal y la isoterma de producción ===

        P_punto = P_prod * 100000  # Pa

        # Dibujar la línea horizontal punteada de presión de producción
        plt.axhline(y=P_punto/1e5, color='r', linestyle=':', label='Producción')

        # === Calcular entalpías de saturación (líquido y vapor) ===
        try:
            h_liq_sat = PropsSI('H', 'P', P_punto, 'Q', 0, 'Water') / 1000  # kJ/kg
            h_vap_sat = PropsSI('H', 'P', P_punto, 'Q', 1, 'Water') / 1000  # kJ/kg

            # Dibujar puntos en las entalpías de saturación
            plt.plot(h_liq_sat, P_punto/1e5, 'gs', label='Líquido saturado (Prod)')
            plt.plot(h_vap_sat, P_punto/1e5, 'ms', label='Vapor saturado (Prod)')

            # Mostrar las etiquetas
            plt.text(h_liq_sat, P_punto/1e5*1.05,
                    f"h_liq={h_liq_sat:.1f} kJ/kg",
                    ha='center', fontsize=9, color='g')

            plt.text(h_vap_sat, P_punto/1e5*1.05,
                    f"h_vap={h_vap_sat:.1f} kJ/kg",
                    ha='center', fontsize=9, color='m')

        except:
            print("No se pudo calcular la entalpía de saturación para la presión de producción.")

        # Etiqueta opcional para la presión
        plt.text(x=(h_liq_sat + h_vap_sat)/2, y=P_punto/1e5*1.1,
                s=f"P={P_punto/1e5:.2f} bar",
                ha='center', fontsize=9, color='r')

        try:
            # Calcular la temperatura de saturación a la presión de producción
            T_iso_prod = PropsSI('T', 'P', P_punto, 'Q', 0, 'Water')  # K

            # Construir la línea de la isoterma
            h_line = []
            P_line = []
            for P in P_range:
                try:
                    h = PropsSI('H', 'P', P, 'T', T_iso_prod, 'Water')
                    h_line.append(h/1000)  # kJ/kg
                    P_line.append(P/1e5)   # bar
                except:
                    h_line.append(np.nan)
                    P_line.append(np.nan)

            # Dibujar la isoterma en rojo punteado
            plt.semilogy(h_line, P_line, '--', color='r',
                        label=f'Isoterma P_prod (T={T_iso_prod-273.15:.1f}°C)')

        except:
            print("No se pudo calcular la isoterma para P_prod.")

        # Etiquetas y estilo
        plt.xlabel('Entalpía específica [kJ/kg]')
        plt.ylabel('Presión [bar]')
        plt.title('Diagrama Presión - Entalpía del Agua con Isotermas')
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.legend(framealpha=1)
        plt.show()

    def plot_final_phase_with_p_h(self, P_ini, h_ini, P_iny, h_iny, P_prod, h_prod):

        import numpy as np
        import matplotlib.pyplot as plt
        from CoolProp.CoolProp import PropsSI

        P_min = 1      # bar
        P_max = 400    # bar

        P_min = P_min * 100000   # Pa
        P_max = P_max * 100000   # Pa
        P_range = np.logspace(np.log10(P_min), np.log10(P_max), 300)

        # Saturación: líquido y vapor
        h_liq = []
        h_vap = []
        for P in P_range:
            try:
                h_liq.append(PropsSI('H','P',P,'Q',0,'Water'))
                h_vap.append(PropsSI('H','P',P,'Q',1,'Water'))
            except:
                h_liq.append(np.nan)
                h_vap.append(np.nan)

        P_bar = P_range / 1e5
        h_liq = np.array(h_liq)/1000
        h_vap = np.array(h_vap)/1000

        plt.figure(figsize=(8,6))
        plt.semilogy(h_liq, P_bar, 'b-', label='Líquido saturado')
        plt.semilogy(h_vap, P_bar, 'r-', label='Vapor saturado')

        # # === Líneas de temperatura constante ===
        # T_list = [T_iny]  # solo se mantiene inyección como isotermas
        # colors = ['b']

        # for T, color in zip(T_list, colors):
        #     h_line = []
        #     P_line = []
        #     for P in P_range:
        #         try:
        #             h = PropsSI('H', 'P', P, 'T', T, 'Water')
        #             h_line.append(h/1000)
        #             P_line.append(P/1e5)
        #         except:
        #             h_line.append(np.nan)
        #             P_line.append(np.nan)
        #     plt.semilogy(h_line, P_line, '--', color=color, label=f'T={T-273.15:.0f}°C')

        # === Punto Inicial:  usa P_ini y h_ini ===
        P_punto = P_ini * 100000    # Pa
        h_ini=h_ini/18.01528
        h_punto = h_ini * 1000      # convertir kJ/kg → J/kg si lo das en kJ

        # Obtener temperatura solo para mostrar (opcional)
        try:
            T_punto = PropsSI('T','P',P_punto,'H',h_punto,'Water')
            temp_label = f"\nT={T_punto-273.15:.1f}°C"
        except:
            temp_label = ""
        plt.plot(h_ini, P_ini, 'ko', label='Inicial')
        plt.text(h_ini*1.1, P_ini, 
                f"P={P_ini:.2f} bar\nH={h_ini:.1f} kJ/kg" + temp_label,
                ha='center', fontsize=9)

        # === Punto de Inyección ===
        P_punto = P_iny
        h_punto=h_iny/18.01528
        #h_punto = h_iny * 1000      # convertir kJ/kg → J/kg si lo das en kJ
        # h_punto = PropsSI('H','P',P_punto,'T',T_punto,'Water')

        # plt.plot(h_punto/1000, P_punto/1e5, 'bo', label='Inyección')
        # plt.text(h_punto/1000, P_punto/1e5*1.1, 
        #         f"T={T_punto-273.15:.1f}°C", 
        #         ha='center', fontsize=9)
        plt.plot(h_punto, P_punto, 'bo', label='Inyeccion')
        plt.text(h_punto*1.1, P_punto, 
                f"P={P_punto:.2f} bar\nH={h_punto:.1f} kJ/kg" + temp_label,
                ha='center', fontsize=9)

        # === Punto de Producción ===
        P_punto = P_prod*100000    # Pa
        h_prod=h_prod/18.01528
        h_punto = h_prod * 1000      # convertir kJ/kg → J/kg si lo das en kJ
        plt.plot(h_punto/1000, P_punto/1e5, 'ro', label='Producción')
        plt.text(h_punto/1000*.9, P_punto/1e5*1.1, 
                f"P={P_punto/1e5:.2f} bar", 
                ha='center', fontsize=9)

        # === Fracción másica de vapor (x) para producción ===
        
        try:
            h_liq_sat = PropsSI('H', 'P', P_punto, 'Q', 0, 'Water') / 1000
            h_vap_sat = PropsSI('H', 'P', P_punto, 'Q', 1, 'Water') / 1000

            if h_liq_sat <= h_prod <= h_vap_sat:
                x = (h_prod - h_liq_sat) / (h_vap_sat - h_liq_sat)
                x_text = f"x = {x:.3f}"
            elif h_prod < h_liq_sat:
                x = 0.0
                x_text = "Subenfriado (x=0)"
            else:
                x = 1.0
                x_text = "Sobrecalentado (x=1)"

            plt.axhline(y=P_prod, color='r', linestyle=':', label='Presión Producción')
            plt.text(h_prod*.9, P_prod*.7, x_text, ha='center', fontsize=12, color='r')
        except:
            print("No se pudo calcular la fracción másica de vapor para la presión de producción.")
        

        # === Calcular entalpías de saturación (líquido y vapor) ===
        try:
            h_liq_sat = PropsSI('H', 'P', P_punto, 'Q', 0, 'Water') / 1000  # kJ/kg
            h_vap_sat = PropsSI('H', 'P', P_punto, 'Q', 1, 'Water') / 1000  # kJ/kg

            # Dibujar puntos en las entalpías de saturación
            plt.plot(h_liq_sat, P_punto/1e5, 'gs', label='Líquido saturado (Prod)')
            plt.plot(h_vap_sat, P_punto/1e5, 'ms', label='Vapor saturado (Prod)')

            # Mostrar las etiquetas
            plt.text(h_liq_sat, P_punto/1e5*1.05,
                    f"h_liq={h_liq_sat:.1f} kJ/kg",
                    ha='center', fontsize=9, color='g')

            plt.text(h_vap_sat, P_punto/1e5*1.05,
                    f"h_vap={h_vap_sat:.1f} kJ/kg",
                    ha='center', fontsize=9, color='m')

        except:
            print("No se pudo calcular la entalpía de saturación para la presión de producción.")
            
        
        try:
            # Calcular la temperatura de saturación a la presión de producción
            T_iso_prod = PropsSI('T', 'P', P_punto, 'Q', 0, 'Water')  # K

            # Construir la línea de la isoterma
            h_line = []
            P_line = []
            for P in P_range:
                try:
                    h = PropsSI('H', 'P', P, 'T', T_iso_prod, 'Water')
                    h_line.append(h/1000)  # kJ/kg
                    P_line.append(P/1e5)   # bar
                except:
                    h_line.append(np.nan)
                    P_line.append(np.nan)

            # Dibujar la isoterma en rojo punteado
            plt.semilogy(h_line, P_line, '--', color='r',
                        label=f'Isoterma P_prod (T={T_iso_prod-273.15:.1f}°C)')

        except:
            print("No se pudo calcular la isoterma para P_prod.")
            
        
        # Etiquetas y estilo
        plt.xlabel('Entalpía específica [kJ/kg]')
        plt.ylabel('Presión [bar]')
        plt.title('Diagrama Presión - Entalpía del Agua con Isotermas')
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.legend(framealpha=1)
        plt.show()
