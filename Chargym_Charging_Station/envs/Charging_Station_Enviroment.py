import numpy as np
import pandas as pd
import os
import sys
import gym
import pathlib
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding
from scipy.io import loadmat, savemat
from Chargym_Charging_Station.utils import Energy_Calculations
from Chargym_Charging_Station.utils import Simulate_Station3
from Chargym_Charging_Station.utils import Init_Values
from Chargym_Charging_Station.utils import Simulate_Actions3
import time


class ChargingEnv(gym.Env):
    def __init__(self, price=4, solar=1):
        # basic_model_parameters
        self.number_of_cars = 10            # Charging spots
        self.number_of_days = 1
        self.price_flag = price             # Curva de precio elegida
        self.solar_flag = solar             # Habilitacion del Panel PV
        self.done = False
        self.Grid_Evol_mem = []
        self.SOC = []
        self.E_almacenada_total = 0
        self.Lista_E_Almac_Total = []

        # EV_parameters
        EV_capacity = 30
        charging_effic = 0.91
        discharging_effic = 0.91
        charging_rate = 11
        discharging_rate = 11

        self.EV_Param = {'charging_effic': charging_effic, 'EV_capacity': EV_capacity,
                         'discharging_effic': discharging_effic, 'charging_rate': charging_rate,
                         'discharging_rate': discharging_rate}

        # Renewable_Energy
        PV_Surface = 2.279 * 1.134 * 20     # = 51,68772 [m2]
        PV_effic = 0.21

        self.PV_Param = {'PV_Surface': PV_Surface, 'PV_effic': PV_effic}

        # self.current_folder = os.getcwd() + '\\utils\\Files\\'
        self.current_folder = os.path.realpath(os.path.join(os.path.dirname(__file__), '..')) + '\\Files\\'

        low = np.array(np.zeros(8+2*self.number_of_cars), dtype=np.float32)     # Lower threshold of state space
        high = np.array(np.ones(8+2*self.number_of_cars), dtype=np.float32)     # Upper threshold of state space
        # Definicion de espacio de accion
        self.action_space = spaces.Box(     # Setea entre {-1 y 1}
            low=-1,
            high=1, shape=(self.number_of_cars,),     # Con el tamaño de la cantidad de autos
            dtype=np.float32
        )
        # Definicion de espacio de estados
        self.observation_space = spaces.Box(     # Setea entre {0 y 1}
            low=low,
            high=high,
            dtype=np.float32
        )

        self.seed
        Cont_n = 0


    def step(self, actions):

        # reward: Costo total
        # Grid: Lo que se consume de la red
        # Res_wasted: Energía renovable disponible,
        # Cost_EV: Costo por no cargar 100% un auto,
        # BOC: SOC

        [reward, Grid, Res_wasted, Cost_EV, self.BOC] = Simulate_Actions3.simulate_clever_control(self, actions)

        self.SOC = self.BOC

        # Almacenar datos en variables historicas

        self.Grid_Evol_mem.append(Grid)
        self.Grid_Evol.append(Grid)
        self.Res_wasted_evol.append(Res_wasted)
        self.Penalty_Evol.append(Cost_EV)
        self.Cost_History.append(reward)

        # ------------------------------------------------------------------------------------------------------
        # Guarda el Total_Charging en un vector
        # Si recién comienza el día reiniciar el vector
        if self.timestep ==0:
            self.Lista_E_Almac_Total.clear()
        self.Lista_E_Almac_Total.append(self.E_almacenada_total)
        # ------------------------------------------------------------------------------------------------------

        # Actualizar "t" y obtener observaciones
        self.timestep = self.timestep + 1
        conditions = self.get_obs()
        # Si se completa el dia, finalizar y almacenar resultados
        if self.timestep == 24:
            self.done = True
            self.timestep = 0
            Results = {'BOC': self.BOC, 'Grid_Final': self.Grid_Evol, 'RES_wasted': self.Res_wasted_evol,
                       'Penalty_Evol':self.Penalty_Evol,
                       'Renewable': self.Energy['Renewable'],'Cost_History': self.Cost_History}
            #------------------------------------------------------------------------------------------------------------


            Generacion_pv = Results['Renewable'][0][:24]
            self.E_almac_pv = Generacion_pv - Results['RES_wasted']

            En_almacenada_total = Results['Grid_Final'] + self.E_almac_pv # En_consumida_total = E. Consumida de la red + Energía consumida del panel
            #Porcentaje_Red = np.zeros(len(En_almacenada_total))
            #Porcentaje_PV = np.zeros(len(En_almacenada_total))
            Energía_desp_EV = np.zeros(len(En_almacenada_total))
            #for ii in range(len(En_almacenada_total)):
            #    if En_almacenada_total[ii] >= 0:
            #        Porcentaje_Red[ii] = Results['Grid_Final'][ii] / En_almacenada_total[ii]
            #        Porcentaje_PV[ii] = Generacion_pv[ii] / En_almacenada_total[ii]
            #    else:
            #        Porcentaje_Red[ii] = 0
            #        Porcentaje_PV[ii] = 0

            horas = np.linspace(0,23,24)
            #print("Consumo de la Red: ", Results['Grid_Final'])     # Lo que se consume de la red
            #print("Generación de PV: ", Generacion_pv)      # Energía generada por PV (cada hora)

            #print("Consumo de energía del PV: ", Consumo_pv[0])        # Energía consumida del PV (cada media hora)
            #print("Irradiación              : ", self.Energy['Radiation'])
            #print("Generac de orig E. del PV: ", Results['Renewable'])
            #print("Generac de energía del PV: ", Generacion_pv)
            #print("Sobrante de Energía de PV: ", Results['RES_wasted'])

            #print("SoC: ", Results['BOC'])      # Estado de carga de los autos las 24 hs
            #print("Energía consumida pv: ", self.Consumo_pv)
            #print("Energía generada pv: ", Results['RES_wasted'])
            #print('Total_charging', self.Lista_E_Almac_Total)

            #plt.plot(Generacion_pv, label='Gen PV', color='green')
            #plt.plot(Results['RES_wasted'], label='Wasted', color='orange')
            #plt.plot(Consumo_pv, label='Consumed', color='red')
            #plt.show()

            """
            #plt.plot(horas, En_consumida_total,'b', label = 'Consumo EVs')
            #plt.plot(horas, Consumo_pv, 'g', label = 'Consumo PV')

            #plt.plot(horas, En_consumida_total, 'b', label='Energía consumida Total')
            #plt.plot(horas, Results['Grid_Final'], 'r', label='Consumo RED')
            #plt.plot(horas, Consumo_pv, 'g', label='Energía consumida PV')
            #plt.legend(loc = 'upper left')
            #plt.show()
            """
            # print("Energía consumida pv: ", Consumo_pv)

            # ------------------------------------------------------------------------------------------------------------
            savemat(self.current_folder + '\Results.mat', {'Results': Results})

        self.info = {'SOC':self.SOC, 'Presence': self.Invalues['present_cars'], 'Cost_3': Cost_EV}
        return conditions, -reward, self.done, self.info        # Devuelve la observación, - (el costo), y si terminó los 24 steps

    def reset(self, reset_flag=0):
        self.timestep = 0
        self.day = 1
        self.done = False

        # Consumed: Array vacío de igual tamaño de Renewable para guardar energía renovable consumida
        # Renewable: Array de [dias, horas] con energía que genera el panel fotovoltáico
        # Price: Array de [dias, horas] con los precios en cada día del experimento
        # Radiation: Array de [dias, horas] con la radiación solar disponible

        Consumed, Renewable, Price, Radiation = Energy_Calculations.Energy_Calculation(self)
        self.Energy = {'Consumed': Consumed, 'Renewable': Renewable,
                       'Price': Price, 'Radiation': Radiation}      # Saca valores de 'Energy_Calculations' y los guarda en Energy
        if reset_flag == 0:     # Si se reseteó, saca valores de Init_Values y los carga en Invalues

            # BOC: SoC
            # ArrivalT: Hora de llegada de cada auto
            # DepartureT: Hora de salida de cada auto
            # evolution_of_cars: Cantidad de autos por hora
            # present_cars: Mapa de qué auto está presente a cada hora

            [BOC, ArrivalT, DepartureT, evolution_of_cars, present_cars] = Init_Values.InitialValues_per_day(self)
            self.Invalues = {'BOC': BOC, 'ArrivalT': ArrivalT, 'evolution_of_cars': evolution_of_cars,
                             'DepartureT': DepartureT, 'present_cars': present_cars}
            savemat(self.current_folder + '\Initial_Values.mat', self.Invalues)     # Guarda Invalues en carpeta
        else:       # Si no se reseteó, saca valores de Invalues de la carpeta y los carga en Invalues
            contents = loadmat(self.current_folder + '\Initial_Values.mat')
            self.Invalues = {'BOC': contents['BOC'], 'Arrival': contents['ArrivalT'][0],
                             'evolution_of_cars': contents['evolution_of_cars'], 'Departure': contents['DepartureT'][0],
                             'present_cars': contents['present_cars'], 'ArrivalT': [], 'DepartureT': []}
            for ii in range(self.number_of_cars):       # Guarda todos los valores de Arrival y Departure en ArrivalT y DepartureT
                self.Invalues['ArrivalT'].append(self.Invalues['Arrival'][ii][0].tolist())
                self.Invalues['DepartureT'].append(self.Invalues['Departure'][ii][0].tolist())

        return self.get_obs()       # Devuelve Observación



    def get_obs(self):
        if self.timestep == 0:
            self.Cost_History = []
            self.Grid_Evol = []
            self.Res_wasted_evol = []
            self.Penalty_Evol =[]
            self.BOC = self.Invalues["BOC"]

        # leave: Autos que se van
        # Departure_hour: Hora que falta para salir
        # Battery: Soc de cada auto
        [self.leave, Departure_hour, Battery] = Simulate_Station3.Simulate_Station(self)

        disturbances = np.array([self.Energy["Radiation"][0, self.timestep] / 1000, self.Energy["Price"][0, self.timestep] / 0.1])
        # disturbances = [Radiación Actual / 1000, Precio Actual / 0.1] --> para mapear a [0,1]
        predictions = np.concatenate((np.array([self.Energy["Radiation"][0, self.timestep + 1:self.timestep + 4] / 1000]), np.array([self.Energy["Price"][0,self.timestep + 1:self.timestep + 4] / 0.1])), axis=None),
        # predictions = [Radiación+1; Radiación+4, Precio+1; Precio+4]
        states = np.concatenate((np.array(Battery), np.array(Departure_hour)/24),axis=None)
        # states = [SoC1; SoC10, Horas para salir1; Horas para salir10]
        observations = np.concatenate((disturbances,predictions,states),axis=None)
        # observations = [disturbances, predictions, states]
        return observations

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return 0
