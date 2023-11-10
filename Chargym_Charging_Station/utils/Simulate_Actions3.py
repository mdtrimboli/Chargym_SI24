import numpy as np
import matplotlib.pyplot as plt
import time


def simulate_clever_control(self, actions):
    hour = self.timestep
    Consumed = self.Energy['Consumed']
    Renewable = self.Energy['Renewable']
    present_cars = self.Invalues['present_cars']
    leave = self.leave
    BOC = self.BOC      # SOC


    P_charging = np.zeros(self.number_of_cars)
    # Calculation of demand based on actions
    # Calculation of actions for cars
    # ----------------------------------------------------------------------------
    for car in range(self.number_of_cars):
        if actions[car] >= 0:      # Si hay que cargar el auto --> Ec (2) del paper (MaxEnergy)
            max_charging_energy = min([10, (1-BOC[car, hour])*self.EV_Param['EV_capacity']])    # Con el 'min' chequea la Ec (3) del paper
        else:
            max_charging_energy = min([10, BOC[car, hour] * self.EV_Param['EV_capacity']])
        # in case action=[-100,100] P_charging[car] = actions[car]/100*max_charging_energy
        # otherwise if action=[-1,1] P_charging[car] = 100*actions[car]/100*max_charging_energy

        # P_charging[car] = actions[car]/100*max_charging_energy
        # P_charging[car] = 100 * actions[car] / 100 * max_charging_energy
        if present_cars[car, hour] == 1:      # Si el auto está --> Ec (4) del paper (Pdem)
            # Divide en vez de multiplicar como en el paper
            # P_charging[car] = 100*actions[car]/100*max_charging_energy
            P_charging[car] = actions[car] * max_charging_energy
            # P_charging[car] > 0 ----> Charging
            # P_charging[car] < 0 ----> Discharging

        else:
            P_charging[car] = 0

    # Calculation of next state of Battery based on actions
    # ----------------------------------------------------------------------------
    for car in range(self.number_of_cars):
        if present_cars[car, hour] == 1:      # Si el auto está --> SoC próximo = SoC actual + Pdem/capacidad
            BOC[car, hour+1] = BOC[car, hour] + P_charging[car]/self.EV_Param['EV_capacity']
            # Pdem/capacidad es lo que se va a cargar la batería en la próxima hora

    # Calculation of energy utilization from the PV
    # Calculation of energy coming from Grid
    # ----------------------------------------------------------------------------
    #RES_avail = max([0, Renewable[0, hour] - Consumed[0, hour]])      # Energía renovable disponible ---> Siempre usa el día 0!!!

    Total_charging = sum(P_charging)      # Potencia demandada y consumida por todos los autos

    # First Cost index
    # ----------------------------------------------------------------------------
    # Grid_final = max([Total_charging - RES_avail, 0])      # Lo que se consume de la red
    RES_Gen = max([0, Renewable[0, hour]])

    if Total_charging >= 0:      # Solo se calcula el sobrante de energía PV y el consumo de la red si los autos consumieron erengía
        RES_avail = max([RES_Gen - Total_charging, 0])      # Solo hay energía sobrante si generó más
    else:
        RES_avail = Renewable[0, hour]

    Grid_final = max([Total_charging - RES_Gen, 0])
    Cost_1 = Grid_final*self.Energy["Price"][0, hour]       # Lo que cuesta consumir de la red (positivo)


    # Second Cost index
    # Penalty of wasted RES energy
    # This is not used in this environment version
    # ----------------------------------------------------------------------------
    # RES_avail = max([RES_avail-Total_charging, 0])
    # Cost_2 = -RES_avail * (self.Energy["Price"][0, hour]/2)

    # Third Cost index
    # Penalty of not fully charging the cars that leave
    # ----------------------------------------------------------------------------
    Cost_EV = []
    for ii in range(len(leave)):
        Cost_EV.append(((1-BOC[leave[ii], hour+1])*2)**2)       # Simulate_Station crea leave
        # BOC[leave[ii], hour+1] solo tiene en cuenta el SoC de los autos que se van a ir en la próxima hora
    Cost_3 = sum(Cost_EV)
    # TODO: Existe un costo 4 que es la energía que sobra de los EV (Total_charging < 0)
    Cost = Cost_1 + Cost_3

    return Cost, Grid_final, RES_avail, Cost_3, BOC
    # Cost: Costo total
    # Grid_final: Lo que se consume de la red
    # RES_avail: Energía renovable disponible,
    # Cost_3: Costo por no cargar 100% un auto,
    # Soc