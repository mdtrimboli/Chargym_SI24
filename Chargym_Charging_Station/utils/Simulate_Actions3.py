import numpy as np
import time


def simulate_clever_control(self, actions):
    hour = self.timestep
    Consumed = self.Energy['Consumed']                  # TODO: Chequear el self dado que no es una clase
    Renewable = self.Energy['Renewable']
    present_cars = self.Invalues['present_cars']
    #print(present_cars[:,hour])

    leave = self.leave
    BOC = self.BOC      # TODO: BOC = Battery State of Charge = SoC

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
            P_charging[car] = 100*actions[car]/100*max_charging_energy
        else:
            P_charging[car] = 0

    # Calculation of next state of Battery based on actions
    # ----------------------------------------------------------------------------
    for car in range(self.number_of_cars):
        if present_cars[car, hour] == 1:      # Si el auto está --> SoC próximo = SoC actual + Pdem/capacidad
            BOC[car, hour+1] = BOC[car, hour] + P_charging[car]/self.EV_Param['EV_capacity']
            """ Puede llegar a ser mayor que 1???"""
            # Pdem/capacidad es lo que se va a cargar la batería en la próxima hora

    # Calculation of energy utilization from the PV
    # Calculation of energy coming from Grid
    # ----------------------------------------------------------------------------
    RES_avail = max([0,Renewable[0, hour] - Consumed[0, hour]])      # Energía renovable disponible
    Total_charging = sum(P_charging)      # Energía demandada por EVs - energía entregada por EVs

    # First Cost index
    # ----------------------------------------------------------------------------
    Grid_final = max([Total_charging - RES_avail, 0])      # Lo que se consume de la red
    Cost_1 = Grid_final*self.Energy["Price"][0, hour]      # Lo que cuesta consumir de la red (positivo)

    # Second Cost index
    # Penalty of wasted RES energy
    # This is not used in this environment version
    # ----------------------------------------------------------------------------
    # RES_avail = max([RES_avail-Total_charging, 0])
    # Cost_2 = -RES_avail * (self.Energy["Price"][0, hour]/2)

    #Third Cost index
    #Penalty of not fully charging the cars that leave
    # ----------------------------------------------------------------------------
    Cost_EV = []
    for ii in range(len(leave)):
        Cost_EV.append(((1-BOC[leave[ii], hour+1])*2)**2)       # Simulate_Station crea leave
        # BOC[leave[ii], hour+1] solo tiene en cuenta el SoC de los autos que se van a ir en la próxima hora
    Cost_3 = sum(Cost_EV)

    Cost = Cost_1 + Cost_3

    return Cost, Grid_final, RES_avail, Cost_3, BOC     #Costo, Lo que consume de la red, Energía renovable disponible,
                                                        # Costo por no cargar 100% un auto, Soc
