import numpy as np

def Simulate_Station(self):

    BOC = self.BOC
    Arrival = self.Invalues['ArrivalT']
    Departure = self.Invalues['DepartureT']
    present_cars = self.Invalues['present_cars']
    number_of_cars = self.number_of_cars
    day = self.day
    hour = self.timestep


    # cálculo de qué coches salen ahora
    leave = []
    if hour < 24:
        for car in range(number_of_cars):
            Departure_car = Departure[car]
            if present_cars[car, hour] == 1 and (hour+1 in Departure_car):
                leave.append(car)       # Si el auto está y se tiene que ir en la hora siguiente ---> Se agrega a leave

    # calculation of the hour each car is leaving
    Departure_hour = []
    for car in range(number_of_cars):
        if present_cars[car,hour] == 0:
            Departure_hour.append(0)        # Si el auto no está --> no tiene hora de salida
        else:
            for ii in range(len(Departure[car])):
                if hour < Departure[car][ii]:     # Si la hora que de salida > a la actual
                    Departure_hour.append(Departure[car][ii]-hour)       # --> Se guarda la cantidad de horas que falta para que salga de cada auto
                    break
                    # TODO: Departure[car] tiene varias horas de salida para cada auto para un día?

    # calculation of the BOC of each car
    Battery = []
    for car in range(number_of_cars):
        Battery.append(self.BOC[car,hour])      # Guarda en Battery[] los SoC de cada auto


    return leave,Departure_hour,Battery     # Autos que se van, Hora que falta para salir, Soc de cada auto.
