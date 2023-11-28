import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from numpy import loadtxt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

sns.set_theme()

actual_date = datetime.now().date()

Train = False

def convert_zero(A, B):
    # Verificar que ambos arrays tengan la misma longitud
    if len(A) != len(B):
        print("Los arrays no tienen la misma longitud")
        return

    # Iterar a través de los elementos de A y B
    for i in range(len(A)):
        if A[i] == 0:
            # Si A es cero, convertir el elemento correspondiente de B a cero
            B[i] = 0

if Train:
    ### COMPARACION DE REWARD

    rew_curves_DDPG = open('Solvers/RL/curves/Rew_DDPG.csv', 'rb')
    rew_curves_PPO = open('Solvers/RL/curves/Rew_PPO.csv', 'rb')
    data_DDPG = gaussian_filter1d(loadtxt(rew_curves_DDPG, delimiter=","), sigma=5)
    data_PPO = gaussian_filter1d(loadtxt(rew_curves_PPO, delimiter=","), sigma=5)

    plt.plot(data_DDPG, label='DDPG', color='tab:red')
    plt.plot(data_PPO, label='PPO', color='tab:blue')

    plt.legend(loc="lower right")
    plt.xlabel("Training Episodes")
    plt.ylabel("Episodic reward")
    plt.savefig(f'Solvers/RL/curves/Reward_comp_{actual_date}.png', dpi=600)
    plt.show()
else:

    ### GRAFICA DE GENERACION

    price_curve = loadtxt(open('Solvers/RL/curves/Precio.csv', 'rb'), delimiter=",")
    E_net_curve = loadtxt(open('Solvers/RL/curves/E_almacenada_red.csv', 'rb'), delimiter=",")
    E_PV_curve = loadtxt(open('Solvers/RL/curves/E_almacenada_PV.csv', 'rb'), delimiter=",")

    E_tot_curve = loadtxt(open('Solvers/RL/curves/E_almacenada_total.csv', 'rb'), delimiter=",")
    E_tot_curve = [0, *E_tot_curve]
    #np.insert(E_tot_curve, 0, 0, 0)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time [h]')
    ax1.set_ylabel('Energy [KWh]')
    ax1.plot(E_net_curve[10:], color='tab:blue')
    ax1.plot(E_PV_curve, color='tab:green')
    ax1.plot(E_tot_curve, color='tab:grey')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cost [$/h]')
    ax2.plot(price_curve, color='tab:red')
    ax2.tick_params(axis='y')

    fig.legend(loc="lower right")
    plt.savefig(f'Solvers/RL/curves/Energy_comp_{actual_date}.png', dpi=600)
    plt.show()


    # GRAFICA DE CARGA

    departure_curve = loadtxt(open('Solvers/RL/curves/Presencia_autos.csv', 'rb'), delimiter=",")
    soc_curve = loadtxt(open('Solvers/RL/curves/SOC.csv', 'rb'), delimiter=",")

    departure_curve = np.hstack((departure_curve[:, -1].reshape(-1, 1), departure_curve[:, :-1]))

    # Crear el subplot de 2 filas y 5 columnas
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    k = 0

    # Rellenar cada subgráfico con los datos
    for i in range(2):
        for j in range(5):
            k += 1

            convert_zero(departure_curve[k - 1, :], soc_curve[k-1, :])
            axs[i, j].fill_between(range(24), price_curve, step="pre", alpha=0.4)
            #axs[i, j].plot(departure_curve[k-1, :], label='departure', color='red')
            axs[i, j].plot(price_curve, color='tab:red')
            axs[i, j].step(range(25), departure_curve[k - 1, :], label='departure', color='green')
            axs[i, j].plot(soc_curve[k-1, :], label='soc', color='blue')
            #axs[i, j].step(timestep, soc_adjusted, label='soc', color='blue')
            axs[i, j].set_title(f'Vehiculo {k}')

    # Ajustar el diseño para evitar superposiciones
    plt.tight_layout()

    # Mostrar el gráfico
    plt.savefig(f'Solvers/RL/curves/Charging_{actual_date}.png', dpi=600)
    plt.show()


