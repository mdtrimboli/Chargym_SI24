import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from numpy import loadtxt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

sns.set_theme()

actual_date = datetime.now().date()

Train = False
algoritmo = 'ppo'
fecha_ddpg = '2024-01-31'
fecha_ppo = '2024-02-15'

if Train:
    ### COMPARACION DE REWARD

    rew_curves_DDPG = open(f'Solvers/RL/curves/Rew_DDPG_{fecha_ddpg}.csv', 'rb')
    rew_curves_PPO = open(f'Solvers/RL/curves/Rew_PPO_{fecha_ppo}.csv', 'rb')
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

    ### GRAFICA DE GENERACION RBC
    if algoritmo == 'rbc':

        price_curve = loadtxt(open('Solvers/RL/curves/Precio.csv', 'rb'), delimiter=",")

        E_net_curve = loadtxt(open(f'algos/RBC/curves/E_almacenada_red_{algoritmo}.csv', 'rb'), delimiter=",")
        E_PV_curve = loadtxt(open(f'algos/RBC/curves/E_almacenada_PV_{algoritmo}.csv', 'rb'), delimiter=",")
        E_tot_curve = loadtxt(open(f'algos/RBC/curves/E_almacenada_total_{algoritmo}.csv', 'rb'), delimiter=",")

    else:
        ### GRAFICA DE GENERACION DDPG Y PPO
        price_curve = loadtxt(open('curves/Precio.csv', 'rb'), delimiter=",")
        sb_consume_curve = loadtxt(open('curves/sb_energy.csv', 'rb'), delimiter=",")
        ev_consume_curve = loadtxt(open('curves/EV_consume.csv', 'rb'), delimiter=",")
        E_net_curve = loadtxt(open(f'curves/E_almacenada_red_{algoritmo}.csv', 'rb'), delimiter=",")
        E_PV_curve = loadtxt(open(f'curves/E_almacenada_PV_{algoritmo}.csv', 'rb'), delimiter=",")
        E_tot_curve = loadtxt(open(f'curves/E_almacenada_total_{algoritmo}.csv', 'rb'), delimiter=",")

        if algoritmo == 'ddpg':
            E_tot_curve = [0, *E_tot_curve]
            E_PV_curve = [0, *E_PV_curve]
            E_net_curve = E_net_curve[10:]

    # CÁLCULO DE ENERGÍA COMPRADA Y SU COSTO
    if algoritmo == "ddpg":

        En_total = np.sum(E_net_curve[:-1])
        print(f"Energía total de {algoritmo}: {En_total}")
        Costo_total = np.sum(price_curve*E_net_curve[:-1])
        print(f"Costo total de {algoritmo}: {Costo_total}")
    else:
        En_total = np.sum(E_net_curve)
        print(f"Energía total de {algoritmo}: {En_total}")
        Costo_total = np.sum(price_curve * E_net_curve)
        print(f"Costo total de {algoritmo}: {Costo_total}")

    Costo_por_hora = []
    for a in range(len(E_tot_curve)):
        Costo_por_hora.append(E_net_curve[a] * price_curve[a])

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time [h]')
    ax1.set_ylabel('Energy cost[$]')
    #ax1.plot(sb_consume_curve, color='tab:orange', label='SB Demand')
    #ax1.plot(E_tot_curve, color='tab:grey', label='Total Consume')
    #ax1.plot(ev_consume_curve, color='tab:cyan', label='EV Consume')
    #ax1.plot(E_net_curve, color='tab:blue', label='Power grid energy')
    #ax1.plot(E_net_curve, color='tab:blue', label='Power grid energy')
    ax1.plot(Costo_por_hora, color='black', label='Energy cost')
    #ax1.plot(E_PV_curve, color='tab:green', label='PV energy')
    ax1.tick_params(axis='y')
    ax1.legend(loc="upper left", framealpha=0.7, facecolor='white')
    #ax1.set_ylim(top=120)
    ax1.set_xlim([0,23])
    ax1.set_xticks(np.arange(0, 23, step=4))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Price [$/kWh]')
    ax2.plot(price_curve, color='tab:red', label='Price')

    ax2.tick_params(axis='y')
    ax2.legend(loc="upper right", framealpha=0.7, facecolor='white')
    #ax2.set_ylim(top=0.12)
    ax2.grid(axis='y', visible=False)


    #ax3 = ax1.twinx()
    #ax3.spines.right.set_position(("axes", 1.08))
    #ax3.plot(Costo_por_hora, color='black', label='Energy cost')
    #ax3.legend(loc="lower left", framealpha=0.7, facecolor='white')


    plt.savefig(f'curves/Energy_comp_{actual_date}_{algoritmo}.png', dpi=600)
    plt.show()


    # GRAFICA DE CARGA
    if algoritmo == 'rbc':
        departure_curve = loadtxt(open(f'algos/RBC/curves/Presencia_autos_{algoritmo}.csv', 'rb'), delimiter=",")
        soc_curve = loadtxt(open(f'algos/RBC/curves/SOC_{algoritmo}.csv', 'rb'), delimiter=",")
    else:
        departure_curve = loadtxt(open(f'curves/Presencia_autos_{algoritmo}.csv', 'rb'), delimiter=",")
        soc_curve = loadtxt(open(f'curves/SOC_{algoritmo}.csv', 'rb'), delimiter=",")

    departure_curve = np.hstack((departure_curve[:, -1].reshape(-1, 1), departure_curve[:, :-1]))

    # Crear el subplot de 2 filas y 5 columnas
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    k = 0
    # Rellenar cada subgráfico con los datos
    for i in range(2):
        for j in range(5):
            k += 1
            #axs[i, j].plot(departure_curve[k-1, :], label='departure', color='red')
            axs[i, j].step(range(25), departure_curve[k - 1, :], label='Presence', color='green')
            axs[i, j].plot(soc_curve[k-1, :], label='SoC', color='blue')
            axs[i, j].set_title(f'EV Spot {k}')

    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    axs[0, 0].set_ylabel('SOC')
    axs[1, 0].set_ylabel('SOC')
    axs[1, 0].set_xlabel('Time [hour]')
    axs[1, 1].set_xlabel('Time [hour]')
    axs[1, 2].set_xlabel('Time [hour]')
    axs[1, 3].set_xlabel('Time [hour]')
    axs[1, 4].set_xlabel('Time [hour]')

    # Ajustar el diseño para evitar superposiciones

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
    lines, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2)

    # Mostrar el gráfico
    plt.savefig(f'curves/Charging_{actual_date}_{algoritmo}.png', dpi=600)
    plt.show()

