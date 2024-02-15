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
fecha_ppo = '2024-02-14'

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

    sb_consume_curve = sb_consume_curve[:-4]
    sb_porc = []
    e_net_curve_SB = []
    e_PV_curve_SB = []
    e_EV_curve_SB = []
    porcentaje_e_net_curve_SB = []
    porcentaje_e_PV_curve_SB = []
    porcentaje_e_EV_curve_SB = []
    Total = []
    Total_sin_EV = []
    for a in range(len(sb_consume_curve)):

        # Cálculo de composición porcentual de consumo SB
        #porcentaje de SB respecto total
        sb_porc.append(min([sb_consume_curve[a] / E_tot_curve[a], 1])) # cuando E_tot_curve < sb_consume_curve es porque los vehículos están entregando energía
        # cantidad de energía red va al SB respecto total
        e_net_curve_SB.append(min([E_net_curve[a] * sb_porc[a], sb_consume_curve[a]])) # no puede superar el sb_consume_curve
        # cantidad de energía PV va al SB respecto total
        e_PV_curve_SB.append(min([E_PV_curve[a] * sb_porc[a], sb_consume_curve[a]])) # no puede superar el sb_consume_curve
        # cantidad de energía EV va al SB respecto total

        np.array(e_EV_curve_SB.append(max([0, sb_consume_curve[a]-e_net_curve_SB[a]-e_PV_curve_SB[a]])))
        # porcentajes de energía de SB
        porcentaje_e_net_curve_SB.append(e_net_curve_SB[a] / sb_consume_curve[a])
        porcentaje_e_PV_curve_SB.append(e_PV_curve_SB[a] / sb_consume_curve[a])
        porcentaje_e_EV_curve_SB.append(e_EV_curve_SB[a] / sb_consume_curve[a])

        Total.append(porcentaje_e_net_curve_SB[a] + porcentaje_e_PV_curve_SB[a] + porcentaje_e_EV_curve_SB[a])
        Total_sin_EV.append(porcentaje_e_net_curve_SB[a] + porcentaje_e_PV_curve_SB[a])

    #print(f"sb_consume_curve: {sb_consume_curve}")
    #print(f"e_net_curve_SB: {porcentaje_e_net_curve_SB}")
    #print(f"e_PV_curve_SB: {porcentaje_e_PV_curve_SB}")
    #print(f"e_EV_curve_SB: {porcentaje_e_EV_curve_SB}")

    index = np.arange(len(sb_consume_curve))
    fig, ax = plt.subplots()
    ax.bar(index, Total, label = 'Porcentaje EV')
    ax.bar(index, Total_sin_EV, label = 'Porcentaje red')
    ax.bar(index, porcentaje_e_PV_curve_SB, label = 'Porcentaje PV')
    ax.plot(E_PV_curve/55.5, color='tab:green', label='PV energy')
    ax.plot(price_curve/0.1, color='tab:red', label='Price')
    ax.legend(loc="upper left", framealpha=0.7, facecolor='white')

    plt.show



    #ax1.plot(sb_consume_curve, color='tab:orange', label='SB Demand')
    #ax1.plot(E_tot_curve, color='tab:grey', label='Total Consume')
    #ax1.plot(ev_consume_curve, color='tab:cyan', label='EV Consume')
    #ax1.plot(E_net_curve, color='tab:blue', label='Power grid energy')
    #ax1.plot(E_PV_curve, color='tab:green', label='PV energy')




    plt.savefig(f'curves/Energy_comp_{actual_date}_{algoritmo}.png', dpi=600)
    plt.show()


    # Ajustar el diseño para evitar superposiciones

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
    lines, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2)

    # Mostrar el gráfico
    plt.savefig(f'curves/Charging_{actual_date}_{algoritmo}.png', dpi=600)
    plt.show()

