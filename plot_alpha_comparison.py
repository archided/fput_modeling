import matplotlib.pyplot as plt
from fput_simulation import simulate_fput

N = 16
dt = 0.01
T_max = 4000
alphas = [0, 0.05, 0.25, 0.5, 1.0]

alpha_list = []
TR_list = []
E1_peak_ratio_list = []

plt.figure(figsize=(10, 6))
for a in alphas:
    times, E_modes, _ = simulate_fput(N, a, dt, T_max)
    plt.plot(times, E_modes[0], label=f'α = {a}')

plt.xlabel('Время')
plt.ylabel('Энергия первой моды')
plt.title('Сравнение возврата при различных α')
plt.legend()
plt.grid(True)
plt.savefig('alpha_comparison.png', dpi=300, bbox_inches='tight')
plt.show()