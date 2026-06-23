import numpy as np
import matplotlib.pyplot as plt
from fput_simulation import simulate_fput

N = 16
alpha = 0.25
dt = 0.01
T_max = 2000

times, E_modes, E_total = simulate_fput(N, alpha, dt, T_max)

# Полная энергия как сумма энергий всех мод
E_sum = np.sum(E_modes, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(times, E_sum, label='Полная энергия 1 (суммирование энергий мод)', color='blue', linewidth=1.5)
plt.plot(times, E_total, label='Полная энергия 2 (через координаты и скорости частиц)', color='red', linewidth=1.5)

plt.xlim(left=0, right=T_max)
plt.ylim(bottom=0)

plt.xlabel('Время')
plt.ylabel('Полная энергия')
plt.title(f'Сохранение энергии (N={N}, α={alpha})')
plt.legend()
plt.grid(True)
plt.savefig('energy_conservation.png', dpi=300, bbox_inches='tight')
plt.show()