import matplotlib.pyplot as plt
from fput_simulation import simulate_fput

N = 16
alpha = 0.25
dt = 0.01
T_max = 2500

times, E_modes, _ = simulate_fput(N, alpha, dt, T_max)

plt.figure(figsize=(10, 6))
for k in range(min(5, N)):
    plt.plot(times, E_modes[k], label=f'{k+1} мода')
plt.xlabel('Время')
plt.ylabel('Энергия')
plt.title(f'Возврат энергии в цепочке FPUT (N={N}, α={alpha})')
plt.legend()
plt.grid(True)
plt.savefig('alpha025.png', dpi=300, bbox_inches='tight')
plt.show()