import numpy as np

def simulate_fput(N, alpha, dt, T_max, save_every=10):
    """
    Моделирование alpha-цепочки FPUT.
    
    Параметры:
        N (int): число частиц
        alpha (float): коэффициент нелинейности
        dt (float): шаг интегрирования
        T_max (float): общее время моделирования
        save_every (int): шаг сохранения данных (каждые save_every шагов)
    
    Возвращает:
        times (np.ndarray): массив времён
        E_modes (np.ndarray): энергии мод (размер N x n_saves)
        E_total (np.ndarray): полная энергия в каждый момент сохранения
    """
    # Частоты нормальных мод
    omega = np.array([2 * np.sin(np.pi * k / (2 * (N + 1))) for k in range(1, N+1)])

    # Начальные условия (энергия в первой моде, нулевые скорости)
    x0 = np.sin(np.pi * np.arange(1, N+1) / (N + 1))
    v0 = np.zeros(N)
    state = np.concatenate([x0, v0])

    def f(state, alpha):
        N = len(state) // 2
        x = state[:N]
        v = state[N:]
        a = np.zeros(N)
        for i in range(N):
            left = x[i-1] if i > 0 else 0.0
            right = x[i+1] if i < N-1 else 0.0
            a[i] = (right - 2*x[i] + left) + alpha * ((right - x[i])**2) - alpha * ((x[i] - left)**2)
        return np.concatenate([v, a])

    def rk4_step(state, dt, alpha):
        k1 = f(state, alpha)
        k2 = f(state + 0.5*dt*k1, alpha)
        k3 = f(state + 0.5*dt*k2, alpha)
        k4 = f(state + dt*k3, alpha)
        return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    steps = int(T_max / dt)
    n_saves = steps // save_every + 1

    E_modes = np.zeros((N, n_saves))
    E_total = np.zeros(n_saves)
    times = np.zeros(n_saves)
    save_count = 0

    state_current = state.copy()
    for step in range(steps):
        if step % save_every == 0:
            x = state_current[:N]
            v = state_current[N:]

            # Преобразование Фурье
            Q = np.zeros(N)
            P = np.zeros(N)
            for k in range(1, N+1):
                sin_vals = np.sin(np.pi * np.arange(1, N+1) * k / (N + 1))
                Q[k-1] = np.sqrt(2.0/(N+1)) * np.sum(x * sin_vals)
                P[k-1] = np.sqrt(2.0/(N+1)) * np.sum(v * sin_vals)

            # Энергия каждой моды
            for k in range(N):
                E_modes[k, save_count] = 0.5 * (P[k]**2 + omega[k]**2 * Q[k]**2)

            # Полная энергия
            E_val = 0.0
            # Кинетическая энергия
            for i in range(N):
                E_val += 0.5 * v[i]**2
            # Потенциальная энергия пружин (с учётом закреплённых концов)
            for i in range(N+1):
                left = x[i-1] if i > 0 else 0.0
                right = x[i] if i < N else 0.0
                r = right - left
                E_val += 0.5 * r**2 + (alpha/3) * r**3
            E_total[save_count] = E_val

            times[save_count] = step * dt
            save_count += 1

        state_current = rk4_step(state_current, dt, alpha)

   
    times = times[:save_count]
    E_modes = E_modes[:, :save_count]
    E_total = E_total[:save_count]

    return times, E_modes, E_total