import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# ============================================================
# 1. Целевая функция и декодирование хромосомы
# ============================================================
BOUND = 5.0          # x, y ∈ [-5, 5]
N_BITS_PER_VAR = 12  # 12 бит на переменную -> точность 10/4095 ≈ 0.0024
N_BITS = 2 * N_BITS_PER_VAR

def decode(chromosome):
    """Бинарная хромосома длины 24 -> (x, y) в [-5, 5]."""
    x_bits = chromosome[:N_BITS_PER_VAR]
    y_bits = chromosome[N_BITS_PER_VAR:]
    x_int = int(''.join(map(str, x_bits)), 2)
    y_int = int(''.join(map(str, y_bits)), 2)
    x = -BOUND + x_int * (2 * BOUND) / (2**N_BITS_PER_VAR - 1)
    y = -BOUND + y_int * (2 * BOUND) / (2**N_BITS_PER_VAR - 1)
    return x, y

def fitness(chromosome):
    """f(x,y) = 1/(1+x^2+y^2). Значения в (0, 1]."""
    x, y = decode(chromosome)
    return 1.0 / (1.0 + x*x + y*y)

# ============================================================
# 2. Генетические операторы
# ============================================================
def init_population(N, rng):
    return rng.integers(0, 2, size=(N, N_BITS))

def selection_roulette(pop, fit, N, rng):
    """Колесо рулетки: вероятность ∝ приспособленности."""
    # Сдвиг, чтобы все фитнесы были > 0 (у нас и так > 0, но на всякий случай)
    f = fit - fit.min() + 1e-9
    probs = f / f.sum()
    idx = rng.choice(len(pop), size=N, p=probs)
    return pop[idx].copy()

def selection_elite(pop, fit, N, rng, elite_frac=0.1):
    """
    Элитный отбор: лучшие elite_frac остаются без изменений,
    остальные — турнирный отбор (size=3) среди всех.
    """
    n_elite = max(1, int(elite_frac * N))
    order = np.argsort(-fit)
    elites = pop[order[:n_elite]].copy()
    # турнирный отбор для остальных
    rest = []
    for _ in range(N - n_elite):
        candidates = rng.choice(len(pop), size=3, replace=False)
        best = candidates[np.argmax(fit[candidates])]
        rest.append(pop[best])
    return np.vstack([elites, np.array(rest)])

def crossover(pop, Pc, rng):
    """Одноточечный кроссинговер (по лекции)."""
    new_pop = pop.copy()
    N = len(pop)
    for i in range(0, N - 1, 2):
        if rng.random() < Pc:
            k = rng.integers(1, N_BITS)   # точка скрещивания 1..N_BITS-1
            new_pop[i, k:], new_pop[i+1, k:] = pop[i+1, k:].copy(), pop[i, k:].copy()
    return new_pop

def mutate(pop, Pm, rng):
    """Побитовая мутация с вероятностью Pm."""
    mask = rng.random(pop.shape) < Pm
    return (pop ^ mask).astype(int)

# ============================================================
# 3. Основной цикл ГА
# ============================================================
def genetic_algorithm(selection='roulette', N=50, Pc=0.8, Pm=0.01,
                      n_gen=100, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    pop = init_population(N, rng)
    fit = np.array([fitness(c) for c in pop])
    history = {'best': [], 'mean': [], 'best_xy': []}
    t0 = time.perf_counter()

    for gen in range(n_gen):
        # ---- 1. отбор ----
        if selection == 'roulette':
            parents = selection_roulette(pop, fit, N, rng)
        else:  # elite
            parents = selection_elite(pop, fit, N, rng)
        # ---- 2. кроссинговер ----
        children = crossover(parents, Pc, rng)
        # ---- 3. мутация ----
        children = mutate(children, Pm, rng)
        # ---- 4. оценка ----
        child_fit = np.array([fitness(c) for c in children])
        # ---- 5. элитизм «поверх» для стабильности сравнения ----
        # (для честности: рулетка — без элитизма, элита — с элитизмом)
        if selection == 'elite':
            # лучшие из объединения pop+children
            combined = np.vstack([pop, children])
            combined_fit = np.concatenate([fit, child_fit])
            order = np.argsort(-combined_fit)[:N]
            pop = combined[order]
            fit = combined_fit[order]
        else:
            pop, fit = children, child_fit

        best_idx = int(np.argmax(fit))
        best_x, best_y = decode(pop[best_idx])
        history['best'].append(fit[best_idx])
        history['mean'].append(fit.mean())
        history['best_xy'].append((best_x, best_y))
        if verbose and (gen + 1) % 20 == 0:
            print(f"    gen {gen+1:3d}: best={fit[best_idx]:.6f}, "
                  f"x={best_x:+.4f}, y={best_y:+.4f}, mean={fit.mean():.4f}")
    elapsed = time.perf_counter() - t0

    best_idx = int(np.argmax(fit))
    best_x, best_y = decode(pop[best_idx])
    return {
        'best_x': best_x, 'best_y': best_y, 'best_f': fit[best_idx],
        'history': history, 'time': elapsed, 'generations': n_gen,
    }

# ============================================================
# 4. Эксперимент 1: сравнение рулетки и элиты (один запуск)
# ============================================================
print("="*64)
print("ЭКСПЕРИМЕНТ 1: рулетка vs элита (N=50, Pc=0.8, Pm=0.01, 100 ген.)")
print("="*64)

print("\nРулетка:")
r_roul = genetic_algorithm('roulette', N=50, Pc=0.8, Pm=0.01, n_gen=100, seed=1, verbose=True)
print(f"  → x={r_roul['best_x']:+.6f}, y={r_roul['best_y']:+.6f}, "
      f"f={r_roul['best_f']:.6f}, время={r_roul['time']:.3f}с")

print("\nЭлита:")
r_elite = genetic_algorithm('elite', N=50, Pc=0.8, Pm=0.01, n_gen=100, seed=1, verbose=True)
print(f"  → x={r_elite['best_x']:+.6f}, y={r_elite['best_y']:+.6f}, "
      f"f={r_elite['best_f']:.6f}, время={r_elite['time']:.3f}с")

print(f"\nИстинный оптимум: x=0, y=0, f=1.0")

# ============================================================
# 5. Эксперимент 2: 30 независимых запусков (статистика)
# ============================================================
print("\n" + "="*64)
print("ЭКСПЕРИМЕНТ 2: статистика по 30 запускам")
print("="*64)

N_RUNS = 30
stats = {'roulette': [], 'elite': []}
gens_to_99 = {'roulette': [], 'elite': []}

for run in range(N_RUNS):
    for sel in ['roulette', 'elite']:
        res = genetic_algorithm(sel, N=50, Pc=0.8, Pm=0.01, n_gen=100, seed=run)
        stats[sel].append(res['best_f'])
        # номер поколения, где best достиг 99% оптимума (0.99)
        h = res['history']['best']
        g99 = next((i + 1 for i, v in enumerate(h) if v >= 0.99), 100)
        gens_to_99[sel].append(g99)

print(f"\n{'Метрика':30s} {'Рулетка':>15s} {'Элита':>15s}")
print("-"*62)
for sel_name, key in [('Средний f_best', None), ('Мин f_best', None), ('Макс f_best', None)]:
    pass

for label, func in [('Средний f_best', np.mean),
                    ('Станд. отклонение f_best', np.std),
                    ('Мин f_best (худший запуск)', np.min),
                    ('Макс f_best (лучший запуск)', np.max)]:
    print(f"{label:32s} {func(stats['roulette']):15.6f} {func(stats['elite']):15.6f}")

print(f"{'Медиана поколений до f≥0.99':32s} "
      f"{np.median(gens_to_99['roulette']):15.1f} {np.median(gens_to_99['elite']):15.1f}")
print(f"{'Среднее поколений до f≥0.99':32s} "
      f"{np.mean(gens_to_99['roulette']):15.1f} {np.mean(gens_to_99['elite']):15.1f}")

# ============================================================
# 6. Эксперимент 3: зависимость от параметров
# ============================================================
print("\n" + "="*64)
print("ЭКСПЕРИМЕНТ 3: зависимость от N, Pc, Pm (элита, 20 запусков)")
print("="*64)

def run_avg(sel, N, Pc, Pm, runs=20, n_gen=100):
    vals = [genetic_algorithm(sel, N=N, Pc=Pc, Pm=Pm, n_gen=n_gen, seed=r)['best_f']
            for r in range(runs)]
    return np.mean(vals), np.std(vals)

print("\nВлияние размера популяции N (Pc=0.8, Pm=0.01):")
print(f"{'N':>5s} {'рулетка (mean±std)':>28s} {'элита (mean±std)':>28s}")
for N in [10, 20, 50, 100, 200]:
    m1, s1 = run_avg('roulette', N, 0.8, 0.01)
    m2, s2 = run_avg('elite',    N, 0.8, 0.01)
    print(f"{N:5d} {m1:14.6f} ± {s1:.4f} {m2:14.6f} ± {s2:.4f}")

print("\nВлияние вероятности кроссинговера Pc (N=50, Pm=0.01):")
print(f"{'Pc':>5s} {'рулетка (mean±std)':>28s} {'элита (mean±std)':>28s}")
for Pc in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
    m1, s1 = run_avg('roulette', 50, Pc, 0.01)
    m2, s2 = run_avg('elite',    50, Pc, 0.01)
    print(f"{Pc:5.1f} {m1:14.6f} ± {s1:.4f} {m2:14.6f} ± {s2:.4f}")

print("\nВлияние вероятности мутации Pm (N=50, Pc=0.8):")
print(f"{'Pm':>7s} {'рулетка (mean±std)':>28s} {'элита (mean±std)':>28s}")
for Pm in [0.0, 0.001, 0.01, 0.05, 0.1]:
    m1, s1 = run_avg('roulette', 50, 0.8, Pm)
    m2, s2 = run_avg('elite',    50, 0.8, Pm)
    print(f"{Pm:7.3f} {m1:14.6f} ± {s1:.4f} {m2:14.6f} ± {s2:.4f}")

# ============================================================
# 7. Визуализация
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 7.1 кривые сходимости (средние по 20 запускам)
ax = axes[0, 0]
runs_r = [genetic_algorithm('roulette', 50, 0.8, 0.01, 100, seed=r)
          for r in range(20)]
runs_e = [genetic_algorithm('elite', 50, 0.8, 0.01, 100, seed=r)
          for r in range(20)]
best_r = np.array([r['history']['best'] for r in runs_r])
best_e = np.array([r['history']['best'] for r in runs_e])
ax.plot(best_r.mean(axis=0), label='Рулетка', lw=2)
ax.fill_between(range(100), best_r.mean(axis=0)-best_r.std(axis=0),
                 best_r.mean(axis=0)+best_r.std(axis=0), alpha=0.2)
ax.plot(best_e.mean(axis=0), label='Элита', lw=2)
ax.fill_between(range(100), best_e.mean(axis=0)-best_e.std(axis=0),
                 best_e.mean(axis=0)+best_e.std(axis=0), alpha=0.2)
ax.axhline(1.0, color='k', ls=':', label='Оптимум f=1')
ax.set_xlabel("Поколение"); ax.set_ylabel("Лучший f")
ax.set_title("Кривые сходимости (среднее ± std по 20 запускам)")
ax.legend(); ax.grid(alpha=0.3)

# 7.2 поверхность функции с найденными точками
ax = axes[0, 1]
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
zz = 1.0 / (1.0 + xx**2 + yy**2)
cs = ax.contourf(xx, yy, zz, levels=30, cmap='viridis')
plt.colorbar(cs, ax=ax)
ax.plot(0, 0, 'r*', ms=20, label='Истинный оптимум (0,0)')
ax.plot(r_roul['best_x'], r_roul['best_y'], 'wo', ms=10,
        label=f"Рулетка ({r_roul['best_x']:+.2f},{r_roul['best_y']:+.2f})")
ax.plot(r_elite['best_x'], r_elite['best_y'], 'kx', ms=12, mew=2,
        label=f"Элита ({r_elite['best_x']:+.2f},{r_elite['best_y']:+.2f})")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Функция f(x,y) = 1/(1+x²+y²)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 7.3 распределение финальных значений по 30 запускам
ax = axes[1, 0]
# --- ИСПРАВЛЕНО: без параметра labels ---
ax.boxplot([stats['roulette'], stats['elite']])
ax.set_xticks([1, 2])
ax.set_xticklabels(['Рулетка', 'Элита'])
ax.axhline(1.0, color='k', ls=':', label='Оптимум')
ax.set_ylabel("Финальный f_best"); ax.set_title("Разброс по 30 запускам")
ax.grid(alpha=0.3); ax.legend()

# 7.4 зависимость от N
ax = axes[1, 1]
Ns = [10, 20, 50, 100, 200]
means_r, means_e = [], []
for N in Ns:
    m1, _ = run_avg('roulette', N, 0.8, 0.01, runs=10)
    m2, _ = run_avg('elite',    N, 0.8, 0.01, runs=10)
    means_r.append(m1); means_e.append(m2)
ax.plot(Ns, means_r, 'o-', label='Рулетка')
ax.plot(Ns, means_e, 's-', label='Элита')
ax.axhline(1.0, color='k', ls=':', label='Оптимум')
ax.set_xlabel("Размер популяции N"); ax.set_ylabel("Средний f_best")
ax.set_title("Зависимость точности от N")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ga_results.png", dpi=110)
plt.show()

print("\nСохранено: ga_results.png")
