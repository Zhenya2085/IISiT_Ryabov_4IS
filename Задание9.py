import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# ============================================================
# 1. Данные: XOR (главный тест) + AND/OR для проверки
# ============================================================
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0,1,1,0], dtype=float)

X_and = X_xor.copy(); y_and = np.array([0,0,0,1], dtype=float)
X_or  = X_xor.copy(); y_or  = np.array([0,1,1,1], dtype=float)


# ============================================================
# 2. Генетический алгоритм общего вида
# ============================================================
class GeneticAlgorithm:
    """
    Простой ГА по лекции (Холланд/Гольдберг):
      - вещественное кодирование (удобно для весов сети);
      - турнирный отбор;
      - арифметический кроссинговер (смешивание генов родителей);
      - гауссова мутация;
      - элитизм (лучший сохраняется).
    """
    def __init__(self, genome_size, fitness_fn, N=60, Pc=0.85, Pm=0.15,
                 sigma=0.5, bounds=(-5, 5), elite=2, seed=0):
        self.genome_size = genome_size
        self.fitness_fn = fitness_fn
        self.N = N
        self.Pc = Pc
        self.Pm = Pm
        self.sigma = sigma
        self.lo, self.hi = bounds
        self.elite = elite
        self.rng = np.random.default_rng(seed)

    def init_population(self):
        return self.rng.uniform(self.lo, self.hi, (self.N, self.genome_size))

    def tournament(self, pop, fit, k=3):
        idx = self.rng.choice(len(pop), size=k, replace=False)
        return pop[idx[np.argmax(fit[idx])]].copy()

    def crossover(self, p1, p2):
        if self.rng.random() < self.Pc:
            # арифметический (смешивающий) кроссинговер
            alpha = self.rng.uniform(0, 1, size=self.genome_size)
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = alpha * p2 + (1 - alpha) * p1
            return c1, c2
        return p1.copy(), p2.copy()

    def mutate(self, ind):
        mask = self.rng.random(self.genome_size) < self.Pm
        noise = self.rng.normal(0, self.sigma, self.genome_size)
        ind = ind.copy()
        ind[mask] += noise[mask]
        return np.clip(ind, self.lo, self.hi)

    def run(self, n_gen=200, verbose_every=None, verbose=False):
        pop = self.init_population()
        fit = np.array([self.fitness_fn(g) for g in pop])
        history = {'best': [], 'mean': []}
        t0 = time.perf_counter()
        for gen in range(n_gen):
            # ---- элитизм ----
            order = np.argsort(-fit)
            new_pop = [pop[i].copy() for i in order[:self.elite]]
            # ---- остальные через отбор/кроссинговер/мутацию ----
            while len(new_pop) < self.N:
                p1 = self.tournament(pop, fit)
                p2 = self.tournament(pop, fit)
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.N:
                    new_pop.append(self.mutate(c2))
            pop = np.array(new_pop)
            fit = np.array([self.fitness_fn(g) for g in pop])
            history['best'].append(fit.max())
            history['mean'].append(fit.mean())
            if verbose and verbose_every and (gen + 1) % verbose_every == 0:
                print(f"    gen {gen+1:3d}: best={fit.max():.4f}, "
                      f"mean={fit.mean():.4f}")
        elapsed = time.perf_counter() - t0
        best_idx = int(np.argmax(fit))
        return pop[best_idx], fit[best_idx], history, elapsed


# ============================================================
# 3. Однослойный перцептрон
# ============================================================
class SinglePerceptron:
    def __init__(self, n_in, n_out, seed=0):
        self.n_in = n_in
        self.n_out = n_out
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.uniform(-0.5, 0.5, (n_out, n_in))
        self.b = self.rng.uniform(-0.5, 0.5, n_out)

    def decode(self, genome):
        n = self.n_in * self.n_out
        W = genome[:n].reshape(self.n_out, self.n_in)
        b = genome[n:n + self.n_out]
        return W, b

    def genome_size(self):
        return self.n_in * self.n_out + self.n_out

    def predict(self, X, W=None, b=None):
        if W is None: W, b = self.W, self.b
        logits = X @ W.T + b
        if self.n_out == 1:
            # бинарный случай: порог по знаку
            return (logits >= 0).astype(int), logits
        # многоклассовый: argmax
        return np.argmax(logits, axis=1).reshape(-1, 1), logits

    def predict_from_genome(self, X, genome):
        W, b = self.decode(genome)
        return self.predict(X, W, b)

    def make_fitness(self, X, y):
        def fitness(genome):
            pred, logits = self.predict_from_genome(X, genome)
            acc = np.mean(pred.flatten() == y)
            margin = np.mean(np.abs(logits))
            return acc + 0.001 * margin
        return fitness

    def fit_rosenblatt(self, X, y, epochs=100, lr=0.1):
        """
        Правило Розенблатта. Поддерживает:
          - n_out == 1: бинарная классификация (метки 0/1),
            коррекция по ошибке (d - y) для единственного нейрона;
          - n_out > 1: one-vs-all (one-hot), коррекция по классам.
        """
        y = y.astype(int)

        if self.n_out == 1:
            # ---- бинарный перцептрон ----
            for _ in range(epochs):
                for xi, di in zip(X, y):
                    yi = 1 if (self.W[0] @ xi + self.b[0]) >= 0 else 0
                    e = di - yi
                    if e != 0:
                        self.W[0] += lr * e * xi
                        self.b[0] += lr * e
            return self

        # ---- многоклассовый (one-vs-all) ----
        y_oh = np.eye(self.n_out)[y]
        for _ in range(epochs):
            for xi, di in zip(X, y_oh):
                logits = self.W @ xi + self.b
                pred = int(np.argmax(logits))
                true = int(np.argmax(di))
                if pred != true:
                    self.W[true] += lr * xi
                    self.b[true] += lr
                    self.W[pred] -= lr * xi
                    self.b[pred] -= lr
        return self


# ============================================================
# 4. Многослойный перцептрон (2 -> H -> 1)
# ============================================================
class MLP:
    def __init__(self, n_in=2, n_hidden=4, n_out=1, seed=0):
        self.n_in, self.n_h, self.n_out = n_in, n_hidden, n_out
        self.rng = np.random.default_rng(seed)

    def genome_size(self):
        return self.n_in * self.n_h + self.n_h + self.n_h * self.n_out + self.n_out

    def decode(self, genome):
        i = 0
        W1 = genome[i:i + self.n_in * self.n_h].reshape(self.n_h, self.n_in); i += W1.size
        b1 = genome[i:i + self.n_h]; i += self.n_h
        W2 = genome[i:i + self.n_h * self.n_out].reshape(self.n_out, self.n_h); i += W2.size
        b2 = genome[i:i + self.n_out]
        return W1, b1, W2, b2

    def forward(self, X, genome):
        W1, b1, W2, b2 = self.decode(genome)
        A1 = np.tanh(X @ W1.T + b1)        # скрытый слой
        logits = A1 @ W2.T + b2            # выход
        return logits, A1

    def predict(self, X, genome):
        logits, _ = self.forward(X, genome)
        return (logits.flatten() >= 0).astype(int)

    def make_fitness(self, X, y):
        def fitness(genome):
            pred = self.predict(X, genome)
            acc = np.mean(pred == y)
            _, A1 = self.forward(X, genome)
            margin = np.mean(np.abs(A1))
            return acc + 0.001 * margin
        return fitness


# ============================================================
# 5. Эксперимент 1: однослойный перцептрон на XOR через ГА
# ============================================================
print("="*66)
print("ЭКСПЕРИМЕНТ 1: однослойный перцептрон на XOR — ГА vs Розенблатт")
print("="*66)

# 5.1 ГА
sp = SinglePerceptron(n_in=2, n_out=1, seed=0)
ga1 = GeneticAlgorithm(
    genome_size=sp.genome_size(),
    fitness_fn=sp.make_fitness(X_xor, y_xor),
    N=60, Pc=0.85, Pm=0.15, sigma=0.4, bounds=(-5,5), elite=2, seed=0,
)
best_genome, best_fit, hist1, t1 = ga1.run(n_gen=200, verbose=True, verbose_every=40)
pred_ga, logits_ga = sp.predict_from_genome(X_xor, best_genome)
print(f"\n  ГА:  fitness={best_fit:.4f}, время={t1:.3f}с")
print(f"  Предсказания ГА: {pred_ga.flatten().tolist()}  (истина: {y_xor.astype(int).tolist()})")
print(f"  Точность ГА: {np.mean(pred_ga.flatten()==y_xor)*100:.1f}%")
if np.mean(pred_ga.flatten()==y_xor) < 1.0:
    print("  ⚠ XOR линейно неразделима → однослойный перцептрон НЕ МОЖЕТ её выучить (Минский).")

# 5.2 Розенблатт
sp2 = SinglePerceptron(n_in=2, n_out=1, seed=0)
t2_start = time.perf_counter()
sp2.fit_rosenblatt(X_xor, y_xor.astype(int), epochs=100, lr=0.1)
t2 = time.perf_counter() - t2_start
pred_rb, _ = sp2.predict(X_xor)
print(f"\n  Розенблатт: время={t2:.3f}с")
print(f"  Предсказания: {pred_rb.flatten().tolist()}  (истина: {y_xor.astype(int).tolist()})")
print(f"  Точность: {np.mean(pred_rb.flatten()==y_xor)*100:.1f}%")
print("  Вывод: оба метода терпят крах на XOR — ограничение архитектуры, а не метода.")

# ============================================================
# 6. Эксперимент 2: однослойный перцептрон на AND / OR через ГА
# ============================================================
print("\n" + "="*66)
print("ЭКСПЕРИМЕНТ 2: однослойный перцептрон на AND / OR через ГА")
print("="*66)

for name, X, y in [('AND', X_and, y_and), ('OR', X_or, y_or)]:
    sp = SinglePerceptron(n_in=2, n_out=1, seed=1)
    ga = GeneticAlgorithm(
        genome_size=sp.genome_size(),
        fitness_fn=sp.make_fitness(X, y),
        N=40, Pc=0.85, Pm=0.15, sigma=0.4, bounds=(-5,5), elite=2, seed=1,
    )
    bg, bf, hist, dt = ga.run(n_gen=100)
    pred, _ = sp.predict_from_genome(X, bg)
    acc = np.mean(pred.flatten() == y) * 100
    print(f"  {name}: fitness={bf:.4f}, точность={acc:.0f}%, время={dt:.3f}с, "
          f"предсказания={pred.flatten().astype(int).tolist()}")

# ============================================================
# 7. Эксперимент 3: многослойный перцептрон на XOR через ГА
# ============================================================
print("\n" + "="*66)
print("ЭКСПЕРИМЕНТ 3: многослойный перцептрон (2→4→1) на XOR через ГА")
print("="*66)

mlp = MLP(n_in=2, n_hidden=4, n_out=1, seed=0)
print(f"  Размер генома: {mlp.genome_size()} вещественных чисел")

ga3 = GeneticAlgorithm(
    genome_size=mlp.genome_size(),
    fitness_fn=mlp.make_fitness(X_xor, y_xor),
    N=80, Pc=0.9, Pm=0.2, sigma=0.6, bounds=(-5,5), elite=2, seed=0,
)
best_genome_mlp, best_fit_mlp, hist3, t3 = ga3.run(n_gen=400, verbose=True, verbose_every=80)
pred_mlp = mlp.predict(X_xor, best_genome_mlp)
print(f"\n  Лучший fitness: {best_fit_mlp:.4f}, время: {t3:.3f}с")
print(f"  Предсказания: {pred_mlp.tolist()}  (истина: {y_xor.astype(int).tolist()})")
print(f"  Точность: {np.mean(pred_mlp == y_xor)*100:.0f}%")

# ============================================================
# 8. Эксперимент 4: статистика по 20 запускам — сравнение методов
# ============================================================
print("\n" + "="*66)
print("ЭКСПЕРИМЕНТ 4: статистика по 20 запускам")
print("="*66)

def run_ga_single(seed):
    sp = SinglePerceptron(2, 1, seed=seed)
    ga = GeneticAlgorithm(sp.genome_size(), sp.make_fitness(X_xor, y_xor),
                          N=60, Pc=0.85, Pm=0.15, sigma=0.4, seed=seed)
    _, bf, _, dt = ga.run(n_gen=150)
    return bf, dt

def run_ga_mlp(seed):
    m = MLP(2, 4, 1, seed=seed)
    ga = GeneticAlgorithm(m.genome_size(), m.make_fitness(X_xor, y_xor),
                          N=80, Pc=0.9, Pm=0.2, sigma=0.6, seed=seed)
    _, bf, _, dt = ga.run(n_gen=300)
    return bf, dt

ga1_fits, ga1_times = zip(*[run_ga_single(s) for s in range(20)])
ga3_fits, ga3_times = zip(*[run_ga_mlp(s)    for s in range(20)])

print(f"\n{'Метод':38s} {'fitness средн.':>15s} {'fitness std':>12s} {'время, с':>10s}")
print("-"*78)
print(f"{'ГА + однослойный (XOR)':38s} {np.mean(ga1_fits):15.4f} {np.std(ga1_fits):12.4f} {np.mean(ga1_times):10.3f}")
print(f"{'ГА + многослойный (XOR)':38s} {np.mean(ga3_fits):15.4f} {np.std(ga3_fits):12.4f} {np.mean(ga3_times):10.3f}")
print(f"\n  Примечание: fitness=1.0 — идеальная точность (учитывая +0.001*margin).")

# ============================================================
# 9. Визуализация
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 9.1 кривые обучения ГА (однослойный XOR vs MLP XOR)
ax = axes[0, 0]
ax.plot(hist1['best'], label='Однослойный (XOR)', lw=2)
ax.plot(hist3['best'], label='Многослойный (XOR)', lw=2)
ax.set_xlabel("Поколение"); ax.set_ylabel("Best fitness")
ax.set_title("ГА: однослойный vs многослойный на XOR")
ax.legend(); ax.grid(alpha=0.3)

# 9.2 AND/OR: кривые обучения
ax = axes[0, 1]
for name, X, y, color in [('AND', X_and, y_and, 'C0'), ('OR', X_or, y_or, 'C1')]:
    sp = SinglePerceptron(2, 1, seed=1)
    ga = GeneticAlgorithm(sp.genome_size(), sp.make_fitness(X, y),
                          N=40, Pc=0.85, Pm=0.15, sigma=0.4, seed=1)
    _, _, h, _ = ga.run(n_gen=100)
    ax.plot(h['best'], label=name, lw=2, color=color)
ax.set_xlabel("Поколение"); ax.set_ylabel("Best fitness")
ax.set_title("ГА: простые функции (AND, OR)")
ax.legend(); ax.grid(alpha=0.3)

# 9.3 XOR: разделяющие границы — однослойный vs MLP (после ГА)
ax = axes[1, 0]
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 200), np.linspace(-0.2, 1.2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# однослойный: линейная граница
_, lg = sp.predict_from_genome(grid, best_genome)
Z_sp = lg.reshape(xx.shape)
ax.contour(xx, yy, Z_sp, levels=[0], colors='red', linestyles='--', linewidths=2)

# MLP: нелинейная граница
_, _ = mlp.forward(grid, best_genome_mlp)
Z_mlp, _ = mlp.forward(grid, best_genome_mlp)
ax.contour(xx, yy, Z_mlp.reshape(xx.shape), levels=[0], colors='blue', linewidths=2)

ax.scatter(X_xor[y_xor==0,0], X_xor[y_xor==0,1], s=150, c='black', marker='o',
           label='класс 0', zorder=5)
ax.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1], s=150, c='gold', marker='s',
           label='класс 1', zorder=5)
ax.set_title("XOR: красный — однослойный (не справляется),\nсиний — MLP (справляется)")
ax.legend(loc='upper right'); ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.2, 1.2)
ax.grid(alpha=0.3)

# 9.4 боксплоты
ax = axes[1, 1]
ax.boxplot([ga1_fits, ga3_fits])
ax.set_xticks([1, 2])
ax.set_xticklabels(['Однослойный (XOR)', 'Многослойный (XOR)'])
ax.axhline(1.0, color='k', ls=':', label='Идеал')
ax.set_ylabel("Финальный fitness")
ax.set_title("Разброс по 20 запускам ГА")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ga_perceptron.png", dpi=110)
plt.show()

print("\nСохранено: ga_perceptron.png")
