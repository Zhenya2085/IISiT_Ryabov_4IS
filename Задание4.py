import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# 1. Шаблоны цифр 0..9, 10x10, биполярные {-1, +1}
# ============================================================
def make_templates():
    T = {}
    T[0] = ["  #####   ",
            " ##   ##  ",
            "##     ## ",
            "##     ## ",
            "##     ## ",
            "##     ## ",
            "##     ## ",
            " ##   ##  ",
            "  #####   ",
            "          "]
    T[1] = ["    ##    ",
            "   ###    ",
            "  ####    ",
            "    ##    ",
            "    ##    ",
            "    ##    ",
            "    ##    ",
            "    ##    ",
            "  ######  ",
            "          "]
    T[2] = ["  #####   ",
            " ##   ##  ",
            "      ##  ",
            "     ##   ",
            "    ##    ",
            "   ##     ",
            "  ##      ",
            " ##       ",
            " ######## ",
            "          "]
    T[3] = ["  #####   ",
            " ##   ##  ",
            "      ##  ",
            "   ####   ",
            "      ##  ",
            "      ##  ",
            "      ##  ",
            " ##   ##  ",
            "  #####   ",
            "          "]
    T[4] = ["     ##   ",
            "    ###   ",
            "   ####   ",
            "  ## ##   ",
            " ##  ##   ",
            "##########",
            "     ##   ",
            "     ##   ",
            "     ##   ",
            "          "]
    T[5] = [" ######## ",
            " ##       ",
            " ##       ",
            " ######   ",
            "      ##  ",
            "      ##  ",
            "      ##  ",
            " ##   ##  ",
            "  #####   ",
            "          "]
    T[6] = ["   ####   ",
            "  ##      ",
            " ##       ",
            " ######   ",
            " ##   ##  ",
            " ##   ##  ",
            " ##   ##  ",
            " ##   ##  ",
            "  #####   ",
            "          "]
    T[7] = [" ######## ",
            " ##   ##  ",
            "      ##  ",
            "     ##   ",
            "    ##    ",
            "    ##    ",
            "    ##    ",
            "    ##    ",
            "    ##    ",
            "          "]
    T[8] = ["  #####   ",
            " ##   ##  ",
            " ##   ##  ",
            "  #####   ",
            " ##   ##  ",
            " ##   ##  ",
            " ##   ##  ",
            " ##   ##  ",
            "  #####   ",
            "          "]
    T[9] = ["  #####   ",
            " ##   ##  ",
            " ##   ##  ",
            " ##   ##  ",
            "  ####### ",
            "      ##  ",
            "      ##  ",
            " ##   ##  ",
            "  #####   ",
            "          "]
    # биполярные: '#' -> +1, пробел -> -1
    return {d: np.array([1.0 if c == '#' else -1.0 for row in pat for c in row])
            for d, pat in T.items()}

TEMPLATES = make_templates()
N = 100

# ============================================================
# 2. Сеть Хопфилда
# ============================================================
class HopfieldNet:
    def __init__(self, n_neurons=100):
        self.N = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))

    # -------- обучение по правилу Хебба --------
    def train(self, patterns):
        """patterns: массив (M, N) со значениями {-1, +1}."""
        M = len(patterns)
        W = np.zeros((self.N, self.N))
        for p in patterns:
            W += np.outer(p, p)
        # Обнуление диагонали (w_ii = 0) и усреднение
        np.fill_diagonal(W, 0.0)
        self.W = W / M
        return self.W

    # -------- энергия Ляпунова --------
    def energy(self, s):
        return -0.5 * s @ self.W @ s

    # -------- синхронный/асинхронный шаг --------
    def step(self, s, mode='async', rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if mode == 'async':
            # обновляем нейроны по одному в случайном порядке (как в первоначальной сети Хопфилда)
            order = rng.permutation(self.N)
            for i in order:
                h_i = self.W[i] @ s
                s[i] = 1.0 if h_i >= 0 else -1.0
            return s, True
        else:
            # синхронное обновление (все нейроны одновременно)
            s_new = np.where(self.W @ s >= 0, 1.0, -1.0)
            changed = not np.array_equal(s_new, s)
            return s_new, changed

    # -------- восстановление образа --------
    def recall(self, x0, mode='async', max_iter=50, rng=None):
        s = x0.copy().astype(float)
        energies = [self.energy(s)]
        for _ in range(max_iter):
            s_new, changed = self.step(s, mode=mode, rng=rng)
            energies.append(self.energy(s_new))
            if not np.any(s_new != s):
                s = s_new
                break
            s = s_new
        return s, energies

# ============================================================
# 3. Искажение образов: шум и маскировка
# ============================================================
def add_noise(img, p_flip=0.15, rng=None):
    if rng is None: rng = np.random.default_rng()
    mask = rng.random(N) < p_flip
    out = img.copy()
    out[mask] *= -1
    return out

def mask_region(img, rng=None):
    """Закрывает случайный прямоугольник (дырка в образе)."""
    if rng is None: rng = np.random.default_rng()
    a = img.reshape(10, 10).copy()
    h, w = rng.integers(3, 6), rng.integers(3, 6)
    y, x = rng.integers(0, 10 - h + 1), rng.integers(0, 10 - w + 1)
    a[y:y+h, x:x+w] = -1.0
    return a.flatten()

# ============================================================
# 4. Обучение на всех 10 цифрах
# ============================================================
patterns = np.array([TEMPLATES[d] for d in range(10)])   # (10, 100)
net = HopfieldNet(N)
net.train(patterns)

print("="*60)
print("Сеть Хопфилда обучена на 10 цифрах (100 нейронов)")
print(f"Ёмкость: ~0.15*N = {0.15*N:.0f} образов; использовано {len(patterns)}")
print("="*60)

# ============================================================
# 5. Тест: шум + маска
# ============================================================
rng = np.random.default_rng(42)

def classify(s):
    """Ближайший (по Хэммингу) шаблон к состоянию s."""
    dists = [np.sum(s != p) for p in patterns]
    d = int(np.argmin(dists))
    return d, dists[d]

results = []
for d in range(10):
    clean = TEMPLATES[d]
    # (а) шум 15%
    noisy = add_noise(clean, p_flip=0.15, rng=rng)
    # (б) маска + шум 5%
    masked = add_noise(mask_region(clean, rng), p_flip=0.05, rng=rng)

    rec_n, E_n = net.recall(noisy,  mode='async', rng=rng)
    rec_m, E_m = net.recall(masked, mode='async', rng=rng)

    dig_n, dist_n = classify(rec_n)
    dig_m, dist_m = classify(rec_m)
    ok_n = (dig_n == d)
    ok_m = (dig_m == d)

    results.append((d, ok_n, ok_m, len(E_n)-1, len(E_m)-1))
    print(f"  Цифра {d}: шум->{dig_n} {'✓' if ok_n else '✗'} ({len(E_n)-1} итер.), "
          f"маска->{dig_m} {'✓' if ok_m else '✗'} ({len(E_m)-1} итер.)")

print(f"\nТочность восстановления (шум):   {sum(r[1] for r in results)}/10")
print(f"Точность восстановления (маска): {sum(r[2] for r in results)}/10")

# ============================================================
# 6. Демонстрация функции Ляпунова (энергия невозрастает)
# ============================================================
print("\n" + "="*60)
print("Проверка функции Ляпунова: энергия невозрастает")
print("="*60)
d0 = 5
x0 = add_noise(mask_region(TEMPLATES[d0], rng), p_flip=0.1, rng=rng)
_, E = net.recall(x0, mode='async', rng=rng)
diffs = np.diff(E)
print(f"  Стартовая энергия: {E[0]:.2f}, конечная: {E[-1]:.2f}")
print(f"  Шагов: {len(E)-1}, нарушений невозрастания: {int(np.sum(diffs > 1e-9))}")
print(f"  {'✓ энергия монотонно убывает' if np.all(diffs <= 1e-9) else '✗ ОШИБКА'}")

# ============================================================
# 7. Визуализация
# ============================================================
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for d in range(10):
    clean = TEMPLATES[d].reshape(10, 10)
    noisy = add_noise(clean.flatten(), p_flip=0.15, rng=np.random.default_rng(d)).reshape(10, 10)
    rec, _ = net.recall(noisy.flatten(), mode='async', rng=np.random.default_rng(d))
    axes[0, d].imshow(clean, cmap='gray_r');   axes[0, d].set_title(str(d), fontsize=8)
    axes[1, d].imshow(noisy, cmap='gray_r');   axes[1, d].set_title("шум", fontsize=8)
    axes[2, d].imshow(rec.reshape(10, 10), cmap='gray_r')
    axes[2, d].set_title("восст.", fontsize=8)
    for r in range(3):
        axes[r, d].axis('off')
axes[0, 0].set_ylabel("идеал", fontsize=9)
axes[1, 0].set_ylabel("вход", fontsize=9)
axes[2, 0].set_ylabel("выход", fontsize=9)
plt.suptitle("Сеть Хопфилда: восстановление цифр из зашумлённых образов")
plt.tight_layout()
plt.savefig("hopfield_digits.png", dpi=110)
plt.show()

# график энергии
plt.figure(figsize=(7, 3.5))
plt.plot(E, 'o-')
plt.xlabel("Итерация (асинхронный шаг)"); plt.ylabel("Энергия E")
plt.title("Динамика энергии сети Хопфилда при восстановлении")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hopfield_energy.png", dpi=110)
plt.show()

print("\nСохранено: hopfield_digits.png, hopfield_energy.png")