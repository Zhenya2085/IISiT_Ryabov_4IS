import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N = 100   # размерность входа (10x10)

# ============================================================
# 1. Шаблоны цифр 0..9, 10x10 (те же, что в сети Хопфилда)
# ============================================================
def make_templates():
    T = {}
    T[0] = ["  #####   "," ##   ##  ","##     ## ","##     ## ",
            "##     ## ","##     ## ","##     ## "," ##   ##  ","  #####   ","          "]
    T[1] = ["    ##    ","   ###    ","  ####    ","    ##    ",
            "    ##    ","    ##    ","    ##    ","    ##    ","  ######  ","          "]
    T[2] = ["  #####   "," ##   ##  ","      ##  ","     ##   ",
            "    ##    ","   ##     ","  ##      "," ##       "," ######## ","          "]
    T[3] = ["  #####   "," ##   ##  ","      ##  ","   ####   ",
            "      ##  ","      ##  ","      ##  "," ##   ##  ","  #####   ","          "]
    T[4] = ["     ##   ","    ###   ","   ####   ","  ## ##   ",
            " ##  ##   ","##########","     ##   ","     ##   ","     ##   ","          "]
    T[5] = [" ######## "," ##       "," ##       "," ######   ",
            "      ##  ","      ##  ","      ##  "," ##   ##  ","  #####   ","          "]
    T[6] = ["   ####   ","  ##      "," ##       "," ######   ",
            " ##   ##  "," ##   ##  "," ##   ##  "," ##   ##  ","  #####   ","          "]
    T[7] = [" ######## "," ##   ##  ","      ##  ","     ##   ",
            "    ##    ","    ##    ","    ##    ","    ##    ","    ##    ","          "]
    T[8] = ["  #####   "," ##   ##  "," ##   ##  ","  #####   ",
            " ##   ##  "," ##   ##  "," ##   ##  "," ##   ##  ","  #####   ","          "]
    T[9] = ["  #####   "," ##   ##  "," ##   ##  "," ##   ##  ",
            "  ####### ","      ##  ","      ##  "," ##   ##  ","  #####   ","          "]
    return {d: np.array([1.0 if c == '#' else -1.0 for row in pat for c in row])
            for d, pat in T.items()}

TEMPLATES = make_templates()
PATTERNS = np.array([TEMPLATES[d] for d in range(10)])   # (10, 100) биполярные

# ============================================================
# 2. Сеть Хопфилда (из предыдущего задания)
# ============================================================
class HopfieldNet:
    def __init__(self, n):
        self.N = n
        self.W = np.zeros((n, n))
    def train(self, patterns):
        W = patterns.T @ patterns
        np.fill_diagonal(W, 0)
        self.W = W / len(patterns)
    def energy(self, s):
        return -0.5 * s @ self.W @ s
    def step_async(self, s, rng):
        for i in rng.permutation(self.N):
            s[i] = 1.0 if self.W[i] @ s >= 0 else -1.0
        return s
    def recall(self, x0, max_iter=50, rng=None):
        if rng is None: rng = np.random.default_rng()
        s = x0.copy().astype(float)
        for _ in range(max_iter):
            s_old = s.copy()
            s = self.step_async(s, rng)
            if np.array_equal(s, s_old):
                break
        return s

# ============================================================
# 3. Сеть Хэмминга (по лекции, биполярный вариант {-1,+1})
# ============================================================
class HammingNet:
    """
    M нейронов в каждом из двух слоёв.
    Слой 1: считает сходство входа с каждым образцом
            NET_j = (1/2) * sum_i x_i^(j) * s_i   (при {-1,+1})
            то есть M - dist_Hamming(x^(j), s) / 2  (чем больше — тем ближе)
    Слой 2: «латеральное торможение» — winner-take-all:
            NET_j = y_j + self_weight*y_j - eps * sum_{k!=j} y_k
            с порогом, обеспечивающим единственного победителя.
    """
    def __init__(self, patterns, eps=0.5):
        self.patterns = patterns.astype(float)   # (M, n)
        self.M, self.n = patterns.shape
        # Слой 1: w1_j = x^(j) / 2,  theta1_j = 0 (биполярный вариант)
        self.W1 = self.patterns / 2.0            # (M, n)
        self.theta1 = np.zeros(self.M)
        # Слой 2: самовозбуждение +1, торможение -eps
        self.eps = eps
        self.W2 = np.full((self.M, self.M), -eps)
        np.fill_diagonal(self.W2, 1.0 + eps)     # self-вес = 1 + eps, чтобы NET_j = y_j - eps*sum_{k!=j} y_k

    def layer1(self, s):
        """Активация первого слоя (сходство с образцами)."""
        return self.W1 @ s - self.theta1        # (M,)

    def layer2_step(self, y1, y2):
        """Один шаг латерального торможения. Возвращает новое y2 и флаг изменения."""
        net2 = self.W2 @ y2 + y1               # учёт входа от первого слоя
        y2_new = (net2 > 0).astype(float)
        return y2_new, not np.array_equal(y2_new, y2)

    def predict(self, s, max_iter=50, verbose=False):
        """Возвращает индекс победившего нейрона (= номер распознанной цифры)."""
        s = s.astype(float)
        y1 = self.layer1(s)                    # непрерывные «сходства»
        # Инициализация второго слоя: как в лекции — значениями аксонов 1-го слоя,
        # но бинаризуем по порогу 0, чтобы стартовать с нескольких кандидатов
        y2 = (y1 > 0).astype(float)
        if y2.sum() == 0:
            y2 = np.zeros(self.M); y2[np.argmax(y1)] = 1.0
        for it in range(max_iter):
            y2_new, changed = self.layer2_step(y1, y2)
            if verbose:
                print(f"    итер {it+1}: активны {np.where(y2_new > 0)[0].tolist()}")
            y2 = y2_new
            if not changed:
                break
        winner = int(np.argmax(y2)) if y2.sum() > 0 else int(np.argmax(y1))
        return winner, y1, y2

# ============================================================
# 4. Искажения: шум + маска (как в предыдущем задании)
# ============================================================
def add_noise(img, p_flip=0.15, rng=None):
    if rng is None: rng = np.random.default_rng()
    out = img.copy()
    out[rng.random(N) < p_flip] *= -1
    return out

def mask_region(img, rng=None):
    if rng is None: rng = np.random.default_rng()
    a = img.reshape(10, 10).copy()
    h, w = rng.integers(3, 6), rng.integers(3, 6)
    y, x = rng.integers(0, 10 - h + 1), rng.integers(0, 10 - w + 1)
    a[y:y+h, x:x+w] = -1.0
    return a.flatten()

# ============================================================
# 5. Двухступенчатый конвейер: Хопфилд → Хэмминг
# ============================================================
hopfield = HopfieldNet(N); hopfield.train(PATTERNS)
hamming  = HammingNet(PATTERNS, eps=0.5)

rng = np.random.default_rng(7)

print("="*72)
print("КОНВЕЙЕР: повреждённая цифра → Хопфилд → Хэмминг")
print("="*72)

stats = {'hop_ok': 0, 'ham_ok': 0, 'total': 0, 'wrong': []}

for d in range(10):
    clean = TEMPLATES[d]

    # разные степени искажения
    cases = {
        'шум 10%':        add_noise(clean, 0.10, rng),
        'шум 20%':        add_noise(clean, 0.20, rng),
        'маска+шум 5%':   add_noise(mask_region(clean, rng), 0.05, rng),
        'маска+шум 15%':  add_noise(mask_region(clean, rng), 0.15, rng),
    }

    for name, x in cases.items():
        # --- Ступень 1: Хопфилд восстанавливает образ ---
        rec = hopfield.recall(x, rng=rng)
        # проверка: восстановление совпало с эталоном?
        hop_ok = np.array_equal(rec, clean)
        # --- Ступень 2: Хэмминг выдаёт номер ---
        pred, y1, y2 = hamming.predict(rec)
        ham_ok = (pred == d)

        stats['total'] += 1
        stats['hop_ok'] += int(hop_ok)
        stats['ham_ok'] += int(ham_ok)
        if not ham_ok:
            stats['wrong'].append((d, name, pred, round(float(rec @ clean), 1)))

        mark1 = "✓" if hop_ok else "≈"   # ≈ — восстановлен не идеально, но близко
        mark2 = "✓" if ham_ok else "✗"
        print(f"  {d} | {name:14s} | Хопфилд {mark1} | Хэмминг -> {pred} {mark2}")

print("-"*72)
print(f"  Хопфилд восстановил точно: {stats['hop_ok']}/{stats['total']}")
print(f"  Хэмминг классифицировал:  {stats['ham_ok']}/{stats['total']}")
print("="*72)

# ============================================================
# 6. Детальная демонстрация одного примера (итерации WTA)
# ============================================================
print("\n" + "="*72)
print("ДЕТАЛЬНО: цифра 5, шум 20%")
print("="*72)
d = 5
x = add_noise(TEMPLATES[d], 0.20, rng)
rec = hopfield.recall(x, rng=rng)
print(f"  Хопфилд: сходство с эталоном 5 = {rec @ TEMPLATES[d]:.0f}/100 бит")
pred, y1, y2 = hamming.predict(rec, verbose=True)
print(f"  Хэмминг: победил нейрон {pred}")
print(f"  Сходства y1 (M - dist/2): {np.round(y1, 1)}")
print(f"  Итоговые активации y2:   {y2.astype(int).tolist()}")

# ============================================================
# 7. Визуализация конвейера
# ============================================================
fig, axes = plt.subplots(3, 10, figsize=(15, 5.5))
rng = np.random.default_rng(11)
for d in range(10):
    clean = TEMPLATES[d].reshape(10, 10)
    noisy = add_noise(TEMPLATES[d], 0.20, rng).reshape(10, 10)
    rec   = hopfield.recall(noisy.flatten(), rng=rng).reshape(10, 10)
    pred, _, _ = hamming.predict(rec.flatten())

    axes[0, d].imshow(clean, cmap='gray_r'); axes[0, d].set_title(f"{d}", fontsize=9)
    axes[1, d].imshow(noisy, cmap='gray_r'); axes[1, d].set_title("шум", fontsize=8)
    axes[2, d].imshow(rec,   cmap='gray_r')
    axes[2, d].set_title(f"→{pred}", fontsize=9,
                         color='green' if pred == d else 'red')
    for r in range(3): axes[r, d].axis('off')

axes[0, 0].set_ylabel("эталон",   fontsize=10)
axes[1, 0].set_ylabel("вход",     fontsize=10)
axes[2, 0].set_ylabel("Хопфилд\n→Хэмминг", fontsize=10)
plt.suptitle("Конвейер: Хопфилд восстанавливает образ → Хэмминг распознаёт номер")
plt.tight_layout()
plt.savefig("hopfield_hamming_pipeline.png", dpi=110)
plt.show()

# ============================================================
# 8. Сравнение «Хэмминг на сыром входе» vs «Хэмминг после Хопфилда»
# ============================================================
print("\n" + "="*72)
print("СРАВНЕНИЕ: Хэмминг по сырому входу vs после Хопфилда")
print("="*72)
rng = np.random.default_rng(2024)
raw_ok, pipe_ok, total = 0, 0, 0
for d in range(10):
    for _ in range(20):
        x = add_noise(mask_region(TEMPLATES[d], rng), 0.15, rng)
        # напрямую
        p_raw, _, _ = hamming.predict(x)
        # после Хопфилда
        rec = hopfield.recall(x, rng=rng)
        p_pipe, _, _ = hamming.predict(rec)
        raw_ok  += int(p_raw  == d)
        pipe_ok += int(p_pipe == d)
        total   += 1
print(f"  Всего тестов: {total}")
print(f"  Хэмминг по сырому входу:       {raw_ok/total*100:5.1f}%")
print(f"  Хэмминг после Хопфилда:        {pipe_ok/total*100:5.1f}%")
print(f"  Прирост точности:              +{(pipe_ok-raw_ok)/total*100:.1f} п.п.")