import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(42)
N = 100          # размерность входа (10x10)
N_CLASSES = 10   # цифры 0..9

# ============================================================
# 1. Шаблоны цифр 10x10, {0,1} — как в перцептроне из Лекции 1
# ============================================================
def make_templates_binary():
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
    return {d: np.array([1.0 if c == '#' else 0.0 for row in pat for c in row])
            for d, pat in T.items()}

TEMPLATES = make_templates_binary()
BASE = np.array([TEMPLATES[d] for d in range(10)])   # (10, 100)

# ============================================================
# 2. Генерация обучающей/тестовой выборки (шум + сдвиги)
# ============================================================
def shift(img, dx, dy):
    a = img.reshape(10, 10)
    out = np.zeros_like(a)
    xs0, xs1 = max(0, dx), min(10, 10 + dx)
    ys0, ys1 = max(0, dy), min(10, 10 + dy)
    xd0, xd1 = max(0, -dx), min(10, 10 - dx)
    yd0, yd1 = max(0, -dy), min(10, 10 - dy)
    out[ys0:ys1, xs0:xs1] = a[yd0:yd1, xd0:xd1]
    return out.flatten()

def make_dataset(n_per_digit, rng, noise=0.03):
    X, y = [], []
    for d in range(10):
        for _ in range(n_per_digit):
            img = BASE[d].copy()
            if rng.random() < 0.7:
                img = shift(img, rng.integers(-1, 2), rng.integers(-1, 2))
            flip = rng.random(N) < noise
            img[flip] = 1.0 - img[flip]
            X.append(img); y.append(d)
    X, y = np.array(X), np.array(y)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

rng = np.random.default_rng(7)
X_train, y_train = make_dataset(80, rng)   # 800 примеров
X_test,  y_test  = make_dataset(20, rng)   # 200 примеров
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# 3. Сеть встречного распространения
# ============================================================
class Counterpropagation:
    """
    Прямое встречное распространение:
      слой 1 (Кохонен):  W (n_in x K),  WTA, обучение без учителя
      слой 2 (Гроссберг): V (K x n_out), обучение с учителем
    """
    def __init__(self, n_in, n_kohonen, n_out, alpha0=0.7, beta0=0.1, seed=42):
        rng_ = np.random.default_rng(seed)
        # --- метод выпуклой комбинации (convex combination) ---
        # все веса = 1/sqrt(n), затем обучаем на «сжатых» входах
        self.W = np.full((n_in, n_kohonen), 1.0 / np.sqrt(n_in)) + \
                 rng_.normal(0, 0.01, (n_in, n_kohonen))
        self.V = rng_.uniform(-0.1, 0.1, (n_kohonen, n_out))
        self.alpha0, self.beta0 = alpha0, beta0
        self.K, self.n_in, self.n_out = n_kohonen, n_in, n_out

    # ---------- нормализация ----------
    @staticmethod
    def _norm(x):
        n = np.linalg.norm(x)
        return x / n if n > 1e-12 else x

    # ---------- Кохонен: кто победитель ----------
    def _kohonen_net(self, x):
        return x @ self.W                 # (K,)

    def _winner(self, x):
        return int(np.argmax(self._kohonen_net(x)))

    # ---------- обучение ----------
    def fit(self, X, Y_onehot, epochs=20, method='convex', verbose=False):
        """
        method:
          'convex' — метод выпуклой комбинации (из лекции, надёжнее);
          'plain'  — обычный Кохонен со случайными весами;
          'fair'   — обычный + «чувство справедливости».
        Y_onehot: (N, n_out)
        """
        n = len(X)
        K = self.K
        win_count = np.zeros(K, dtype=int)
        t_total = 0
        for ep in range(epochs):
            alpha = self.alpha0 * (1 - ep / epochs) + 0.05
            beta  = self.beta0  * (1 - ep / epochs) + 0.01
            # --- прогресс выпуклой комбинации ---
            a_conv = min(1.0, 0.1 + 0.9 * ep / max(1, epochs - 1)) if method == 'convex' else 1.0
            order = rng.permutation(n)
            t0 = time.perf_counter()
            for i in order:
                x = X[i].astype(float)
                x_c = a_conv * x + (1 - a_conv) / np.sqrt(self.n_in) if method == 'convex' else x
                net = x_c @ self.W
                # чувство справедливости
                if method == 'fair':
                    net = net - 0.3 * win_count / (ep + 1)
                j = int(np.argmax(net))
                win_count[j] += 1
                # --- Кохонен: подстройка весов победителя ---
                self.W[:, j] += alpha * (x_c - self.W[:, j])
                # --- Гроссберг: подстройка весов от победителя ---
                self.V[j] += beta * (Y_onehot[i] - self.V[j])
            t_total += time.perf_counter() - t0
            if verbose:
                acc = self.score(X, np.argmax(Y_onehot, axis=1))
                print(f"  эпоха {ep+1:2d}: alpha={alpha:.3f}, beta={beta:.3f}, "
                      f"accuracy={acc*100:5.1f}%, время={time.perf_counter()-t0:.3f}s")
        return t_total

    # ---------- предсказание ----------
    def predict_proba(self, X):
        # Слой Кохонена: WTA
        nets = X @ self.W                     # (N, K)
        Kvec = np.zeros_like(nets)
        Kvec[np.arange(len(X)), np.argmax(nets, axis=1)] = 1.0
        # Слой Гроссберга
        return Kvec @ self.V

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # ---------- размеры и число весов ----------
    @property
    def n_weights(self):
        return self.W.size + self.V.size

    def n_active_weights(self):
        """Сколько весов реально участвуют в одном прямом проходе."""
        return self.n_in + self.n_out      # W-строка победителя + V-строка


# ============================================================
# 4. Однослойный перцептрон (для сравнения)
# ============================================================
class SinglePerceptron:
    def __init__(self, n_in, n_out, lr=0.01, seed=42):
        rng_ = np.random.default_rng(seed)
        self.W = rng_.uniform(-0.1, 0.1, (n_out, n_in))
        self.b = np.zeros(n_out)
        self.lr = lr
    def net(self, x):
        return self.W @ x + self.b
    def predict(self, X):
        return np.argmax(X @ self.W.T + self.b, axis=1)
    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))
    def fit(self, X, y, epochs=30, verbose=False):
        n = len(X)
        t_total = 0.0
        for ep in range(epochs):
            errs = 0
            t0 = time.perf_counter()
            for i in rng.permutation(n):
                x, d = X[i], y[i]
                pred = int(np.argmax(self.net(x)))
                if pred != d:
                    errs += 1
                    self.W[d]    += self.lr * x
                    self.b[d]    += self.lr
                    self.W[pred] -= self.lr * x
                    self.b[pred] -= self.lr
            t_total += time.perf_counter() - t0
            if verbose:
                print(f"  эпоха {ep+1:2d}: ошибок {errs:4d}, "
                      f"accuracy={self.score(X,y)*100:5.1f}%, время={time.perf_counter()-t0:.3f}s")
            if errs == 0:
                break
        return t_total
    @property
    def n_weights(self):
        return self.W.size + self.b.size


# ============================================================
# 5. One-hot кодировка
# ============================================================
def one_hot(y, n):
    Y = np.zeros((len(y), n))
    Y[np.arange(len(y)), y] = 1.0
    return Y
Y_train_oh = one_hot(y_train, N_CLASSES)
Y_test_oh  = one_hot(y_test,  N_CLASSES)

# ============================================================
# 6. Эксперимент 1: CP с разным числом нейронов Кохонена
# ============================================================
print("\n" + "="*72)
print("ЭКСПЕРИМЕНТ 1: Сеть встречного распространения (метод выпуклой комбинации)")
print("="*72)
cp_results = {}
for K in [10, 20, 40, 80]:
    net = Counterpropagation(N, K, N_CLASSES, alpha0=0.7, beta0=0.1, seed=K)
    t = net.fit(X_train, Y_train_oh, epochs=15, method='convex')
    acc_tr = net.score(X_train, y_train)
    acc_te = net.score(X_test,  y_test)
    cp_results[K] = dict(acc_tr=acc_tr, acc_te=acc_te, time=t, weights=net.n_weights)
    print(f"  K={K:3d}: train={acc_tr*100:5.1f}%  test={acc_te*100:5.1f}%  "
          f"time={t:6.3f}s  weights={net.n_weights}")

# ============================================================
# 7. Эксперимент 2: CP с разными методами инициализации
# ============================================================
print("\n" + "="*72)
print("ЭКСПЕРИМЕНТ 2: Методы обучения слоя Кохонена (K=40)")
print("="*72)
for method, name in [('plain', 'обычный Кохонен'),
                     ('fair',  'чувство справедливости'),
                     ('convex','выпуклая комбинация')]:
    net = Counterpropagation(N, 40, N_CLASSES, alpha0=0.7, beta0=0.1, seed=123)
    t = net.fit(X_train, Y_train_oh, epochs=15, method=method)
    print(f"  {name:24s}: train={net.score(X_train,y_train)*100:5.1f}%  "
          f"test={net.score(X_test,y_test)*100:5.1f}%  time={t:6.3f}s")

# ============================================================
# 8. Эксперимент 3: однослойный перцептрон
# ============================================================
print("\n" + "="*72)
print("ЭКСПЕРИМЕНТ 3: Однослойный перцептрон (100 → 10)")
print("="*72)
for lr in [0.01, 0.05]:
    sp = SinglePerceptron(N, N_CLASSES, lr=lr, seed=1)
    t = sp.fit(X_train, y_train, epochs=30)
    print(f"  lr={lr:4.2f}: train={sp.score(X_train,y_train)*100:5.1f}%  "
          f"test={sp.score(X_test,y_test)*100:5.1f}%  time={t:6.3f}s  "
          f"weights={sp.n_weights}")

# ============================================================
# 9. Итоговое сравнение (лучшая CP vs лучший SP)
# ============================================================
print("\n" + "="*72)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print("="*72)

# лучшая CP по test accuracy
best_K = max(cp_results, key=lambda k: cp_results[k]['acc_te'])
best_cp = cp_results[best_K]

# лучший SP
sp = SinglePerceptron(N, N_CLASSES, lr=0.01, seed=1)
t_sp = sp.fit(X_train, y_train, epochs=30)
best_sp = dict(acc_tr=sp.score(X_train, y_train),
               acc_te=sp.score(X_test, y_test),
               time=t_sp, weights=sp.n_weights)

# число "эффективных" вычислений на 1 предсказание
cp = Counterpropagation(N, best_K, N_CLASSES); cp.W = np.full((N, best_K), 1/np.sqrt(N))
cp_ops_per_pred  = N * best_K + N_CLASSES            # W^T x + V k
sp_ops_per_pred  = N * N_CLASSES                     # W x

header = f"{'Метрика':32s} {'Встречное распр.':>20s} {'Односл. перцептрон':>20s}"
print(header); print("-" * len(header))
print(f"{'Точность на train':32s} {best_cp['acc_tr']*100:18.1f}% {best_sp['acc_tr']*100:18.1f}%")
print(f"{'Точность на test':32s} {best_cp['acc_te']*100:18.1f}% {best_sp['acc_te']*100:18.1f}%")
print(f"{'Время обучения (с)':32s} {best_cp['time']:18.3f}  {best_sp['time']:18.3f}")
print(f"{'Всего весов':32s} {best_cp['weights']:18d}  {best_sp['weights']:18d}")
print(f"{'Операций на предсказание':32s} {cp_ops_per_pred:18d}  {sp_ops_per_pred:18d}")

# ============================================================
# 10. Визуализация
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

# (а) точность vs число нейронов Кохонена
Ks = sorted(cp_results.keys())
tr = [cp_results[k]['acc_tr'] * 100 for k in Ks]
te = [cp_results[k]['acc_te'] * 100 for k in Ks]
axes[0].plot(Ks, tr, 'o-', label='train')
axes[0].plot(Ks, te, 's--', label='test')
axes[0].axhline(best_sp['acc_te']*100, color='red', ls=':', label='SP test')
axes[0].set_xlabel("Число нейронов Кохонена K")
axes[0].set_ylabel("Точность, %")
axes[0].set_title("Встречное распространение: точность vs K")
axes[0].legend(); axes[0].grid(alpha=0.3)

# (б) время обучения vs K
times = [cp_results[k]['time'] for k in Ks]
axes[1].plot(Ks, times, 'o-', label='CP time')
axes[1].axhline(best_sp['time'], color='red', ls=':', label='SP time')
axes[1].set_xlabel("Число нейронов Кохонена K")
axes[1].set_ylabel("Время обучения, с")
axes[1].set_title("Время обучения")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("cp_vs_perceptron.png", dpi=110)
plt.show()

# ============================================================
# 11. Демонстрация распознавания (CP)
# ============================================================
net = Counterpropagation(N, best_K, N_CLASSES, seed=best_K)
net.fit(X_train, Y_train_oh, epochs=15, method='convex')

fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
rng_vis = np.random.default_rng(99)
for d in range(10):
    idx = np.where(y_test == d)[0][0]
    img = X_test[idx].reshape(10, 10)
    pred_cp = int(net.predict(X_test[idx:idx+1])[0])
    pred_sp = int(sp.predict(X_test[idx:idx+1])[0])
    axes[0, d].imshow(img, cmap='gray_r')
    axes[0, d].set_title(str(d), fontsize=9)
    axes[1, d].imshow(img, cmap='gray_r')
    axes[1, d].set_title(f"CP:{pred_cp} SP:{pred_sp}", fontsize=8,
                         color='green' if pred_cp == d else 'red')
    axes[0, d].axis('off'); axes[1, d].axis('off')
plt.suptitle("Верх: тестовый пример. Низ: предсказания CP и SP")
plt.tight_layout()
plt.savefig("cp_digits.png", dpi=110)
plt.show()

print("\nСохранено: cp_vs_perceptron.png, cp_digits.png")