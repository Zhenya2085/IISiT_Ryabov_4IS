import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 1. Двухслойный перцептрон (скрытый слой + выходной)
#    Активация: сигмоида. Обучение: обратное распространение.
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(a):        # a = sigmoid(z)
    return a * (1.0 - a)

class MLP:
    """
    Полносвязная сеть: n_in -> n_hidden -> n_out.
    Один скрытый слой с сигмоидой, выход — сигмоида (бинарная классификация).
    """
    def __init__(self, n_in, n_hidden, n_out=1, lr=0.5, seed=42):
        rng = np.random.default_rng(seed)
        # инициализация Ксавье — важно для стабильного обучения
        self.W1 = rng.normal(0, np.sqrt(2.0/(n_in + n_hidden)),  (n_hidden, n_in))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0/(n_hidden + n_out)), (n_out, n_hidden))
        self.b2 = np.zeros(n_out)
        self.lr = lr

    def forward(self, X):
        """X: (N, n_in). Возвращает выходы и кэш для backprop."""
        Z1 = X @ self.W1.T + self.b1      # (N, H)
        A1 = sigmoid(Z1)                  # (N, H)
        Z2 = A1 @ self.W2.T + self.b2     # (N, O)
        A2 = sigmoid(Z2)                  # (N, O)
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X, y, epochs=5000, verbose_every=1000):
        """y: (N,) или (N,1) значения 0/1. Полный батч (batch GD)."""
        y = y.reshape(-1, 1).astype(float)
        N = len(X)
        history = []
        for ep in range(1, epochs + 1):
            # --- forward ---
            A2, (X_, Z1, A1, Z2, _) = self.forward(X)
            # --- loss (binary cross-entropy) ---
            eps = 1e-12
            loss = -np.mean(y*np.log(A2+eps) + (1-y)*np.log(1-A2+eps))
            history.append(loss)
            # --- backward ---
            dZ2 = (A2 - y) / N                 # (N, O)
            dW2 = dZ2.T @ A1                   # (O, H)
            db2 = dZ2.sum(axis=0)
            dA1 = dZ2 @ self.W2                # (N, H)
            dZ1 = dA1 * sigmoid_deriv(A1)      # (N, H)
            dW1 = dZ1.T @ X_                   # (H, n_in)
            db1 = dZ1.sum(axis=0)
            # --- update (gradient descent) ---
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            if verbose_every and ep % verbose_every == 0:
                acc = np.mean(self.predict(X).flatten() == y.flatten())
                print(f"  Эпоха {ep:5d}: loss={loss:.6f}, точность={acc*100:5.1f}%")
        return history


# ============================================================
# 2. Однослойный перцептрон (для сравнения на XOR)
# ============================================================
class SinglePerceptron:
    def __init__(self, n_in, lr=0.1, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-0.5, 0.5, n_in)
        self.b = rng.uniform(-0.5, 0.5)
        self.lr = lr
    def predict(self, x):
        return 1 if (self.w @ x + self.b) >= 0 else 0
    def fit(self, X, y, epochs=500):
        history = []
        for _ in range(epochs):
            errs = 0
            for xi, di in zip(X, y):
                yi = self.predict(xi)
                e = di - yi
                if e != 0:
                    self.w += self.lr * e * xi
                    self.b += self.lr * e
                    errs += 1
            history.append(errs)
        return history


# ============================================================
# 3. Наборы данных
# ============================================================
X2 = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_and = np.array([0,0,0,1])
y_or  = np.array([0,1,1,1])
y_xor = np.array([0,1,1,0])
X1 = np.array([[0],[1]], dtype=float)
y_not = np.array([1,0])

# ============================================================
# 4. Сравнение: однослойный vs двухслойный на XOR
# ============================================================
print("="*60)
print("ЧАСТЬ 1. XOR: однослойный перцептрон")
print("="*60)
sp = SinglePerceptron(2, lr=0.1)
h = sp.fit(X2, y_xor, epochs=200)
final_err = h[-1]
pred = [sp.predict(x) for x in X2]
print(f"  Ошибок в последней эпохе: {final_err}")
print(f"  Предсказания: {pred}  (истина: {y_xor.tolist()})")
print(f"  Итог: {'ВЫУЧЕНА' if final_err == 0 else 'НЕ ВЫУЧЕНА (линейно неразделима)'}")

print("\n" + "="*60)
print("ЧАСТЬ 2. XOR: двухслойный перцептрон (2 -> 4 -> 1)")
print("="*60)
mlp_xor = MLP(n_in=2, n_hidden=4, n_out=1, lr=0.5, seed=1)
loss_hist = mlp_xor.fit(X2, y_xor, epochs=5000, verbose_every=1000)
pred = mlp_xor.predict(X2).flatten()
prob = mlp_xor.predict_proba(X2).flatten()
print(f"\n  Вероятности: {np.round(prob, 4)}")
print(f"  Предсказания: {pred.tolist()}  (истина: {y_xor.tolist()})")
print(f"  Итог: {'ВЫУЧЕНА' if np.array_equal(pred, y_xor) else 'НЕ ВЫУЧЕНА'}")

# ============================================================
# 5. Проверка: та же архитектура на простых функциях
# ============================================================
print("\n" + "="*60)
print("ЧАСТЬ 3. Двухслойная сеть на простых функциях")
print("="*60)

tasks = [
    ("И (AND)",      X2, y_and),
    ("ИЛИ (OR)",     X2, y_or),
    ("НЕ (NOT)",     X1, y_not),
    ("XOR",          X2, y_xor),
    ("XNOR (НЕ-XOR)", X2, (1 - y_xor)),
    ("И-НЕ (NAND)",  X2, (1 - y_and)),
]

results = {}
for name, X, y in tasks:
    n_in = X.shape[1]
    # небольшая сеть: 4 скрытых нейрона — с запасом для всех этих функций
    net = MLP(n_in=n_in, n_hidden=4, n_out=1, lr=0.5, seed=7)
    hist = net.fit(X, y, epochs=5000, verbose_every=0)   # без вывода
    pred = net.predict(X).flatten()
    ok = np.array_equal(pred, y)
    results[name] = (ok, hist[-1], pred.tolist(), y.tolist())
    status = "✓ ВЫУЧЕНА" if ok else "✗ НЕ ВЫУЧЕНА"
    print(f"  {name:14s} loss={hist[-1]:.6f}  {status}")
    print(f"      предсказания {pred.tolist()}  истина {y.tolist()}")

# ============================================================
# 6. Визуализация
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# 6.1 Разделяющие границы на XOR: однослойный не справляется, MLP — да
ax = axes[0]
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 200), np.linspace(-0.2, 1.2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# однослойный: линейная граница w·x + b = 0
Z_sp = (grid @ sp.w + sp.b).reshape(xx.shape)
ax.contour(xx, yy, Z_sp, levels=[0], colors='red', linestyles='--', linewidths=2)

# MLP: нелинейная граница 0.5
Z_mlp = mlp_xor.predict_proba(grid).reshape(xx.shape)
ax.contour(xx, yy, Z_mlp, levels=[0.5], colors='blue', linewidths=2)

ax.scatter(X2[y_xor==0,0], X2[y_xor==0,1], s=150, c='black', marker='o', label='класс 0', zorder=5)
ax.scatter(X2[y_xor==1,0], X2[y_xor==1,1], s=150, c='gold', marker='s', label='класс 1', zorder=5)
ax.set_title("XOR: красная пунктирная — однослойный,\nсиняя — двухслойный (0.5)")
ax.legend(loc='upper right'); ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.2, 1.2)
ax.grid(alpha=0.3)

# 6.2 Кривые обучения
ax = axes[1]
ax.plot(loss_hist, label='XOR (2→4→1)', lw=2)
ax.set_xlabel("Эпоха"); ax.set_ylabel("Binary cross-entropy")
ax.set_title("Кривая обучения MLP на XOR")
ax.grid(alpha=0.3); ax.legend()
ax.set_yscale('log')

plt.tight_layout()
plt.savefig("mlp_xor.png", dpi=100)
plt.show()

# ============================================================
# 7. Итоговая таблица
# ============================================================
print("\n" + "="*60)
print("ИТОГОВАЯ ТАБЛИЦА")
print("="*60)
print(f"{'Функция':16s} {'Однослойный':>12s} {'Двухслойный':>12s}")
print("-"*44)
single_ok = {
    "И (AND)": True, "ИЛИ (OR)": True, "НЕ (NOT)": True,
    "XOR": False, "XNOR (НЕ-XOR)": False, "И-НЕ (NAND)": True,
}
for name in ["И (AND)", "ИЛИ (OR)", "НЕ (NOT)", "XOR", "XNOR (НЕ-XOR)", "И-НЕ (NAND)"]:
    s = "✓" if single_ok[name] else "✗"
    m = "✓" if results[name][0] else "✗"
    print(f"{name:16s} {s:>12s} {m:>12s}")
