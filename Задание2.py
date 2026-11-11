import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 1. Датасет: 10x10 пиксельные цифры (шаблоны 0..9)
# ============================================================
def make_digit_templates():
    """Возвращает словарь {digit: 10x10 array of 0/1}."""
    T = {}
    # 0
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
    # 1
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
    # 2
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
    # 3
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
    # 4
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
    # 5
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
    # 6
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
    # 7
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
    # 8
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
    # 9
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
    return {d: np.array([[1.0 if c == '#' else 0.0 for c in row] for row in pat]).flatten()
            for d, pat in T.items()}

TEMPLATES = make_digit_templates()

# ============================================================
# 2. Генерация обучающей выборки с шумом и сдвигами
# ============================================================
def add_noise(img, p_flip=0.03):
    """Случайно переворачивает пиксели с вероятностью p_flip."""
    mask = np.random.rand(100) < p_flip
    out = img.copy()
    out[mask] = 1.0 - out[mask]
    return out

def shift(img, dx, dy):
    """Сдвиг 10x10 изображения (заполнение нулями)."""
    a = img.reshape(10, 10)
    out = np.zeros_like(a)
    xs0, xs1 = max(0, dx), min(10, 10 + dx)
    ys0, ys1 = max(0, dy), min(10, 10 + dy)
    xd0, xd1 = max(0, -dx), min(10, 10 - dx)
    yd0, yd1 = max(0, -dy), min(10, 10 - dy)
    out[ys0:ys1, xs0:xs1] = a[yd0:yd1, xd0:xd1]
    return out.flatten()

def make_dataset(n_per_digit=60):
    X, y = [], []
    for d in range(10):
        base = TEMPLATES[d]
        for _ in range(n_per_digit):
            img = base.copy()
            # случайный сдвиг -1..1
            if np.random.rand() < 0.5:
                img = shift(img, np.random.randint(-1, 2), np.random.randint(-1, 2))
            # шум
            img = add_noise(img, p_flip=0.02)
            X.append(img)
            y.append(d)
    X = np.array(X)
    y = np.array(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

X_train, y_train = make_dataset(60)
X_test,  y_test  = make_dataset(20)
print(f"Обучающая выборка: {X_train.shape}, тест: {X_test.shape}")

# ============================================================
# 3. Многоклассовый однослойный перцептрон
# ============================================================
class MultiPerceptron:
    """
    100 входов -> 10 выходных нейронов.
    Каждый нейрон — независимый линейный классификатор «свой/чужой».
    Обучение по правилу Розенблатта (один против всех, one-hot).
    """
    def __init__(self, n_in=100, n_out=10, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-0.1, 0.1, (n_out, n_in))  # веса
        self.b = np.zeros(n_out)                        # пороги (b = -theta)
        self.lr = lr

    def net(self, x):
        return self.W @ x + self.b   # (10,)

    def predict(self, x):
        return int(np.argmax(self.net(x)))

    def fit(self, X, y, epochs=30, verbose=True):
        n = len(X)
        history = []
        for ep in range(epochs):
            errs = 0
            order = np.random.permutation(n)
            for i in order:
                x, d = X[i], y[i]
                out = self.net(x)
                pred = int(np.argmax(out))
                if pred != d:
                    errs += 1
                    # «Правильное» правило: +lr*x для класса d,
                    # -lr*x для ошибочно выигравшего класса pred
                    self.W[d]    += self.lr * x
                    self.b[d]    += self.lr
                    self.W[pred] -= self.lr * x
                    self.b[pred] -= self.lr
            acc = 1.0 - errs / n
            history.append((errs, acc))
            if verbose:
                print(f"  Эпоха {ep+1:2d}: ошибок {errs:4d}/{n}  точность {acc*100:5.2f}%")
            if errs == 0:
                break
        return history

    def accuracy(self, X, y):
        return np.mean([self.predict(x) == d for x, d in zip(X, y)])

# ============================================================
# 4. Обучение
# ============================================================
print("\n=== Обучение однослойного перцептрона (100 -> 10) ===")
model = MultiPerceptron(n_in=100, n_out=10, lr=0.01)
hist = model.fit(X_train, y_train, epochs=25)

print(f"\nТочность на обучающей выборке: {model.accuracy(X_train, y_train)*100:.2f}%")
print(f"Точность на тестовой выборке:  {model.accuracy(X_test, y_test)*100:.2f}%")

# ============================================================
# 5. Матрица ошибок на тесте
# ============================================================
cm = np.zeros((10, 10), dtype=int)
for x, d in zip(X_test, y_test):
    cm[d, model.predict(x)] += 1

print("\nМатрица ошибок (строки — истина, столбцы — предсказание):")
print("     " + "".join(f"{c:4d}" for c in range(10)))
for r in range(10):
    print(f"  {r}: " + "".join(f"{cm[r, c]:4d}" for c in range(10)))

# ============================================================
# 6. Визуализация: шаблоны, пример предсказания, кривая обучения
# ============================================================
fig, axes = plt.subplots(2, 10, figsize=(14, 3.5))
for d in range(10):
    axes[0, d].imshow(TEMPLATES[d].reshape(10, 10), cmap='gray_r')
    axes[0, d].set_title(str(d), fontsize=9)
    axes[0, d].axis('off')

# по одному тестовому примеру на каждую цифру
for d in range(10):
    idx = np.where(y_test == d)[0][0]
    img = X_test[idx].reshape(10, 10)
    pred = model.predict(X_test[idx])
    axes[1, d].imshow(img, cmap='gray_r')
    color = 'green' if pred == d else 'red'
    axes[1, d].set_title(f"→{pred}", fontsize=9, color=color)
    axes[1, d].axis('off')
plt.suptitle("Сверху: шаблоны. Снизу: тестовые примеры и предсказание")
plt.tight_layout()
plt.savefig("digits_predictions.png", dpi=100)
plt.show()

# кривая обучения
errs = [h[0] for h in hist]
plt.figure(figsize=(7, 3))
plt.plot(range(1, len(errs)+1), errs, 'o-')
plt.xlabel("Эпоха"); plt.ylabel("Число ошибок"); plt.grid(True)
plt.title("Кривая обучения однослойного перцептрона")
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=100)
plt.show()

print("\nГотово. Сохранены: digits_predictions.png, learning_curve.png")