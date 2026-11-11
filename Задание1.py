import numpy as np

class Perceptron:
    """Однослойный перцептрон с пороговой функцией активации.
    Обучение по правилу Розенблатта (с учителем)."""

    def __init__(self, n_inputs, learning_rate=0.1, seed=42):
        rng = np.random.default_rng(seed)
        # Веса + порог (порог храним как вес при фиктивном входе -1)
        self.w = rng.uniform(-0.5, 0.5, n_inputs)
        self.theta = rng.uniform(-0.5, 0.5)
        self.lr = learning_rate

    def predict(self, x):
        net = np.dot(self.w, x) - self.theta
        return 1 if net >= 0 else 0

    def fit(self, X, y, max_epochs=1000, verbose=False):
        """X: (N, n) массив входов; y: (N,) правильные ответы."""
        history = []
        for epoch in range(max_epochs):
            errors = 0
            for xi, di in zip(X, y):
                yi = self.predict(xi)
                e = di - yi
                if e != 0:
                    # Правило коррекции: увеличить/уменьшить веса
                    self.w += self.lr * e * xi
                    self.theta -= self.lr * e
                    errors += 1
            history.append(errors)
            if verbose:
                print(f"  Эпоха {epoch+1}: ошибок = {errors}")
            if errors == 0:
                if verbose:
                    print(f"  Сходимость достигнута за {epoch+1} эпох(и).")
                break
        return history


def test_function(name, X, y):
    print(f"\n{'='*50}\nЛогическая функция: {name}\n{'='*50}")
    p = Perceptron(n_inputs=X.shape[1], learning_rate=0.1)
    p.fit(X, y, verbose=True)

    print(f"\n  Веса: {np.round(p.w, 3)}, порог θ = {round(p.theta, 3)}")
    print("  Проверка на всех входах:")
    ok = True
    for xi, di in zip(X, y):
        yi = p.predict(xi)
        mark = "✓" if yi == di else "✗"
        if yi != di:
            ok = False
        print(f"    {xi} -> {yi}  (ожидалось {di})  {mark}")
    print(f"  Итог: {'ФУНКЦИЯ ВЫУЧЕНА' if ok else 'ОШИБКА ОБУЧЕНИЯ'}")
    return ok


# ---------- Логическая функция И ----------
X_and = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_and = np.array([0, 0, 0, 1])
test_function("И (AND)", X_and, y_and)

# ---------- Логическая функция ИЛИ ----------
X_or = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y_or = np.array([0, 1, 1, 1])
test_function("ИЛИ (OR)", X_or, y_or)

# ---------- Логическая функция НЕ ----------
X_not = np.array([[0],
                  [1]])
y_not = np.array([1, 0])
test_function("НЕ (NOT)", X_not, y_not)

# ---------- Дополнительно: XOR (не должна выучиться) ----------
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_xor = np.array([0, 1, 1, 0])
print(f"\n{'='*50}\nXOR (ИСКЛЮЧАЮЩЕЕ ИЛИ) — линейно неразделима\n{'='*50}")
p_xor = Perceptron(n_inputs=2, learning_rate=0.1)
h = p_xor.fit(X_xor, y_xor, max_epochs=50)
print(f"  Ошибок в последней эпохе: {h[-1]} (ожидаемо > 0 — сходимости нет)")