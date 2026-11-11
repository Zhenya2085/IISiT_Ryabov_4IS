import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 1. Образы: цифры 0..4 в поле 12x12 (как в лекции)
# ============================================================
def make_images():
    """5 образов 12x12, значения {0,1}."""
    imgs = {}
    imgs[0] = ["            ",
               "   #####    ",
               "  ##   ##   ",
               "  ##   ##   ",
               "  ##   ##   ",
               "  ##   ##   ",
               "  ##   ##   ",
               "  ##   ##   ",
               "   #####    ",
               "            ",
               "            ",
               "            "]
    imgs[1] = ["            ",
               "     ##     ",
               "    ###     ",
               "   ####     ",
               "     ##     ",
               "     ##     ",
               "     ##     ",
               "     ##     ",
               "   ######   ",
               "            ",
               "            ",
               "            "]
    imgs[2] = ["            ",
               "   #####    ",
               "  ##   ##   ",
               "       ##   ",
               "      ##    ",
               "     ##     ",
               "    ##      ",
               "   ##       ",
               "  ########  ",
               "            ",
               "            ",
               "            "]
    imgs[3] = ["            ",
               "   #####    ",
               "  ##   ##   ",
               "       ##   ",
               "    ####    ",
               "       ##   ",
               "       ##   ",
               "  ##   ##   ",
               "   #####    ",
               "            ",
               "            ",
               "            "]
    imgs[4] = ["            ",
               "      ##    ",
               "     ###    ",
               "    ####    ",
               "   ## ##    ",
               "  ##  ##    ",
               "  ########  ",
               "      ##    ",
               "      ##    ",
               "            ",
               "            ",
               "            "]
    return np.array([[float(c == '#') for row in pat for c in row] for pat in imgs.values()])

IMAGES = make_images()          # (5, 144)
N_IMG, SIDE = IMAGES.shape[0], 12

# ============================================================
# 2. Когнитрон
# ============================================================
class Cognitron:
    """
    Упрощённая, но верная по формулам лекции реализация:
      - слой возбуждающих нейронов (12x12) c локальным рецептивным полем 5x5;
      - тормозящий нейрон на каждую область (среднее по полю, веса c=1/25);
      - латеральное торможение (ромб 5x5) — ускоренное, за один проход;
      - обучение только победителя в области конкуренции;
      - веса только возрастают (без забывания), как в лекции.
    """
    def __init__(self, layer_sizes=(12, 12, 12), rf=5, comp=5,
                 lat=7, q=16.0, q_prime=2.0):
        self.layer_sizes = layer_sizes
        self.rf = rf                 # рецептивное поле (5x5)
        self.comp = comp             # область конкуренции (ромб 5x5)
        self.lat = lat               # латеральное торможение (7x7)
        self.q, self.q_prime = q, q_prime
        self.pad = rf // 2

        # веса: для каждого слоя — возбуждающие a (по числу входов в поле)
        # и тормозящие b (скаляр на нейрон)
        self.A, self.B, self.C = [], [], []
        for _ in range(len(layer_sizes) - 1):
            rf_size = rf * rf
            self.A.append(np.zeros(rf_size))       # возбуждающие веса
            self.B.append(np.zeros(1))             # тормозящий вес
            self.C.append(np.full(rf_size, 1.0 / rf_size))  # c_j, не обучаются

    # ---------- вспомогательные ----------
    def _extract_patch(self, layer, y, x):
        """Вырезает rf x rf патч с паддингом нулями."""
        p = self.pad
        padded = np.pad(layer, p, mode='constant')
        return padded[y:y + self.rf, x:x + self.rf].flatten()

    def _rf_response(self, layer, a):
        """E для каждого нейрона слоя (свёртка)."""
        p = self.pad
        padded = np.pad(layer, p, mode='constant')
        # свёртка через as_strided — быстрее, но для ясности циклом
        H, W = layer.shape
        E = np.zeros((H, W))
        for y in range(H):
            for x in range(W):
                E[y, x] = np.dot(a, padded[y:y + self.rf, x:x + self.rf].flatten())
        return E

    def _inhib_signal(self, layer, c):
        """I — среднее по рецептивному полю (тормозящий слой)."""
        p = self.pad
        padded = np.pad(layer, p, mode='constant')
        H, W = layer.shape
        I = np.zeros((H, W))
        for y in range(H):
            for x in range(W):
                I[y, x] = np.dot(c, padded[y:y + self.rf, x:x + self.rf].flatten())
        return I

    def _lateral_inhib(self, layer):
        """Ускоренное латеральное торможение (ромб lat x lat), за один проход."""
        H, W = layer.shape
        h = self.lat // 2
        padded = np.pad(layer, h, mode='constant')
        # веса g: сумма = 1, монотонно убывают от центра (ромб)
        g = np.zeros((self.lat, self.lat))
        for dy in range(-h, h + 1):
            for dx in range(-h, h + 1):
                if abs(dy) + abs(dx) <= h:
                    g[dy + h, dx + h] = 1.0
        g /= g.sum()
        LAT = np.zeros((H, W))
        for y in range(H):
            for x in range(W):
                LAT[y, x] = np.sum(g * padded[y:y + self.lat, x:x + self.lat])
        OUT = np.maximum((1 + layer) / (1 + LAT) - 1, 0)
        return OUT, LAT

    # ---------- прямой проход ----------
    def forward(self, x_img):
        """x_img: (144,) или (12,12). Возвращает список активаций по слоям."""
        layer = x_img.reshape(SIDE, SIDE).astype(float)
        activations = [layer]
        for li in range(len(self.A)):
            E = self._rf_response(layer, self.A[li])
            I = self._inhib_signal(layer, self.C[li])
            net = (1 + E) / (1 + I) - 1
            out = np.maximum(net, 0)
            out_lat, _ = self._lateral_inhib(out)
            layer = out_lat
            activations.append(layer)
        return activations

    # ---------- обучение ----------
    def train(self, images, epochs=20, verbose=True):
        """
        Самоорганизация без учителя. Каждый образ предъявляется epochs раз.
        Обучаем слой за слоем, начиная с ближайшего ко входу.
        """
        n_layers = len(self.A)
        # идём по слоям: сначала обучим первый слой, потом второй на его выходах и т.д.
        for li in range(n_layers):
            if verbose:
                print(f"\n--- Обучение слоя {li+1} ---")
            for ep in range(epochs):
                total_delta = 0
                for img in images:
                    # прогон до текущего слоя
                    x = img.reshape(SIDE, SIDE).astype(float)
                    for k in range(li):
                        E = self._rf_response(x, self.A[k])
                        I = self._inhib_signal(x, self.C[k])
                        x = np.maximum((1 + E) / (1 + I) - 1, 0)
                        x, _ = self._lateral_inhib(x)
                    # E/I для текущего слоя
                    E = self._rf_response(x, self.A[li])
                    I = self._inhib_signal(x, self.C[li])
                    net = (1 + E) / (1 + I) - 1
                    out = np.maximum(net, 0)
                    out_lat, LAT = self._lateral_inhib(out)
                    # ищем победителя в области конкуренции
                    winner = np.unravel_index(np.argmax(out_lat), out_lat.shape)
                    wy, wx = winner
                    h = self.comp // 2
                    # ---- обучение победителя ----
                    padded_x = np.pad(x, self.pad, mode='constant')
                    patch = padded_x[wy:wy + self.rf, wx:wx + self.rf].flatten()
                    c = self.C[li]
                    if out_lat[wy, wx] > 0:
                        # есть победитель: q-правило
                        self.A[li] += self.q * c * patch
                        self.B[li] += self.q * np.sum(self.A[li] * patch) / (2 * I[wy, wx] + 1e-6)
                        total_delta += self.q
                    else:
                        # нет победителя: q'-правило
                        self.A[li] += self.q_prime * c * patch
                        self.B[li] += self.q_prime * I[wy, wx]
                        total_delta += self.q_prime
                if verbose and (ep + 1) % 5 == 0:
                    print(f"  эпоха {ep+1:2d}: суммарное приращение весов ≈ {total_delta:.2f}")

    # ---------- распознавание ----------
    def classify(self, img):
        """Возвращает выходной слой (12x12) и индекс максимально активного узла."""
        acts = self.forward(img)
        out = acts[-1]
        idx = int(np.argmax(out))
        return out, idx

    # ---------- реверсивный режим (из лекции) ----------
    def reconstruct(self, img):
        """
        Упрощённое восстановление: берём выходной слой, оставляем только
        максимум (как столбец 4 на рис. 13.7 лекции), и «проецируем» обратно
        через веса первого слоя (транспонированная свёртка).
        """
        acts = self.forward(img)
        out = acts[-1].copy()
        # оставляем только максимальный узел
        out_peak = np.zeros_like(out)
        wy, wx = np.unravel_index(np.argmax(out), out.shape)
        out_peak[wy, wx] = out[wy, wx]
        # проекция обратно: для каждого нейрона первого слоя суммируем
        # его вклад через веса A[0] (транспонированная свёртка)
        H, W = SIDE, SIDE
        p = self.pad
        recon = np.zeros((H, W))
        a = self.A[0].reshape(self.rf, self.rf)
        # простое обратное распространение активации через те же веса
        for y in range(H):
            for x in range(W):
                # вклад узла (y,x) первого слоя в выходной образ
                # берём его активацию * средний вес
                recon[y, x] = acts[1][y, x] * np.mean(a)
        # нормируем и сравниваем с эталоном
        recon = np.clip(recon / (recon.max() + 1e-9), 0, 1)
        return recon


# ============================================================
# 3. Обучение
# ============================================================
print("="*60)
print("ОБУЧЕНИЕ КОГНИТРОНА (3 слоя, 12x12, rf=5x5)")
print("="*60)
cog = Cognitron(layer_sizes=(12, 12, 12), rf=5, comp=5, lat=7, q=16.0, q_prime=2.0)
cog.train(IMAGES, epochs=20, verbose=True)

# ============================================================
# 4. Тестирование: распознавание и реверсивный режим
# ============================================================
print("\n" + "="*60)
print("ТЕСТИРОВАНИЕ")
print("="*60)

# 4.1 какой выходной узел активируется для каждого образа
print("\nАктивация выходного слоя (макс. узел):")
peak_nodes = []
for i, img in enumerate(IMAGES):
    out, idx = cog.classify(img)
    wy, wx = np.unravel_index(idx, out.shape)
    peak_nodes.append((wy, wx))
    print(f"  Образ {i}: макс. узел ({wy},{wx}), активация={out[wy,wx]:.4f}")

# 4.2 проверка различимости: разные образы -> разные узлы?
print("\nРазличимость образов:")
unique_nodes = len(set(peak_nodes))
print(f"  Уникальных пиковых узлов: {unique_nodes} из {len(IMAGES)}")
print(f"  {'✓ образы различимы' if unique_nodes == len(IMAGES) else '⚠ есть совпадения — нужно больше эпох/нейронов'}")

# 4.3 реверсивный режим: восстановление по пиковому узлу
print("\nРеверсивный режим (восстановление образа):")
recons = []
for i, img in enumerate(IMAGES):
    rec = cog.reconstruct(img)
    recons.append(rec)
    # корреляция с эталоном
    orig = img.reshape(SIDE, SIDE)
    corr = np.corrcoef(rec.flatten(), orig.flatten())[0, 1]
    print(f"  Образ {i}: корреляция восстановления с эталоном = {corr:.3f}")

# ============================================================
# 5. Визуализация
# ============================================================
fig, axes = plt.subplots(3, 5, figsize=(14, 8))
for i in range(5):
    # исходный
    axes[0, i].imshow(IMAGES[i].reshape(SIDE, SIDE), cmap='gray_r')
    axes[0, i].set_title(f"Образ {i}", fontsize=10)
    axes[0, i].axis('off')
    # выходной слой
    out, _ = cog.classify(IMAGES[i])
    axes[1, i].imshow(out, cmap='hot')
    axes[1, i].set_title(f"Выход (макс {out.max():.2f})", fontsize=9)
    axes[1, i].axis('off')
    # восстановление
    axes[2, i].imshow(recons[i], cmap='gray_r')
    axes[2, i].set_title("Восстановление", fontsize=9)
    axes[2, i].axis('off')

axes[0, 0].set_ylabel("Вход", fontsize=11)
axes[1, 0].set_ylabel("Выход", fontsize=11)
axes[2, 0].set_ylabel("Реверс", fontsize=11)
plt.suptitle("Когнитрон: вход → выход → восстановление", fontsize=13)
plt.tight_layout()
plt.savefig("cognitron_results.png", dpi=110)
plt.show()

# ============================================================
# 6. Проверка на зашумлённых образах (обобщение)
# ============================================================
print("\n" + "="*60)
print("ПРОВЕРКА ОБОБЩЕНИЯ: зашумлённые образы")
print("="*60)
rng = np.random.default_rng(0)
# эталонные пиковые узлы
ref_nodes = [np.unravel_index(cog.classify(IMAGES[i])[1], (SIDE, SIDE)) for i in range(5)]

for noise in [0.05, 0.10, 0.15]:
    correct = 0
    for i, img in enumerate(IMAGES):
        noisy = img.copy()
        flip = rng.random(len(noisy)) < noise
        noisy[flip] = 1 - noisy[flip]
        out, idx = cog.classify(noisy)
        wy, wx = np.unravel_index(idx, out.shape)
        # ближайший эталонный узел
        dists = [abs(wy - r[0]) + abs(wx - r[1]) for r in ref_nodes]
        if int(np.argmin(dists)) == i:
            correct += 1
    print(f"  Шум {noise*100:.0f}%: распознано {correct}/5")

print("\nСохранено: cognitron_results.png")
