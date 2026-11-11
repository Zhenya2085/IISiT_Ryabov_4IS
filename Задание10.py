import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider
import matplotlib.animation as animation

np.random.seed(42)

# ============================================================
# 1. Параметры модели
# ============================================================
class Config:
    GRID = 60                # размер среды (квадрат)
    N_PLANTS = 300           # начальное число растений
    PLANT_GROW_RATE = 0.15   # скорость роста растений
    N_HERB = 40              # начальное число травоядных
    N_PRED = 15              # начальное число хищников
    # метаболизм
    E_INIT = 100.0           # стартовая энергия
    E_MAX = 200.0
    E_PLANT = 25.0           # энергия, даваемая растением
    E_HERB = 60.0            # энергия, даваемая травоядным
    MOVE_COST = 1.0          # стоимость шага
    MOVE_COST_HERB = 1.0
    MOVE_COST_PRED = 1.5
    IDLE_COST = 0.3          # стоимость стояния
    E_REPRO = 150.0          # порог размножения
    REPRO_COST = 0.5         # доля энергии, отдаваемая потомку
    N_SENSORS = 7            # число входов перцептрона
    N_ACTIONS = 4            # число выходов (4 направления)

cfg = Config()

# ============================================================
# 2. Контроллер: однослойный перцептрон
# ============================================================
class PerceptronController:
    """
    Однослойный перцептрон: N_SENSORS входов -> N_ACTIONS выходов.
    Выходы — вероятности действий через softmax.
    """
    def __init__(self, W=None):
        if W is None:
            # Xavier-инициализация
            self.W = np.random.normal(0, 0.5, (cfg.N_SENSORS, cfg.N_ACTIONS))
        else:
            self.W = W.copy()

    def act(self, sensors):
        """sensors: (N_SENSORS,) -> индекс действия."""
        logits = self.W.T @ sensors
        # softmax + выбор по вероятности (стохастическое поведение)
        z = logits - logits.max()
        p = np.exp(z) / np.exp(z).sum()
        return int(np.random.choice(cfg.N_ACTIONS, p=p))

    def mutate(self, sigma=0.1, p=0.2):
        """Гауссова мутация весов (для ГА)."""
        W = self.W.copy()
        mask = np.random.random(W.shape) < p
        W[mask] += np.random.normal(0, sigma, mask.sum())
        return PerceptronController(W)

# ============================================================
# 3. Агенты: растение, травоядное, хищник
# ============================================================
class Plant:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.alive = True

class Herbivore:
    def __init__(self, x, y, controller=None, energy=None):
        self.x, self.y = x, y
        self.energy = energy if energy is not None else cfg.E_INIT
        self.controller = controller if controller is not None else PerceptronController()
        self.alive = True
        self.age = 0
        self.children = 0

    def sensors(self, env):
        """Сенсоры: пища/опасность/стены впереди, слева, справа + энергия."""
        def danger_at(x, y):
            if not (0 <= x < cfg.GRID and 0 <= y < cfg.GRID): return 1.0
            for p in env.predators:
                if p.alive and (p.x, p.y) == (x, y): return 1.0
            return 0.0

        def food_ahead(x, y):
            if not (0 <= x < cfg.GRID and 0 <= y < cfg.GRID): return 0.0
            for pl in env.plants:
                if pl.alive and (pl.x, pl.y) == (x, y): return 1.0
            return 0.0

        dx, dy = [0, 1, -1, 0], [-1, 0, 0, 1]
        # направление берём "вперёд" по текущему желанию — упрощённо
        # используем 4 направления как 3 сенсора: впереди(+), слева(-1), справа(+1)
        # а также общий уровень энергии
        # простая модель: сенсоры по 4 направлениям
        s = np.array([
            food_ahead(self.x, self.y - 1),      # сверху
            food_ahead(self.x + 1, self.y),      # справа
            food_ahead(self.x, self.y + 1),      # снизу
            food_ahead(self.x - 1, self.y),      # слева
            danger_at(self.x, self.y - 1) + danger_at(self.x + 1, self.y) +
            danger_at(self.x, self.y + 1) + danger_at(self.x - 1, self.y),
            self.energy / cfg.E_MAX,             # нормированная энергия
            1.0,                                 # bias
        ])
        return s

    def step(self, env):
        """Один шаг: перцептрон выбирает действие, движемся, едим, тратим энергию."""
        self.age += 1
        s = self.sensors(env)
        action = self.controller.act(s)  # 0..3 — направления
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        nx, ny = self.x + dx, self.y + dy
        # границы
        if not (0 <= nx < cfg.GRID and 0 <= ny < cfg.GRID):
            nx, ny = self.x, self.y
            self.energy -= cfg.MOVE_COST
        else:
            self.energy -= cfg.MOVE_COST_HERB
        self.x, self.y = nx, ny
        # поедание растения
        for pl in env.plants:
            if pl.alive and (pl.x, pl.y) == (self.x, self.y):
                pl.alive = False
                self.energy = min(cfg.E_MAX, self.energy + cfg.E_PLANT)
                break
        # смерть / размножение
        if self.energy <= 0:
            self.alive = False
        elif self.energy >= cfg.E_REPRO and np.random.rand() < 0.3:
            self.reproduce(env)

    def reproduce(self, env):
        child_energy = self.energy * cfg.REPRO_COST
        self.energy -= child_energy
        # потомок с мутированным контроллером
        child_ctrl = self.controller.mutate(sigma=0.15, p=0.3)
        # ищем свободную соседнюю клетку
        for _ in range(10):
            nx = self.x + np.random.randint(-1, 2)
            ny = self.y + np.random.randint(-1, 2)
            if 0 <= nx < cfg.GRID and 0 <= ny < cfg.GRID:
                env.herbivores.append(Herbivore(nx, ny, child_ctrl, child_energy))
                self.children += 1
                return

class Predator:
    def __init__(self, x, y, controller=None, energy=None):
        self.x, self.y = x, y
        self.energy = energy if energy is not None else cfg.E_INIT
        self.controller = controller if controller is not None else PerceptronController()
        self.alive = True
        self.age = 0
        self.children = 0

    def sensors(self, env):
        def prey_at(x, y):
            if not (0 <= x < cfg.GRID and 0 <= y < cfg.GRID): return 1.0  # стена — как препятствие
            for h in env.herbivores:
                if h.alive and (h.x, h.y) == (x, y): return 1.0
            return 0.0

        s = np.array([
            prey_at(self.x, self.y - 1),
            prey_at(self.x + 1, self.y),
            prey_at(self.x, self.y + 1),
            prey_at(self.x - 1, self.y),
            0.0,  # хищники не боятся
            self.energy / cfg.E_MAX,
            1.0,
        ])
        return s

    def step(self, env):
        self.age += 1
        s = self.sensors(env)
        action = self.controller.act(s)
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        nx, ny = self.x + dx, self.y + dy
        if not (0 <= nx < cfg.GRID and 0 <= ny < cfg.GRID):
            nx, ny = self.x, self.y
            self.energy -= cfg.MOVE_COST_PRED
        else:
            self.energy -= cfg.MOVE_COST_PRED
        self.x, self.y = nx, ny
        # охота
        for h in env.herbivores:
            if h.alive and (h.x, h.y) == (self.x, self.y):
                h.alive = False
                self.energy = min(cfg.E_MAX, self.energy + cfg.E_HERB)
                break
        if self.energy <= 0:
            self.alive = False
        elif self.energy >= cfg.E_REPRO and np.random.rand() < 0.2:
            self.reproduce(env)

    def reproduce(self, env):
        child_energy = self.energy * cfg.REPRO_COST
        self.energy -= child_energy
        child_ctrl = self.controller.mutate(sigma=0.15, p=0.3)
        for _ in range(10):
            nx = self.x + np.random.randint(-1, 2)
            ny = self.y + np.random.randint(-1, 2)
            if 0 <= nx < cfg.GRID and 0 <= ny < cfg.GRID:
                env.predators.append(Predator(nx, ny, child_ctrl, child_energy))
                self.children += 1
                return

# ============================================================
# 4. Среда (биоценоз)
# ============================================================
class Biocenosis:
    def __init__(self):
        self.plants = []
        self.herbivores = []
        self.predators = []
        self.tick = 0
        self.history = {'plants': [], 'herbivores': [], 'predators': [],
                        'avg_herb_energy': [], 'avg_pred_energy': []}
        self._init_world()

    def _init_world(self):
        # растения
        for _ in range(cfg.N_PLANTS):
            self.plants.append(Plant(np.random.randint(0, cfg.GRID),
                                     np.random.randint(0, cfg.GRID)))
        # травоядные
        for _ in range(cfg.N_HERB):
            self.herbivores.append(Herbivore(np.random.randint(0, cfg.GRID),
                                              np.random.randint(0, cfg.GRID)))
        # хищники
        for _ in range(cfg.N_PRED):
            self.predators.append(Predator(np.random.randint(0, cfg.GRID),
                                            np.random.randint(0, cfg.GRID)))

    def grow_plants(self):
        """Рост растений с вероятностью PLANT_GROW_RATE."""
        n_alive = sum(1 for p in self.plants if p.alive)
        if n_alive < cfg.N_PLANTS:
            n_to_add = int((cfg.N_PLANTS - n_alive) * cfg.PLANT_GROW_RATE) + 1
            for _ in range(n_to_add):
                self.plants.append(Plant(np.random.randint(0, cfg.GRID),
                                         np.random.randint(0, cfg.GRID)))

    def step(self):
        self.tick += 1
        self.grow_plants()
        # шаги агентов (в случайном порядке)
        np.random.shuffle(self.herbivores)
        for h in self.herbivores:
            if h.alive: h.step(self)
        np.random.shuffle(self.predators)
        for p in self.predators:
            if p.alive: p.step(self)
        # очистка мёртвых
        self.plants = [p for p in self.plants if p.alive]
        self.herbivores = [h for h in self.herbivores if h.alive]
        self.predators = [p for p in self.predators if p.alive]
        # статистика
        self.history['plants'].append(len(self.plants))
        self.history['herbivores'].append(len(self.herbivores))
        self.history['predators'].append(len(self.predators))
        self.history['avg_herb_energy'].append(
            np.mean([h.energy for h in self.herbivores]) if self.herbivores else 0)
        self.history['avg_pred_energy'].append(
            np.mean([p.energy for p in self.predators]) if self.predators else 0)

# ============================================================
# 5. Быстрый прогон без GUI (для проверки/статистики)
# ============================================================
def run_simulation(n_steps=500, seed=0):
    np.random.seed(seed)
    world = Biocenosis()
    for _ in range(n_steps):
        world.step()
        if len(world.herbivores) == 0 and len(world.predators) == 0:
            break
    return world

print("="*60)
print("БЫСТРЫЙ ПРОГОН (500 шагов)")
print("="*60)
w = run_simulation(500, seed=1)
print(f"  Растений:    {len(w.plants)}")
print(f"  Травоядных:  {len(w.herbivores)}")
print(f"  Хищников:    {len(w.predators)}")
print(f"  Тиков:       {w.tick}")
if w.herbivores:
    print(f"  Ср. энергия травоядных: {np.mean([h.energy for h in w.herbivores]):.1f}")
if w.predators:
    print(f"  Ср. энергия хищников:   {np.mean([p.energy for p in w.predators]):.1f}")

# График динамики популяций
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ax = axes[0]
ax.plot(w.history['plants'],     label='Растения',   color='green')
ax.plot(w.history['herbivores'], label='Травоядные', color='blue')
ax.plot(w.history['predators'],  label='Хищники',    color='red')
ax.set_xlabel("Тик"); ax.set_ylabel("Численность")
ax.set_title("Динамика популяций"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(w.history['avg_herb_energy'], label='Ср. энергия травоядных', color='blue')
ax.plot(w.history['avg_pred_energy'], label='Ср. энергия хищников',    color='red')
ax.set_xlabel("Тик"); ax.set_ylabel("Энергия")
ax.set_title("Средняя энергия агентов"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("biocenosis_stats.png", dpi=110)
plt.show()

# ============================================================
# 6. GUI: анимация + кнопки + слайдеры
# ============================================================
class BiocenosisGUI:
    def __init__(self):
        self.world = Biocenosis()
        self.running = True
        self.speed = 2  # тиков на кадр

        self.fig = plt.figure(figsize=(13, 9))
        # --- поле ---
        self.ax_field = self.fig.add_axes([0.05, 0.25, 0.55, 0.70])
        self.ax_field.set_xlim(0, cfg.GRID); self.ax_field.set_ylim(0, cfg.GRID)
        self.ax_field.set_xticks([]); self.ax_field.set_yticks([])
        self.ax_field.set_facecolor('#1a1a2e')
        self.ax_field.set_title("Биоценоз: растения (зелёные) • травоядные (синие) • хищники (красные)")

        # артисты
        self.sc_plants, = self.ax_field.plot([], [], 's', color='#2ecc71', ms=4, alpha=0.8)
        self.sc_herb,   = self.ax_field.plot([], [], 'o', color='#3498db', ms=7)
        self.sc_pred,   = self.ax_field.plot([], [], '^', color='#e74c3c', ms=9)

        # --- график динамики ---
        self.ax_plot = self.fig.add_axes([0.65, 0.55, 0.32, 0.40])
        self.ax_plot.set_title("Динамика популяций")
        self.ax_plot.set_xlabel("Тик"); self.ax_plot.set_ylabel("Численность")
        self.line_p, = self.ax_plot.plot([], [], color='green', label='Растения')
        self.line_h, = self.ax_plot.plot([], [], color='blue',  label='Травоядные')
        self.line_d, = self.ax_plot.plot([], [], color='red',   label='Хищники')
        self.ax_plot.legend(fontsize=8); self.ax_plot.grid(alpha=0.3)

        # --- информация ---
        self.ax_info = self.fig.add_axes([0.65, 0.25, 0.32, 0.22])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.02, 0.95, "", va='top', fontsize=11,
                                            family='monospace')

        # --- кнопки ---
        ax_btn_pause = self.fig.add_axes([0.05, 0.12, 0.10, 0.05])
        ax_btn_reset = self.fig.add_axes([0.17, 0.12, 0.10, 0.05])
        self.btn_pause = Button(ax_btn_pause, 'Пауза')
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_reset = Button(ax_btn_reset, 'Сброс')
        self.btn_reset.on_clicked(self.reset)

        # --- слайдеры ---
        ax_sl_speed = self.fig.add_axes([0.05, 0.05, 0.25, 0.03])
        self.sl_speed = Slider(ax_sl_speed, 'Скорость', 1, 10, valinit=2, valstep=1)
        self.sl_speed.on_changed(self.set_speed)

        ax_sl_grow = self.fig.add_axes([0.40, 0.05, 0.25, 0.03])
        self.sl_grow = Slider(ax_sl_grow, 'Рост растений', 0.01, 0.5, valinit=cfg.PLANT_GROW_RATE)
        self.sl_grow.on_changed(self.set_grow)

        self.anim = animation.FuncAnimation(self.fig, self.update,
                                             interval=100, blit=False, cache_frame_data=False)

    def toggle_pause(self, _event):
        self.running = not self.running
        self.btn_pause.label.set_text('Продолжить' if not self.running else 'Пауза')

    def reset(self, _event):
        self.world = Biocenosis()

    def set_speed(self, val):
        self.speed = int(val)

    def set_grow(self, val):
        cfg.PLANT_GROW_RATE = float(val)

    def update(self, _frame):
        if not self.running:
            return
        for _ in range(self.speed):
            self.world.step()
        w = self.world
        # поле
        if w.plants:
            self.sc_plants.set_data([p.x for p in w.plants], [p.y for p in w.plants])
        else:
            self.sc_plants.set_data([], [])
        if w.herbivores:
            self.sc_herb.set_data([h.x for h in w.herbivores], [h.y for h in w.herbivores])
        else:
            self.sc_herb.set_data([], [])
        if w.predators:
            self.sc_pred.set_data([p.x for p in w.predators], [p.y for p in w.predators])
        else:
            self.sc_pred.set_data([], [])
        # график
        self.line_p.set_data(range(len(w.history['plants'])), w.history['plants'])
        self.line_h.set_data(range(len(w.history['herbivores'])), w.history['herbivores'])
        self.line_d.set_data(range(len(w.history['predators'])), w.history['predators'])
        self.ax_plot.relim(); self.ax_plot.autoscale_view()
        # информация
        info = (f"Тик: {w.tick}\n"
                f"Растений:   {len(w.plants):4d}\n"
                f"Травоядных: {len(w.herbivores):4d}\n"
                f"Хищников:   {len(w.predators):4d}\n")
        if w.herbivores:
            info += f"Ср. E(трава): {np.mean([h.energy for h in w.herbivores]):.1f}\n"
        if w.predators:
            info += f"Ср. E(хищн):  {np.mean([p.energy for p in w.predators]):.1f}\n"
        self.info_text.set_text(info)

if __name__ == "__main__":
    gui = BiocenosisGUI()
    plt.show()
