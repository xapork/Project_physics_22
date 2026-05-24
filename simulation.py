"""
Численное моделирование прецессии и нутации осесимметричного волчка
с точечной опорой в поле силы тяжести при наличии трения.

Волчок описывается тремя углами Эйлера: θ (нутация), φ (прецессия), ψ (вращение).
Система уравнений приведена к безразмерному виду и решается методом RK4.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # режим без графического окна — нужен для сохранения в файл
import matplotlib.pyplot as plt
import os

# Создаём папку images/
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)



# ФИЗИЧЕСКАЯ МОДЕЛЬ
#
# Используются углы Эйлера:
#   θ — полярный (нутационный) угол: насколько ось наклонена от вертикали
#   φ — азимутальный (прецессионный) угол: куда смотрит ось по горизонту
#   ψ — угол собственного вращения волчка вокруг своей оси
#
# Все величины переведены в безразмерный вид через масштаб времени:
#   t* = sqrt(A / (m·g·l))
#
# Два главных безразмерных параметра:
#   α = C/A — отношение моментов инерции (насколько волчок "гироскопичен")
#   C_fr — коэффициент трения (0 = трения нет, >0 = движение затухает)
#
# Система из трёх безразмерных уравнений:
#
#   θ'' = sinθ·cosθ·(φ')² − α·n₃·sinθ·φ' − sinθ − C_fr·θ'
#         (центробежный эффект) (гироскоп)  (тяжесть) (трение)
#
#   φ'' = −2·(cosθ/sinθ)·θ'·φ' + α·n₃·θ'/sinθ − C_fr·φ'
#          (эффект Кориолиса)    (гироскоп)      (трение)
#
#   n₃' = −C_fr·n₃
#          собственное вращение затухает экспоненциально
#
# n₃ — безразмерная скорость собственного вращения


def rhs(t, state, alpha, C_fr):
    """
    Правая часть системы ОДУ — вычисляет ускорения и производные.

    Принимает текущее состояние системы:
        state = [θ, θ', φ, φ', n₃]
    Возвращает производные:
        [θ', θ'', φ', φ'', n₃']

    Именно эта функция кодирует физику волчка —
    здесь записаны все три уравнения движения.
    """
    theta, dtheta, phi, dphi, n3 = state

    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # Защита от деления на ноль: sin(0) = 0 при θ=0 или θ=π
    # (вертикальное положение — вырожденный случай)
    if abs(sin_th) < 1e-10:
        sin_th = 1e-10

    # Уравнение нутации (θ'')
    # Четыре слагаемых:
    # 1) sin·cos·(φ')²  — центробежный эффект от прецессии (пытается изменить θ)
    # 2) −α·n₃·sin·φ'  — гироскопический момент (держит ось)
    # 3) −sin           — момент силы тяжести (пытается уронить волчок)
    # 4) −C_fr·θ'       — торможение трением по θ
    ddtheta = (sin_th * cos_th * dphi**2
               - alpha * n3 * sin_th * dphi
               - sin_th
               - C_fr * dtheta)

    # Уравнение прецессии (φ'')
    # Три слагаемых:
    # 1) −2·(cos/sin)·θ'·φ'  — эффект Кориолиса
    # 2) α·n₃·θ'/sin         — гироскопический момент
    # 3) −C_fr·φ'             — торможение трением по φ
    ddphi = (-2.0 * (cos_th / sin_th) * dtheta * dphi
             + alpha * n3 / sin_th * dtheta
             - C_fr * dphi)

    # Затухание собственного вращения (n₃')
    # Трение тормозит вращение пропорционально текущей скорости
    # Решение: n₃(τ) = n₃₀ · exp(−C_fr · τ) — экспоненциальное затухание
    dn3 = -C_fr * n3

    return np.array([dtheta, ddtheta, dphi, ddphi, dn3])


def rk4_step(t, state, dt, alpha, C_fr):
    """
    Один шаг метода Рунге-Кутты 4-го порядка.

    Идея метода: вместо одного грубого шага делаем четыре пробных
    вычисления наклона и берём их взвешенное среднее.
    Это даёт точность O(h⁵) против O(h²) у метода Эйлера.

    k1 — наклон в начале шага
    k2 — наклон в середине, если идти по k1
    k3 — наклон в середине, если идти по k2 (уточнение)
    k4 — наклон в конце, если идти по k3

    Итог: y_new = y + h/6 · (k1 + 2k2 + 2k3 + k4)
    Средние наклоны (k2, k3) входят с весом 2 — они важнее крайних.
    """
    k1 = rhs(t,          state,           alpha, C_fr)
    k2 = rhs(t + dt/2,   state + dt/2*k1, alpha, C_fr)
    k3 = rhs(t + dt/2,   state + dt/2*k2, alpha, C_fr)
    k4 = rhs(t + dt,     state + dt*k3,   alpha, C_fr)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def integrate(theta0, dtheta0, phi0, dphi0, n3_0,
              alpha, C_fr, t_end, dt=0.005):
    """
    Интегрирование системы от τ=0 до τ=t_end методом RK4.

    Начальные условия:
        theta0  — начальный угол наклона оси (рад)
        dtheta0 — начальная скорость нутации
        phi0    — начальный азимутальный угол
        dphi0   — начальная скорость прецессии
        n3_0    — начальная скорость собственного вращения

    Параметры модели:
        alpha   — отношение моментов инерции C/A
        C_fr    — коэффициент трения
        t_end   — конец интегрирования
        dt      — шаг (должен быть меньше 2π/(10·α·n₃) для точности)

    Возвращает массивы: время, θ(τ), φ(τ)
    """
    # Собираем начальный вектор состояния
    state = np.array([theta0, dtheta0, phi0, dphi0, n3_0], dtype=float)

    n_steps = int(t_end / dt)

    # Заранее выделяем память под результаты (быстрее чем append)
    t_arr     = np.zeros(n_steps + 1)
    theta_arr = np.zeros(n_steps + 1)
    phi_arr   = np.zeros(n_steps + 1)

    # Записываем начальное состояние
    t_arr[0]     = 0.0
    theta_arr[0] = state[0]
    phi_arr[0]   = state[2]

    # Основной цикл интегрирования — шаг за шагом по времени
    t = 0.0
    for i in range(n_steps):
        state = rk4_step(t, state, dt, alpha, C_fr)
        t += dt
        t_arr[i+1]     = t
        theta_arr[i+1] = state[0]   # сохраняем θ
        phi_arr[i+1]   = state[2]   # сохраняем φ

    return t_arr, theta_arr, phi_arr



# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ


def save_fig(fig, filename):
    "Сохраняет график в папку images/ и закрывает окно."
    path = os.path.join(IMAGES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сохранено: {path}")



# БАЗОВЫЕ ПАРАМЕТРЫ
# Используются во всех сериях если не указано иное

THETA0  = np.radians(30.0)  # начальный угол наклона = 30 градусов
DTHETA0 = 0.0               # начальная скорость нутации = 0 (ось не кивает)
PHI0    = 0.0               # начальный азимутальный угол = 0
DPHI0   = 0.05              # начальная скорость прецессии
N3_0    = 10.0              # начальная скорость собственного вращения
ALPHA   = 2.0               # α = C/A = 2 (волчок умеренно "гироскопичен")
C_FR    = 0.0               # трения нет по умолчанию
T_END   = 80.0              # моделируем до τ = 80
DT      = 0.005             # шаг интегрирования (с запасом < 0.031)


#
# СЕРИЯ 1: Влияние коэффициента трения C_fr
# Все остальные параметры базовые.

#
print("Серия 1: влияние коэффициента трения")

C_fr_values = [0.0, 0.02, 0.05, 0.10]
colors_cfr  = ["blue", "green", "orange", "red"]

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for C_fr_val, col in zip(C_fr_values, colors_cfr):
    t, th, ph = integrate(THETA0, DTHETA0, PHI0, DPHI0, N3_0,
                          ALPHA, C_fr_val, T_END, DT)
    # Переводим радианы в градусы для наглядности
    axes[0].plot(t, np.degrees(th), color=col, linewidth=1.0,
                 label=r"$C_{fr}=" + f"{C_fr_val}$")
    # % 360 — чтобы φ не уходил за 360°, а начинал с нуля снова
    axes[1].plot(t, np.degrees(ph) % 360, color=col, linewidth=1.0,
                 label=r"$C_{fr}=" + f"{C_fr_val}$")

axes[0].set_ylabel(r"$\theta$, градусы")
axes[0].set_title(r"Влияние коэффициента трения $C_{fr}$ на движение волчка")
axes[0].legend(fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].set_ylabel(r"$\varphi$, градусы")
axes[1].set_xlabel(r"Безразмерное время $\tau$")
axes[1].legend(fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
save_fig(fig, "fig_series1_friction.pdf")



# СЕРИЯ 2: Влияние параметра α = C/A
# Трения нет (C_fr=0). Смотрим только первые 5 единиц времени —
# иначе колебания сливаются в сплошную полосу из-за высокой частоты нутации.

print("Серия 2: влияние параметра α")

alpha_values = [1.0, 2.0, 4.0, 8.0]

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for alp, col in zip(alpha_values, ["blue", "green", "orange", "red"]):
    # t_end=5.0 — зум на начало, чтобы видеть отдельные колебания
    t, th, ph = integrate(THETA0, DTHETA0, PHI0, DPHI0, N3_0,
                          alp, 0.0, 5.0, DT)
    axes[0].plot(t, np.degrees(th), color=col, linewidth=1.0,
                 label=r"$\alpha=" + f"{alp}$")
    axes[1].plot(t, np.degrees(ph) % 360, color=col, linewidth=1.0,
                 label=r"$\alpha=" + f"{alp}$")

axes[0].set_ylabel(r"$\theta$, градусы")
axes[0].set_title(r"Влияние $\alpha = C/A$ — первые 5 единиц времени (зум)")
axes[0].legend(fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].set_ylabel(r"$\varphi$, градусы")
axes[1].set_xlabel(r"Безразмерное время $\tau$")
axes[1].legend(fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
save_fig(fig, "fig_series2_zoom.pdf")


# СЕРИЯ 3: Влияние начальной скорости прецессии φ'(0)
# Трения нет. Зум на первые 5 единиц времени.
print("Серия 3: влияние начальной скорости прецессии")

dphi0_values = [0.01, 0.05, 0.15, 0.30]
colors_dp    = ["blue", "green", "orange", "red"]

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for dp, col in zip(dphi0_values, colors_dp):
    t, th, ph = integrate(THETA0, DTHETA0, PHI0, dp, N3_0,
                          ALPHA, 0.0, 5.0, DT)
    axes[0].plot(t, np.degrees(th), color=col, linewidth=1.0,
                 label=r"$\dot\varphi_0=" + f"{dp}$")
    axes[1].plot(t, np.degrees(ph) % 360, color=col, linewidth=1.0,
                 label=r"$\dot\varphi_0=" + f"{dp}$")

axes[0].set_ylabel(r"$\theta$, градусы")
axes[0].set_title(r"Влияние $\dot\varphi_0$ — первые 5 единиц времени (зум)")
axes[0].legend(fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].set_ylabel(r"$\varphi$, градусы")
axes[1].set_xlabel(r"Безразмерное время $\tau$")
axes[1].legend(fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
save_fig(fig, "fig_series3_zoom.pdf")


# СЕРИЯ 4: Влияние начального полярного угла θ₀
# Трения нет. Полное время τ=80.
print("Серия 4: влияние начального полярного угла")

theta0_values = [10, 30, 60, 80]
colors_th0    = ["blue", "green", "orange", "red"]

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for th0_deg, col in zip(theta0_values, colors_th0):
    th0 = np.radians(th0_deg)
    t, th, ph = integrate(th0, DTHETA0, PHI0, DPHI0, N3_0,
                          ALPHA, 0.0, T_END, DT)
    axes[0].plot(t, np.degrees(th), color=col, linewidth=1.0,
                 label=r"$\theta_0=" + f"{th0_deg}°$")
    axes[1].plot(t, np.degrees(ph) % 360, color=col, linewidth=1.0,
                 label=r"$\theta_0=" + f"{th0_deg}°$")

axes[0].set_ylabel(r"$\theta$, градусы")
axes[0].set_title(r"Влияние начального полярного угла $\theta_0$")
axes[0].legend(fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].set_ylabel(r"$\varphi$, градусы")
axes[1].set_xlabel(r"Безразмерное время $\tau$")
axes[1].legend(fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
save_fig(fig, "fig_series4_theta0.pdf")



# СЕРИЯ 5: Сравнение четырёх режимов трения в одном окне
# Для C_fr=0 показываем зум (10 единиц) — иначе нутация сливается.
# Для остальных полное время τ=80, чтобы видеть затухание.
print("Серия 5: сравнение четырёх режимов трения")

configs = [
    (0.00, r"Без трения ($C_{fr}=0$)"),
    (0.03, r"Малое трение ($C_{fr}=0.03$)"),
    (0.07, r"Среднее трение ($C_{fr}=0.07$)"),
    (0.15, r"Большое трение ($C_{fr}=0.15$)"),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

for col_i, (c_fr, title) in enumerate(configs):
    t_plot = 10.0 if c_fr == 0.0 else T_END  # зум только для нулевого трения
    t, th, ph = integrate(THETA0, DTHETA0, PHI0, DPHI0, N3_0,
                          ALPHA, c_fr, t_plot, DT)

    axes[0, col_i].plot(t, np.degrees(th), linewidth=1.0, color="steelblue")
    axes[0, col_i].set_title(title, fontsize=8)
    axes[0, col_i].set_ylabel(r"$\theta$, °")
    axes[0, col_i].grid(True, linestyle="--", alpha=0.5)

    axes[1, col_i].plot(t, np.degrees(ph) % 360, linewidth=1.0, color="darkorange")
    axes[1, col_i].set_ylabel(r"$\varphi$, °")
    axes[1, col_i].set_xlabel(r"$\tau$")
    axes[1, col_i].grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Сравнение режимов движения при различных значениях трения", fontsize=11)
fig.tight_layout()
save_fig(fig, "fig_series5_comparison.pdf")


# СЕРИЯ 6: Фазовые портреты
# Сравниваем два случая: без трения и с трением C_fr=0.05
#
# Левый график (θ, θ'): показывает тип движения
#   — замкнутая кривая = периодическое движение (без трения)
#   — спираль к центру = затухающее движение (с трением)
#
# Правый график (θ, φ): показывает траекторию оси в пространстве
#   — вертикальная полоса = ось держится на одном угле (без трения)
#   — петли смещаются влево = ось выпрямляется (с трением)

print("Серия 6: фазовые портреты")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for c_fr, col, lbl in [(0.0,  "blue", r"$C_{fr}=0$"),
                        (0.05, "red",  r"$C_{fr}=0.05$")]:

    # Здесь нужен полный вектор состояния (включая θ'),
    # поэтому интегрируем вручную а не через функцию integrate()
    state = np.array([THETA0, DTHETA0, PHI0, DPHI0, N3_0])
    n_steps = int(T_END / DT)

    # Массивы для хранения пар (θ, φ) и (θ, θ')
    th_ph  = np.zeros((n_steps+1, 2))  # для правого графика
    th_dth = np.zeros((n_steps+1, 2))  # для левого графика

    th_ph[0]  = [state[0], state[2]]   # начальные θ и φ
    th_dth[0] = [state[0], state[1]]   # начальные θ и θ'

    t = 0.0
    for i in range(n_steps):
        state = rk4_step(t, state, DT, ALPHA, c_fr)
        t += DT
        th_ph[i+1]  = [state[0], state[2]]  # θ и φ
        th_dth[i+1] = [state[0], state[1]]  # θ и θ'

    # Левый график: фазовый портрет (θ, θ')
    axes[0].plot(np.degrees(th_dth[:, 0]), th_dth[:, 1],
                 color=col, linewidth=0.7, label=lbl, alpha=0.8)

    # Правый график: траектория (θ, φ)
    axes[1].plot(np.degrees(th_ph[:, 0]), np.degrees(th_ph[:, 1]) % 360,
                 color=col, linewidth=0.7, label=lbl, alpha=0.8, marker=",")

axes[0].set_xlabel(r"$\theta$, °")
axes[0].set_ylabel(r"$\dot\theta$")
axes[0].set_title(r"Фазовый портрет $(\theta,\,\dot\theta)$")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].set_xlabel(r"$\theta$, °")
axes[1].set_ylabel(r"$\varphi$, °")
axes[1].set_title(r"Траектория в плоскости $(\theta,\,\varphi)$")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
save_fig(fig, "fig_series6_phase.pdf")

print("\nВсе графики сохранены в папку images/")
