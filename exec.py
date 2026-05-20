#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Суточная симуляция расчёта режимов ГПА (24 часа)
v3.0 (динамический профиль Pin, интеграция сетевой характеристики компрессора, унифицированный интерфейс)
"""
import os
import numpy as np
from functional import objective_function
from gpa_model import GPA
from compressor_model import Compressor
from RegBase import NstRegulatorBase
from Nst_SHOPreg import PIDRegulator
from Nst_PSOreg import PyswarmsRegulator
from Nst_GWOreg import GreyWolfRegulator
from Nst_ACOreg import AntColonyRegulator
from pressure_imit import generate_pin_profile  # Импорт генератора профиля давления
import csv  # Для работы с CSV-файлами
import matplotlib
matplotlib.use('Agg')  # Использовать неинтерактивный backend для графиков
import matplotlib.pyplot as plt

# === КОНФИГУРАЦИЯ ===
REGULATION_MODE = "ACO"  # Варианты: "PID" | "PSO" | "GWO" | "ACO"
Q_TARGET_DAY = 2400.0    # тыс.м³/сутки
SIMULATION_HOURS = 24
Tin_ext = 22.0           # Температура на входе (постоянная для данного сценария)

# Генерация суточного профиля входного давления
# Режим "daily_cycle": пик потребления днём, спад ночью.
# База 5.8 МПа, амплитуда ±0.6 МПа, небольшой измерительный шум.
pin_profile = generate_pin_profile(
    hours=SIMULATION_HOURS,
    mode="daily_cycle",
    base_pin=5.8,
    amplitude=0.6,
    noise_std=0.04,
    seed=42  # Фиксированный seed для воспроизводимости результатов
)

# Инициализация оборудования
gpa_instances = [
    GPA(ID="1", Q_min_lim=210.0, Q_max_lim=800.0, Nst_min_lim=4200.0, Nst_max_lim=9900.0),
    GPA(ID="2", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8500.0),
    GPA(ID="3", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8200.0)
]
comp_instances = [
    Compressor(Nst_nom=6000.0, Q_nom=600.0),
    Compressor(Nst_nom=5900.0, Q_nom=500.0),
    Compressor(Nst_nom=5700.0, Q_nom=500.0)
]

# Структура выбора стратегии
regulators = {
    "PID": PIDRegulator(Kp=50.0, Ki=0.3, Kd=1.5, nst_step_max=800.0),
    "PSO": PyswarmsRegulator(n_particles=30, iters=40),
    "GWO": GreyWolfRegulator(n_wolves=30, iters=40, max_nst_step=800.0),
    "ACO": AntColonyRegulator(n_ants=30, iters=40, alpha=1.0, beta=2.0, rho=0.1, max_nst_step=800.0),
    # "LLM_GPT4": LLMRegulator(model="gpt-4", iterations=5)
}

if REGULATION_MODE not in regulators:
    raise ValueError(f"Неподдерживаемый режим: {REGULATION_MODE}. Доступные: {list(regulators.keys())}")

# === ВЫВОД ПАРАМЕТРОВ АЛГОРИТМОВ ===
print(f"\n{'='*100}")
print("ПАРАМЕТРЫ АЛГОРИТМОВ ОПТИМИЗАЦИИ")
print(f"{'='*100}\n")

# PID параметры
pid_reg = regulators["PID"]
print("PID РЕГУЛЯТОР:")
print(f"  Kp (пропорциональный коэффициент): {pid_reg.Kp}")
print(f"  Ki (интегральный коэффициент): {pid_reg.Ki}")
print(f"  Kd (дифференциальный коэффициент): {pid_reg.Kd}")
print(f"  nst_step_max (макс. изменение оборотов): {pid_reg.nst_step_max}")
print()

# PSO параметры
pso_reg = regulators["PSO"]
print("PSO (PARTICLE SWARM OPTIMIZATION):")
print(f"  n_particles (кол-во частиц): {pso_reg.n_particles}")
print(f"  iters (кол-во итераций): {pso_reg.iters}")
print(f"  c1 (познавательный параметр): {pso_reg.options['c1']}")
print(f"  c2 (социальный параметр): {pso_reg.options['c2']}")
print(f"  w (инерционный вес): {pso_reg.options['w']}")
print(f"  delta_nst_max (макс. шаг изменения): {pso_reg.delta_nst_max}")
print()

# GWO параметры
gwo_reg = regulators["GWO"]
print("GWO (GREY WOLF OPTIMIZER):")
print(f"  n_wolves (кол-во волков): {gwo_reg.n_wolves}")
print(f"  iters (кол-во итераций): {gwo_reg.iters}")
print(f"  max_nst_step (макс. изменение оборотов): {gwo_reg.max_nst_step}")
print()

# ACO параметры
aco_reg = regulators["ACO"]
print("ACO (ANT COLONY OPTIMIZATION):")
print(f"  n_ants (кол-во муравьёв): {aco_reg.n_ants}")
print(f"  iters (кол-во итераций): {aco_reg.iters}")
print(f"  alpha (важность феромона): {aco_reg.alpha}")
print(f"  beta (важность привлекательности): {aco_reg.beta}")
print(f"  rho (скорость испарения феромона): {aco_reg.rho}")
print(f"  q (константа для феромона): {aco_reg.q}")
print(f"  max_nst_step (макс. изменение оборотов): {aco_reg.max_nst_step}")
print()

print(f"{'='*100}\n")

regulator = regulators[REGULATION_MODE]
prev_nst = np.array([4500.0, 4600.0, 4400.0]) 
q_cumulative_actual = 0.0
total_objective_value = 0.0
hourly_results = []

# === ИНИЦИАЛИЗАЦИЯ ФАЙЛА ДЛЯ ЭКСПОРТА ДАННЫХ ===
os.makedirs("results", exist_ok=True)  # Создание папки results при её отсутствии
export_filename = os.path.join("results", f"simulation_results_{REGULATION_MODE}_{Q_TARGET_DAY:.0f}.csv")
export_file = open(export_filename, 'w', newline='', encoding='utf-8-sig')  # utf-8-sig для корректного отображения кириллицы в Excel
writer = None

print(f"{'='*120}")
print(f"СИМУЛЯЦИЯ | Режим: {REGULATION_MODE} | Цель за сутки: {Q_TARGET_DAY:.1f} тыс.м³")
print(f"{'='*120}")

for hour in range(SIMULATION_HOURS):
    pin_current = pin_profile[hour]
    q_prev_hour = hourly_results[-1]["q_fact"] if hour > 0 else 0.0
    
    # === ДИНАМИЧЕСКИЙ РАСЧЁТ ЦЕЛИ ===
    # Остаток, который необходимо прокачать до конца суток
    q_remaining = max(0.0, Q_TARGET_DAY - q_cumulative_actual)
    hours_remaining = SIMULATION_HOURS - hour
    
    # Целевой расход на текущий час
    target_for_hour = q_remaining / hours_remaining if hours_remaining > 0 else 0.0
 
    P_out_req = pin_current * 1.25

    # Унифицированный вызов регулятора с динамической уставкой
    nst_values = regulator.calculate(
        gpa_list=gpa_instances,
        comp_list=comp_instances,
        Pin=pin_current,
        Tin=Tin_ext,
        Q_target=target_for_hour,  # <-- Передаётся остаточная цель
        prev_nst=prev_nst,
        P_out_req=P_out_req
    )
    nst_values = np.asarray(nst_values).flatten()
    prev_nst = nst_values.copy()

    # Расчёт физических параметров агрегатов (без изменений)
    Qprod_arr, Qg_arr, Nst_arr, Eps_arr = [], [], [], []
    for gpa, comp, nst in zip(gpa_instances, comp_instances, nst_values):
        comp.Nst_current = nst
        if hasattr(comp, 'set_network_conditions'):
            comp.set_network_conditions(pin_current, P_out_req)
            comp_out = comp.get_state_with_network(nst, pin_current, P_out_req)
        else:
            comp_out = comp.get_state()
            
        gpa.Q_min_lim = 210.0 * 60.0 / 1000.0 * 1.05
        gpa.update_state(comp_out, Pin=pin_current, Tin=Tin_ext)
        gpa.calc_production_flow()
        gpa.calc_compression_power()
        gpa.calc_available_power(TinD=Tin_ext, Patm=101.325)
        gpa.calc_fuel_flow(TinD=Tin_ext)
        
        Qprod_arr.append(gpa.Qprod)
        Qg_arr.append(gpa.Qtg)
        Nst_arr.append(gpa.Nst)
        Eps_arr.append(gpa.Eps)

    Q_current_hour = sum(Qprod_arr)
    q_cumulative_actual += Q_current_hour
    deviation = q_cumulative_actual - Q_TARGET_DAY

    pid_out_str = f"{regulator.pid_correction:+7.1f}" if hasattr(regulator, 'pid_correction') else "   N/A"
    print(f"ЧАС {hour:02d} | Pin={pin_current:4.2f} МПа | Цель_часа={target_for_hour:5.1f} | "
          f"Nst: {nst_values.astype(int)} | Расход: {Q_current_hour:7.1f} | "
          f"Накоплено: {q_cumulative_actual:7.1f} | Ошибка: {deviation:+7.1f}")

# Расчёт функционала качества (почасовой)
    res = objective_function(
        gpa_list=[g.ID for g in gpa_instances],
        Q_target=target_for_hour,
        E_target=1.5,
        Qprod_i=np.array(Qprod_arr), Qg_i=np.array(Qg_arr),
        Nst_i=np.array(Nst_arr), Eps_i=np.array(Eps_arr),
        Q_min_lim_i=np.array([g.Q_min_lim for g in gpa_instances]),
        Q_max_lim_i=np.array([g.Q_max_lim for g in gpa_instances]),
        Nst_min_lim_i=np.array([g.Nst_min_lim for g in gpa_instances]),
        Nst_max_lim_i=np.array([g.Nst_max_lim for g in gpa_instances])
    )
    total_objective_value += res["total"]

    # === НОВЫЙ БЛОК: Детализация функционала ===
    print(f"  [ДЕТАЛИЗАЦИЯ ЧАСА {hour:02d}]")
    print(f"    - Ошибка регулирования (q_term): {res['q_term']:.2f}")
    print(f"    - Затраты на топливо (f_term):   {res['f_term']:.2f}")
    print(f"    - Неравномерность (b_term):      {res['b_term']:.2f}")
    print(f"    - Штрафы ограничений:            {res['constraint_term']:.2f}")
    print(f"    - ИТОГО (total):                 {res['total']:.2f}")
    print("-" * 40)
    # ==========================================

     # === ЭКСПОРТ ДАННЫХ В ФАЙЛ ===
    # Формирование заголовка при первой итерации
    if writer is None:
        header = [
            'Hour', 'Pin_MPa', 'Tin_C', 'Target_hour', 'Q_fact', 'Deviation',
            'Nst_GPA1', 'Nst_GPA2', 'Nst_GPA3',
            'Npol_GPA1', 'Npol_GPA2', 'Npol_GPA3',
            'Qprod_GPA1', 'Qprod_GPA2', 'Qprod_GPA3', 'Qprod_total',
            'q_term', 'f_term', 'b_term', 'constraint_term', 'total_cost',
            'Fuel_total_kg_h'
        ]
        writer = csv.writer(export_file, delimiter=';')
        writer.writerow(header)
    
    # Формирование строки данных
    row = [
        hour,
        f"{pin_current:.3f}",
        f"{Tin_ext:.1f}",
        f"{target_for_hour:.2f}",
        f"{Q_current_hour:.2f}",
        f"{deviation:+.2f}",
        # Обороты
        f"{nst_values[0]:.0f}", f"{nst_values[1]:.0f}", f"{nst_values[2]:.0f}",
        # КПД (политропный) - ИСПРАВЛЕННАЯ ВЕРСИЯ
       f"{Eps_arr[0]:.3f}", f"{Eps_arr[1]:.3f}", f"{Eps_arr[2]:.3f}",  # Исправлено
        # Коммерческий расход по агрегатам
        f"{Qprod_arr[0]:.2f}", f"{Qprod_arr[1]:.2f}", f"{Qprod_arr[2]:.2f}",
        f"{Q_current_hour:.2f}",
        # Компоненты функционала
        f"{res['q_term']:.2f}", f"{res['f_term']:.2f}", f"{res['b_term']:.2f}",
        f"{res['constraint_term']:.2f}", f"{res['total']:.2f}",
        f"{res['fuel_total']:.2f}"
    ]
    writer.writerow(row)
    # ==========================================

    hourly_results.append({
        "hour": hour, "total": res["total"], "q_fact": Q_current_hour,
        "fuel_total": res.get("fuel_total", 0.0), "constraint_term": res.get("constraint_term", 0.0)
    })

# Закрытие файла экспорта
if export_file and not export_file.closed:
    export_file.close()
    print(f"Данные экспортированы в файл: {os.path.abspath(export_filename)}")

print(f"{'='*120}")
print(f"ИТОГ: Функционал = {total_objective_value:.4f} | Накоплено: {q_cumulative_actual:.1f} | "
      f"Цель: {Q_TARGET_DAY:.1f} | Ошибка: {deviation:+.1f}")
print(f"{'='*120}")

# Детальный вывод параметров агрегатов
print(f"{'ID':<4} | {'Nst':>6} | {'Q':>7} | {'Qprod':>7} | {'Eps':>5} | {'SR':>5} | "
      f"{'Tout':>6} | {'Npol':>5} | {'Nef':>8} | {'Ner':>8} | Ограничения")
print("-" * 145)
for gpa in gpa_instances:
    status = gpa.check_constraints()
    violations = [key for key, val in status.items() if val and key != 'is_within_limits']
    status_str = ", ".join(violations) if violations else "В ПРЕДЕЛАХ"
    print(f"{gpa.ID:<4} | {gpa.Nst:>6.0f} | {gpa.Q:>7.1f} | {gpa.Qprod:>7.2f} | {gpa.Eps:>5.3f} | "
          f"{gpa.SR:>5.3f} | {gpa.Tout:>6.1f} | {gpa.Npol:>5.3f} | {gpa.Nef:>8.1f} | "
          f"{gpa.Ner:>8.1f} | {status_str}")

# === ФУНКЦИЯ ДЛЯ ВИЗУАЛИЗАЦИИ ТАБЛИЦЫ ФИНАЛЬНЫХ ПАРАМЕТРОВ ===
def plot_final_parameters_table(gpa_list, regulation_mode, q_cumulative, q_target, total_cost):
    """
    Отрисовка таблицы с финальными параметрами ГПА после оптимизации.
    
    Args:
        gpa_list: список объектов ГПА
        regulation_mode: строка режима регулирования
        q_cumulative: накопленный расход за сутки
        q_target: целевой расход
        total_cost: общее значение функционала
    """
    # Подготовка данных для таблицы
    table_data = []
    for gpa in gpa_list:
        status = gpa.check_constraints()
        violations = [key for key, val in status.items() if val and key != 'is_within_limits']
        status_str = "✓ OK" if not violations else "✗ VIOLATION"
        
        row = [
            gpa.ID,                           # ID ГПА
            f"{gpa.Nst:.0f}",                # Обороты
            f"{gpa.Q:.1f}",                  # Давление
            f"{gpa.Qprod:.2f}",              # Коммерческий расход
            f"{gpa.Eps:.3f}",                # КПД
            f"{gpa.SR:.3f}",                 # Степень сжатия
            f"{gpa.Tout:.1f}",               # Температура на выходе
            f"{gpa.Npol:.3f}",               # Политропный КПД
            f"{gpa.Nef:.1f}",                # Эффективная мощность
            f"{gpa.Ner:.1f}",                # Требуемая мощность
            status_str
        ]
        table_data.append(row)
    
    # Заголовки столбцов
    columns = ['ГПА', 'Nst\n(об/мин)', 'P_in\n(МПа)', 'Qпроизв\n(тыс.м³/ч)', 
               'Eps\n(КПД)', 'SR\n(сжатие)', 'Tout\n(°C)', 'Npol\n(пол.КПД)', 
               'Nef\n(кВт)', 'Ner\n(кВт)', 'Статус']
    
    # Создание фигуры и осей
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Создание таблицы
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                     colWidths=[0.08, 0.12, 0.12, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    
    # Стилизация таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Форматирование заголовков
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Чередование цветов строк
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    # Добавление общей информации
    summary_text = f"""
    АЛГОРИТМ: {regulation_mode}  |  НАКОПЛЕНО: {q_cumulative:.1f} тыс.м³  |  ЦЕЛЬ: {q_target:.1f} тыс.м³  |  ФУНКЦИОНАЛ: {total_cost:.4f}
    """
    
    fig.text(0.5, 0.95, summary_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), weight='bold')
    
    plt.title(f'Финальные параметры agрегатов ГПА (режим: {regulation_mode})', 
              fontsize=14, weight='bold', pad=30)
    
    # Сохранение и отображение
    table_filename = os.path.join("results", f"final_parameters_{regulation_mode}_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(table_filename, dpi=150, bbox_inches='tight')
    print(f"Таблица параметров сохранена: {os.path.abspath(table_filename)}")
    
    plt.show()

# Вызов функции визуализации таблицы для всех режимов (включая GWO)
plot_final_parameters_table(gpa_instances, REGULATION_MODE, q_cumulative_actual, 
                              Q_TARGET_DAY, total_objective_value)

# === ВИЗУАЛИЗАЦИЯ СХОДИМОСТИ ДЛЯ PSO ===
if REGULATION_MODE == "PSO" and hasattr(regulator, 'optimizer') and regulator.optimizer is not None:
    # График истории стоимости (convergence plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    cost_history = regulator.optimizer.cost_history
    ax.plot(cost_history, 'g-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Итерация', fontsize=12, weight='bold')
    ax.set_ylabel('Значение функционала', fontsize=12, weight='bold')
    ax.set_title('Сходимость алгоритма PSO (Particle Swarm Optimization)', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(len(cost_history)), cost_history, alpha=0.3, color='green')
    
    pso_convergence_filename = os.path.join("results", f"pso_convergence_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(pso_convergence_filename, dpi=150, bbox_inches='tight')
    print(f"График сходимости PSO сохранен: {os.path.abspath(pso_convergence_filename)}")
    plt.show()
    
    # Траектория частиц в 2D (опционально)
    if hasattr(regulator, 'plot_contour_2d'):
        try:
            regulator.plot_contour_2d(0, 1)
        except Exception as e:
            print(f"Примечание: 2D визуализация PSO не доступна ({e})")

# === ВИЗУАЛИЗАЦИЯ СХОДИМОСТИ ДЛЯ GWO ===
if REGULATION_MODE == "GWO" and hasattr(regulator, 'cost_history'):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(regulator.cost_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Итерация', fontsize=12, weight='bold')
    ax.set_ylabel('Значение функционала', fontsize=12, weight='bold')
    ax.set_title('Сходимость алгоритма GWO (серые волки)', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    convergence_filename = os.path.join("results", f"gwo_convergence_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(convergence_filename, dpi=150, bbox_inches='tight')
    print(f"График сходимости GWO сохранен: {os.path.abspath(convergence_filename)}")
    
    plt.show()

# === ВИЗУАЛИЗАЦИЯ СХОДИМОСТИ ДЛЯ ACO ===
if REGULATION_MODE == "ACO" and hasattr(regulator, 'cost_history'):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(regulator.cost_history, 'r-', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Итерация', fontsize=12, weight='bold')
    ax.set_ylabel('Значение функционала', fontsize=12, weight='bold')
    ax.set_title('Сходимость алгоритма ACO (муравьиная колония)', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(len(regulator.cost_history)), regulator.cost_history, alpha=0.2, color='red')
    
    convergence_filename = os.path.join("results", f"aco_convergence_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(convergence_filename, dpi=150, bbox_inches='tight')
    print(f"График сходимости ACO сохранен: {os.path.abspath(convergence_filename)}")
    
    plt.show()

# === ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ФЕРОМОНОВ ДЛЯ ACO ===
if REGULATION_MODE == "ACO" and hasattr(regulator, 'pheromone_history') and len(regulator.pheromone_history) > 0:
    pheromone_history = regulator.pheromone_history
    
    # Агрегируем феромоны по итерациям и сегментам
    n_iterations = len(pheromone_history)
    n_dims = pheromone_history[0].shape[0] if len(pheromone_history[0].shape) > 1 else pheromone_history[0].size
    
    # Вычисляем среднее значение феромонов для каждой итерации
    pheromone_means = []
    for iteration_phero in pheromone_history:
        if len(iteration_phero.shape) > 1:
            mean_val = np.mean(iteration_phero)
        else:
            mean_val = np.mean(iteration_phero)
        pheromone_means.append(mean_val)
    
    # Создаем визуализацию тепловой карты
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Преобразуем историю в матрицу для heatmap
    if len(pheromone_history[0].shape) > 1:
        pheromone_matrix = np.array([np.mean(p, axis=1) for p in pheromone_history]).T
    else:
        pheromone_matrix = np.array(pheromone_means).reshape(1, -1)
    
    im = ax.imshow(pheromone_matrix, aspect='auto', cmap='hot', interpolation='bilinear')
    ax.set_xlabel('Итерация', fontsize=12, weight='bold')
    ax.set_ylabel('Размерность', fontsize=12, weight='bold')
    ax.set_title('Распределение феромонов в алгоритме ACO', fontsize=14, weight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Интенсивность феромона', fontsize=11, weight='bold')
    
    pheromone_heatmap_filename = os.path.join("results", f"aco_pheromone_heatmap_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(pheromone_heatmap_filename, dpi=150, bbox_inches='tight')
    print(f"Тепловая карта феромонов ACO сохранена: {os.path.abspath(pheromone_heatmap_filename)}")
    
    plt.show()

# === АНАЛИЗ ВЛИЯНИЯ ФЕРОМОНОВ ВС ПРИВЛЕКАТЕЛЬНОСТИ ===
if REGULATION_MODE == "ACO" and hasattr(regulator, 'pheromone_history') and hasattr(regulator, 'attractiveness_history'):
    if len(regulator.pheromone_history) > 0 and len(regulator.attractiveness_history) > 0:
        pheromone_history = regulator.pheromone_history
        attractiveness_history = regulator.attractiveness_history
        
        n_iterations = min(len(pheromone_history), len(attractiveness_history))
        
        # Вычисляем средние значения феромона и привлекательности
        pheromone_influence = []
        attractiveness_influence = []
        
        for i in range(n_iterations):
            phero = pheromone_history[i]
            attr = attractiveness_history[i]
            
            # Вычисляем средние значения
            phero_mean = np.mean(phero) if phero.size > 0 else 0
            attr_mean = np.mean(attr) if attr.size > 0 else 0
            
            pheromone_influence.append(phero_mean)
            attractiveness_influence.append(attr_mean)
        
        # Нормализуем для сравнения
        phero_max = max(pheromone_influence) if max(pheromone_influence) > 0 else 1
        attr_max = max(attractiveness_influence) if max(attractiveness_influence) > 0 else 1
        
        pheromone_normalized = [p / phero_max for p in pheromone_influence]
        attractiveness_normalized = [a / attr_max for a in attractiveness_influence]
        
        # Создаем график сравнения влияния
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # График 1: Абсолютные значения
        iterations = range(n_iterations)
        ax1.plot(iterations, pheromone_influence, 'b-', linewidth=2.5, marker='o', markersize=5, label='Феромоны (τ)')
        ax1.plot(iterations, attractiveness_influence, 'g-', linewidth=2.5, marker='s', markersize=5, label='Привлекательность (η)')
        ax1.set_xlabel('Итерация', fontsize=12, weight='bold')
        ax1.set_ylabel('Среднее значение', fontsize=12, weight='bold')
        ax1.set_title('Абсолютные значения: Феромоны vs Привлекательность', fontsize=13, weight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Нормализованные значения (стacked area)
        ax2.fill_between(iterations, 0, pheromone_normalized, alpha=0.6, label='Влияние феромонов (τ)', color='blue')
        ax2.fill_between(iterations, pheromone_normalized, 
                        [p + a for p, a in zip(pheromone_normalized, attractiveness_normalized)],
                        alpha=0.6, label='Влияние привлекательности (η)', color='green')
        ax2.set_xlabel('Итерация', fontsize=12, weight='bold')
        ax2.set_ylabel('Нормализованное влияние', fontsize=12, weight='bold')
        ax2.set_title('Относительное влияние компонентов алгоритма ACO', fontsize=13, weight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 2])
        
        plt.tight_layout()
        
        influence_filename = os.path.join("results", f"aco_influence_analysis_{Q_TARGET_DAY:.0f}.png")
        plt.savefig(influence_filename, dpi=150, bbox_inches='tight')
        print(f"Анализ влияния компонентов ACO сохранен: {os.path.abspath(influence_filename)}")
        
        plt.show()

# === ФУНКЦИЯ ДЛЯ СРАВНЕНИЯ АЛГОРИТМОВ PSO И GWO ===
def run_simulation_and_collect_results(regulation_mode, gpa_instances, comp_instances, pin_profile, 
                                        Q_TARGET_DAY, SIMULATION_HOURS, Tin_ext):
    """
    Запускает симуляцию для заданного режима регулирования и собирает финальные результаты.
    
    Returns:
        dict: {
            'gpa_ids': список ID,
            'nst_final': финальные обороты для каждого ГПА,
            'q_final': финальный расход для каждого ГПА,
            'total_q': суммарный расход,
            'total_cost': общая стоимость
        }
    """
    import copy
    
    # Создание копий объектов для независимой симуляции
    gpa_list = [copy.deepcopy(gpa) for gpa in gpa_instances]
    comp_list = [copy.deepcopy(comp) for comp in comp_instances]
    
    regulators_local = {
        "PID": PIDRegulator(Kp=50.0, Ki=0.3, Kd=1.5, nst_step_max=800.0),
        "PSO": PyswarmsRegulator(n_particles=30, iters=40),
        "GWO": GreyWolfRegulator(n_wolves=30, iters=40, max_nst_step=800.0),
        "ACO": AntColonyRegulator(n_ants=30, iters=40, alpha=1.0, beta=2.0, rho=0.1, max_nst_step=800.0),
    }
    
    regulator = regulators_local[regulation_mode]
    prev_nst = np.array([4500.0, 4600.0, 4400.0])
    q_cumulative_actual = 0.0
    total_objective_value = 0.0
    hourly_results = []
    
    print(f"\n{'='*80}")
    print(f"ЗАПУСК СИМУЛЯЦИИ: {regulation_mode.upper()}")
    print(f"{'='*80}")
    
    for hour in range(SIMULATION_HOURS):
        pin_current = pin_profile[hour]
        q_prev_hour = hourly_results[-1]["q_fact"] if hour > 0 else 0.0
        
        q_remaining = max(0.0, Q_TARGET_DAY - q_cumulative_actual)
        hours_remaining = SIMULATION_HOURS - hour
        target_for_hour = q_remaining / hours_remaining if hours_remaining > 0 else 0.0
        P_out_req = pin_current * 1.25

        nst_values = regulator.calculate(
            gpa_list=gpa_list,
            comp_list=comp_list,
            Pin=pin_current,
            Tin=Tin_ext,
            Q_target=target_for_hour,
            prev_nst=prev_nst,
            P_out_req=P_out_req
        )
        nst_values = np.asarray(nst_values).flatten()
        prev_nst = nst_values.copy()

        Qprod_arr, Qg_arr, Nst_arr, Eps_arr = [], [], [], []
        for gpa, comp, nst in zip(gpa_list, comp_list, nst_values):
            comp.Nst_current = nst
            if hasattr(comp, 'set_network_conditions'):
                comp.set_network_conditions(pin_current, P_out_req)
                comp_out = comp.get_state_with_network(nst, pin_current, P_out_req)
            else:
                comp_out = comp.get_state()
                
            gpa.Q_min_lim = 210.0 * 60.0 / 1000.0 * 1.05
            gpa.update_state(comp_out, Pin=pin_current, Tin=Tin_ext)
            gpa.calc_production_flow()
            gpa.calc_compression_power()
            gpa.calc_available_power(TinD=Tin_ext, Patm=101.325)
            gpa.calc_fuel_flow(TinD=Tin_ext)
            
            Qprod_arr.append(gpa.Qprod)
            Qg_arr.append(gpa.Qtg)
            Nst_arr.append(gpa.Nst)
            Eps_arr.append(gpa.Eps)

        Q_current_hour = sum(Qprod_arr)
        q_cumulative_actual += Q_current_hour
        
        res = objective_function(
            gpa_list=[g.ID for g in gpa_list],
            Q_target=target_for_hour,
            E_target=1.5,
            Qprod_i=np.array(Qprod_arr), Qg_i=np.array(Qg_arr),
            Nst_i=np.array(Nst_arr), Eps_i=np.array(Eps_arr),
            Q_min_lim_i=np.array([g.Q_min_lim for g in gpa_list]),
            Q_max_lim_i=np.array([g.Q_max_lim for g in gpa_list]),
            Nst_min_lim_i=np.array([g.Nst_min_lim for g in gpa_list]),
            Nst_max_lim_i=np.array([g.Nst_max_lim for g in gpa_list])
        )
        total_objective_value += res["total"]
        
        hourly_results.append({
            "hour": hour, "total": res["total"], "q_fact": Q_current_hour,
            "fuel_total": res.get("fuel_total", 0.0), "constraint_term": res.get("constraint_term", 0.0)
        })
    
    print(f"ИТОГ {regulation_mode}: Функционал = {total_objective_value:.4f} | "
          f"Накоплено: {q_cumulative_actual:.1f} тыс.м³")
    
    return {
        'gpa_ids': [g.ID for g in gpa_list],
        'nst_final': np.array([g.Nst for g in gpa_list]),
        'q_final': np.array([g.Qprod for g in gpa_list]),
        'total_q': q_cumulative_actual,
        'total_cost': total_objective_value,
        'gpa_objects': gpa_list  # Сохраняем объекты для деталей
    }

def plot_pso_gwo_comparison(pso_results, gwo_results, Q_TARGET_DAY):
    """
    Создаёт сравнительные графики и таблицы PSO vs GWO по Nst и Q для всех трёх ГПА.
    """
    gpa_ids = pso_results['gpa_ids']
    
    # === ТАБЛИЦА СРАВНЕНИЯ Nst ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Таблица Nst (обороты)
    nst_comparison_data = []
    for i, gpa_id in enumerate(gpa_ids):
        pso_nst = pso_results['nst_final'][i]
        gwo_nst = gwo_results['nst_final'][i]
        diff = gwo_nst - pso_nst
        diff_pct = (diff / pso_nst * 100) if pso_nst != 0 else 0
        
        nst_comparison_data.append([
            f"GPA-{gpa_id}",
            f"{pso_nst:.0f}",
            f"{gwo_nst:.0f}",
            f"{diff:+.0f}",
            f"{diff_pct:+.1f}%"
        ])
    
    ax1 = axes[0]
    ax1.axis('tight')
    ax1.axis('off')
    
    nst_table = ax1.table(
        cellText=nst_comparison_data,
        colLabels=['ГПА', 'PSO Nst', 'GWO Nst', 'Разница', 'Δ %'],
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2]
    )
    nst_table.auto_set_font_size(False)
    nst_table.set_fontsize(11)
    nst_table.scale(1, 2.5)
    
    # Стилизация заголовков
    for i in range(5):
        nst_table[(0, i)].set_facecolor('#4472C4')
        nst_table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Чередование цветов строк
    for i in range(1, len(nst_comparison_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                nst_table[(i, j)].set_facecolor('#E7E6E6')
            else:
                nst_table[(i, j)].set_facecolor('#F2F2F2')
    
    ax1.set_title('Сравнение Nst (об/мин)', fontsize=13, weight='bold', pad=20)
    
    # === ТАБЛИЦА СРАВНЕНИЯ Q ===
    q_comparison_data = []
    for i, gpa_id in enumerate(gpa_ids):
        pso_q = pso_results['q_final'][i]
        gwo_q = gwo_results['q_final'][i]
        diff = gwo_q - pso_q
        diff_pct = (diff / pso_q * 100) if pso_q != 0 else 0
        
        q_comparison_data.append([
            f"GPA-{gpa_id}",
            f"{pso_q:.2f}",
            f"{gwo_q:.2f}",
            f"{diff:+.2f}",
            f"{diff_pct:+.1f}%"
        ])
    
    ax2 = axes[1]
    ax2.axis('tight')
    ax2.axis('off')
    
    q_table = ax2.table(
        cellText=q_comparison_data,
        colLabels=['ГПА', 'PSO Q', 'GWO Q', 'Разница', 'Δ %'],
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2]
    )
    q_table.auto_set_font_size(False)
    q_table.set_fontsize(11)
    q_table.scale(1, 2.5)
    
    # Стилизация заголовков
    for i in range(5):
        q_table[(0, i)].set_facecolor('#70AD47')
        q_table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Чередование цветов строк
    for i in range(1, len(q_comparison_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                q_table[(i, j)].set_facecolor('#E7E6E6')
            else:
                q_table[(i, j)].set_facecolor('#F2F2F2')
    
    ax2.set_title('Сравнение Q (тыс.м³/ч)', fontsize=13, weight='bold', pad=20)
    
    # Общий заголовок
    fig.suptitle(f'Сравнение PSO vs GWO | Цель: {Q_TARGET_DAY:.0f} тыс.м³/сутки', 
                 fontsize=15, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    comparison_filename = os.path.join("results", f"comparison_PSO_vs_GWO_tables_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
    print(f"Таблица сравнения сохранена: {os.path.abspath(comparison_filename)}")
    plt.show()
    
    # === ГРАФИКИ СРАВНЕНИЯ ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # График Nst
    x_pos = np.arange(len(gpa_ids))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, pso_results['nst_final'], width, label='PSO', alpha=0.8, color='#4472C4')
    axes[0].bar(x_pos + width/2, gwo_results['nst_final'], width, label='GWO', alpha=0.8, color='#70AD47')
    axes[0].set_xlabel('ГПА', fontsize=12, weight='bold')
    axes[0].set_ylabel('Nst (об/мин)', fontsize=12, weight='bold')
    axes[0].set_title('Сравнение оборотов Nst', fontsize=13, weight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f'GPA-{gid}' for gid in gpa_ids])
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # График Q
    axes[1].bar(x_pos - width/2, pso_results['q_final'], width, label='PSO', alpha=0.8, color='#4472C4')
    axes[1].bar(x_pos + width/2, gwo_results['q_final'], width, label='GWO', alpha=0.8, color='#70AD47')
    axes[1].set_xlabel('ГПА', fontsize=12, weight='bold')
    axes[1].set_ylabel('Q (тыс.м³/ч)', fontsize=12, weight='bold')
    axes[1].set_title('Сравнение расходов Q', fontsize=13, weight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'GPA-{gid}' for gid in gpa_ids])
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    fig.suptitle(f'Гистограммы сравнения PSO vs GWO', fontsize=14, weight='bold')
    plt.tight_layout()
    
    histogram_filename = os.path.join("results", f"comparison_PSO_vs_GWO_histograms_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(histogram_filename, dpi=150, bbox_inches='tight')
    print(f"Гистограммы сравнения сохранены: {os.path.abspath(histogram_filename)}")
    plt.show()
    
    # === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В CSV ===
    csv_filename = os.path.join("results", f"comparison_PSO_vs_GWO_{Q_TARGET_DAY:.0f}.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer_csv = __import__('csv').writer(csvfile, delimiter=';')
        
        writer_csv.writerow(['СРАВНЕНИЕ АЛГОРИТМОВ PSO И GWO'])
        writer_csv.writerow([f'Целевой расход: {Q_TARGET_DAY:.1f} тыс.м³/сутки'])
        writer_csv.writerow([])
        
        # Сравнение Nst
        writer_csv.writerow(['СРАВНЕНИЕ NST (об/мин)'])
        writer_csv.writerow(['ГПА', 'PSO Nst', 'GWO Nst', 'Разница', 'Δ %'])
        for i, gpa_id in enumerate(gpa_ids):
            pso_nst = pso_results['nst_final'][i]
            gwo_nst = gwo_results['nst_final'][i]
            diff = gwo_nst - pso_nst
            diff_pct = (diff / pso_nst * 100) if pso_nst != 0 else 0
            writer_csv.writerow([f'GPA-{gpa_id}', f'{pso_nst:.0f}', f'{gwo_nst:.0f}', f'{diff:+.0f}', f'{diff_pct:+.1f}%'])
        
        writer_csv.writerow([])
        
        # Сравнение Q
        writer_csv.writerow(['СРАВНЕНИЕ Q (тыс.м³/ч)'])
        writer_csv.writerow(['ГПА', 'PSO Q', 'GWO Q', 'Разница', 'Δ %'])
        for i, gpa_id in enumerate(gpa_ids):
            pso_q = pso_results['q_final'][i]
            gwo_q = gwo_results['q_final'][i]
            diff = gwo_q - pso_q
            diff_pct = (diff / pso_q * 100) if pso_q != 0 else 0
            writer_csv.writerow([f'GPA-{gpa_id}', f'{pso_q:.2f}', f'{gwo_q:.2f}', f'{diff:+.2f}', f'{diff_pct:+.1f}%'])
        
        writer_csv.writerow([])
        
        # Итоговая статистика
        writer_csv.writerow(['ИТОГОВАЯ СТАТИСТИКА'])
        writer_csv.writerow(['Метрика', 'PSO', 'GWO', 'Разница'])
        writer_csv.writerow(['Суммарный расход (тыс.м³)', f'{pso_results["total_q"]:.1f}', 
                            f'{gwo_results["total_q"]:.1f}', f'{gwo_results["total_q"] - pso_results["total_q"]:+.1f}'])
        writer_csv.writerow(['Функционал качества', f'{pso_results["total_cost"]:.4f}', 
                            f'{gwo_results["total_cost"]:.4f}', f'{gwo_results["total_cost"] - pso_results["total_cost"]:+.4f}'])
    
    print(f"Результаты сравнения сохранены в CSV: {os.path.abspath(csv_filename)}")

def plot_all_algorithms_comparison(pid_results, pso_results, gwo_results, aco_results, Q_TARGET_DAY):
    """
    Создаёт сравнительные гистограммы для всех 4 алгоритмов (PID, PSO, GWO, ACO) по Nst и Q.
    """
    gpa_ids = pid_results['gpa_ids']
    print(f"\n{'='*80}")
    print("СОЗДАНИЕ СРАВНИТЕЛЬНЫХ ГИСТОГРАММ ДЛЯ ВСЕХ 4 АЛГОРИТМОВ")
    print(f"{'='*80}\n")
    
    # === ГИСТОГРАММА СРАВНЕНИЯ Nst ===
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(gpa_ids))
    width = 0.2
    
    # Данные для каждого алгоритма
    pid_nst = pid_results['nst_final']
    pso_nst = pso_results['nst_final']
    gwo_nst = gwo_results['nst_final']
    aco_nst = aco_results['nst_final']
    
    # Создание столбцов
    bars1 = ax.bar(x - 1.5*width, pid_nst, width, label='PID', alpha=0.9, color='#FFC000', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x - 0.5*width, pso_nst, width, label='PSO', alpha=0.9, color='#4472C4', edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + 0.5*width, gwo_nst, width, label='GWO', alpha=0.9, color='#70AD47', edgecolor='black', linewidth=1.5)
    bars4 = ax.bar(x + 1.5*width, aco_nst, width, label='ACO', alpha=0.9, color='#C55A11', edgecolor='black', linewidth=1.5)
    
    # Добавление значений на столбцы
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=9, weight='bold')
    
    ax.set_xlabel('ГПА', fontsize=13, weight='bold')
    ax.set_ylabel('Nst (об/мин)', fontsize=13, weight='bold')
    ax.set_title('Сравнение оборотов Nst: PID vs PSO vs GWO vs ACO', fontsize=15, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'GPA-{gid}' for gid in gpa_ids], fontsize=12, weight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    nst_filename = os.path.join("results", f"comparison_all_algorithms_Nst_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(nst_filename, dpi=150, bbox_inches='tight')
    print(f">>> Гистограмма Nst для всех алгоритмов сохранена: {os.path.abspath(nst_filename)}")
    plt.close()
    
    # === ГИСТОГРАММА СРАВНЕНИЯ Q ===
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Данные для каждого алгоритма
    pid_q = pid_results['q_final']
    pso_q = pso_results['q_final']
    gwo_q = gwo_results['q_final']
    aco_q = aco_results['q_final']
    
    # Создание столбцов
    bars1 = ax.bar(x - 1.5*width, pid_q, width, label='PID', alpha=0.9, color='#FFC000', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x - 0.5*width, pso_q, width, label='PSO', alpha=0.9, color='#4472C4', edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + 0.5*width, gwo_q, width, label='GWO', alpha=0.9, color='#70AD47', edgecolor='black', linewidth=1.5)
    bars4 = ax.bar(x + 1.5*width, aco_q, width, label='ACO', alpha=0.9, color='#C55A11', edgecolor='black', linewidth=1.5)
    
    # Добавление значений на столбцы
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9, weight='bold')
    
    ax.set_xlabel('ГПА', fontsize=13, weight='bold')
    ax.set_ylabel('Q (тыс.м³/ч)', fontsize=13, weight='bold')
    ax.set_title('Сравнение расходов Q: PID vs PSO vs GWO vs ACO', fontsize=15, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'GPA-{gid}' for gid in gpa_ids], fontsize=12, weight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    q_filename = os.path.join("results", f"comparison_all_algorithms_Q_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(q_filename, dpi=150, bbox_inches='tight')
    print(f">>> Гистограмма Q для всех алгоритмов сохранена: {os.path.abspath(q_filename)}")
    plt.close()
    
    # === ТАБЛИЦА СРАВНЕНИЯ ИТОГОВЫХ МЕТРИК ===
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Подготовка данных для таблицы
    metrics_data = [
        ['Суммарный расход', f"{pid_results['total_q']:.1f}", f"{pso_results['total_q']:.1f}", 
         f"{gwo_results['total_q']:.1f}", f"{aco_results['total_q']:.1f}"],
        ['Функционал качества', f"{pid_results['total_cost']:.4f}", f"{pso_results['total_cost']:.4f}", 
         f"{gwo_results['total_cost']:.4f}", f"{aco_results['total_cost']:.4f}"],
        ['Средний Nst', f"{np.mean(pid_nst):.0f}", f"{np.mean(pso_nst):.0f}", 
         f"{np.mean(gwo_nst):.0f}", f"{np.mean(aco_nst):.0f}"],
        ['Средний Q', f"{np.mean(pid_q):.2f}", f"{np.mean(pso_q):.2f}", 
         f"{np.mean(gwo_q):.2f}", f"{np.mean(aco_q):.2f}"]
    ]
    
    table = ax.table(cellText=metrics_data,
                    colLabels=['Метрика', 'PID', 'PSO', 'GWO', 'ACO'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Стилизация заголовков
    for i in range(5):
        table[(0, i)].set_facecolor('#1F4E78')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Чередование цветов строк
    colors = ['#E7E6E6', '#F2F2F2']
    for i in range(1, len(metrics_data) + 1):
        for j in range(5):
            table[(i, j)].set_facecolor(colors[i % 2])
    
    plt.title('Итоговое сравнение всех 4 алгоритмов', fontsize=14, weight='bold', pad=20)
    
    summary_filename = os.path.join("results", f"comparison_all_algorithms_summary_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
    print(f">>> Таблица итогов всех алгоритмов сохранена: {os.path.abspath(summary_filename)}")
    plt.close()
    
    # === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В CSV ===
    csv_filename = os.path.join("results", f"comparison_all_algorithms_{Q_TARGET_DAY:.0f}.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer_csv = __import__('csv').writer(csvfile, delimiter=';')
        
        writer_csv.writerow(['СРАВНЕНИЕ ВСЕХ 4 АЛГОРИТМОВ: PID vs PSO vs GWO vs ACO'])
        writer_csv.writerow([f'Целевой расход: {Q_TARGET_DAY:.1f} тыс.м³/сутки'])
        writer_csv.writerow([])
        
        # Сравнение Nst
        writer_csv.writerow(['СРАВНЕНИЕ NST (об/мин)'])
        writer_csv.writerow(['ГПА', 'PID', 'PSO', 'GWO', 'ACO', 'Max', 'Min', 'Разброс'])
        for i, gpa_id in enumerate(gpa_ids):
            values = [pid_nst[i], pso_nst[i], gwo_nst[i], aco_nst[i]]
            max_v = max(values)
            min_v = min(values)
            span = max_v - min_v
            writer_csv.writerow([f'GPA-{gpa_id}', f'{pid_nst[i]:.0f}', f'{pso_nst[i]:.0f}', 
                                f'{gwo_nst[i]:.0f}', f'{aco_nst[i]:.0f}', f'{max_v:.0f}', f'{min_v:.0f}', f'{span:.0f}'])
        
        writer_csv.writerow([])
        
        # Сравнение Q
        writer_csv.writerow(['СРАВНЕНИЕ Q (тыс.м³/ч)'])
        writer_csv.writerow(['ГПА', 'PID', 'PSO', 'GWO', 'ACO', 'Max', 'Min', 'Разброс'])
        for i, gpa_id in enumerate(gpa_ids):
            values = [pid_q[i], pso_q[i], gwo_q[i], aco_q[i]]
            max_v = max(values)
            min_v = min(values)
            span = max_v - min_v
            writer_csv.writerow([f'GPA-{gpa_id}', f'{pid_q[i]:.2f}', f'{pso_q[i]:.2f}', 
                                f'{gwo_q[i]:.2f}', f'{aco_q[i]:.2f}', f'{max_v:.2f}', f'{min_v:.2f}', f'{span:.2f}'])
        
        writer_csv.writerow([])
        
        # Итоговая статистика
        writer_csv.writerow(['ИТОГОВАЯ СТАТИСТИКА'])
        writer_csv.writerow(['Метрика', 'PID', 'PSO', 'GWO', 'ACO'])
        writer_csv.writerow(['Суммарный расход (тыс.м³)', f'{pid_results["total_q"]:.1f}', 
                            f'{pso_results["total_q"]:.1f}', f'{gwo_results["total_q"]:.1f}', f'{aco_results["total_q"]:.1f}'])
        writer_csv.writerow(['Функционал качества', f'{pid_results["total_cost"]:.4f}', 
                            f'{pso_results["total_cost"]:.4f}', f'{gwo_results["total_cost"]:.4f}', f'{aco_results["total_cost"]:.4f}'])
        writer_csv.writerow(['Средний Nst', f'{np.mean(pid_nst):.0f}', f'{np.mean(pso_nst):.0f}', 
                            f'{np.mean(gwo_nst):.0f}', f'{np.mean(aco_nst):.0f}'])
        writer_csv.writerow(['Средний Q', f'{np.mean(pid_q):.2f}', f'{np.mean(pso_q):.2f}', 
                            f'{np.mean(gwo_q):.2f}', f'{np.mean(aco_q):.2f}'])
    
    print(f">>> CSV отчёт всех алгоритмов сохранён: {os.path.abspath(csv_filename)}\n")

# === ЗАПУСК СРАВНЕНИЯ PSO vs GWO ===
print(f"\n{'='*100}")
print("СРАВНЕНИЕ АЛГОРИТМОВ: PSO vs GWO")
print(f"{'='*100}")

# Создаём СВЕЖИЕ экземпляры для сравнения (не используем уже модифицированные объекты)
gpa_instances_for_comparison = [
    GPA(ID="1", Q_min_lim=210.0, Q_max_lim=800.0, Nst_min_lim=4200.0, Nst_max_lim=9900.0),
    GPA(ID="2", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8500.0),
    GPA(ID="3", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8200.0)
]
comp_instances_for_comparison = [
    Compressor(Nst_nom=6000.0, Q_nom=600.0),
    Compressor(Nst_nom=5900.0, Q_nom=500.0),
    Compressor(Nst_nom=5700.0, Q_nom=500.0)
]

gpa_instances_for_gwo = [
    GPA(ID="1", Q_min_lim=210.0, Q_max_lim=800.0, Nst_min_lim=4200.0, Nst_max_lim=9900.0),
    GPA(ID="2", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8500.0),
    GPA(ID="3", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8200.0)
]
comp_instances_for_gwo = [
    Compressor(Nst_nom=6000.0, Q_nom=600.0),
    Compressor(Nst_nom=5900.0, Q_nom=500.0),
    Compressor(Nst_nom=5700.0, Q_nom=500.0)
]

# Запускаем симуляции для обоих алгоритмов с СВЕЖИМИ объектами
pso_results = run_simulation_and_collect_results("PSO", gpa_instances_for_comparison, comp_instances_for_comparison, pin_profile, 
                                                  Q_TARGET_DAY, SIMULATION_HOURS, Tin_ext)
gwo_results = run_simulation_and_collect_results("GWO", gpa_instances_for_gwo, comp_instances_for_gwo, pin_profile, 
                                                  Q_TARGET_DAY, SIMULATION_HOURS, Tin_ext)

# Создаём сравнительные графики и таблицы
plot_pso_gwo_comparison(pso_results, gwo_results, Q_TARGET_DAY)

# === СОЗДАНИЕ ФИНАЛЬНЫХ ТАБЛИЦ ПАРАМЕТРОВ ДЛЯ КАЖДОГО АЛГОРИТМА ===
def create_final_parameters_table_from_results(algorithm_name, results, Q_TARGET_DAY):
    """
    Создаёт и сохраняет таблицу финальных параметров для заданного алгоритма.
    """
    gpa_list = results['gpa_objects']
    q_cumulative = results['total_q']
    total_cost = results['total_cost']
    
    # Подготовка данных для таблицы
    table_data = []
    for gpa in gpa_list:
        status = gpa.check_constraints()
        violations = [key for key, val in status.items() if val and key != 'is_within_limits']
        status_str = "✓ OK" if not violations else "✗ VIOLATION"
        
        row = [
            gpa.ID,
            f"{gpa.Nst:.0f}",
            f"{gpa.Q:.1f}",
            f"{gpa.Qprod:.2f}",
            f"{gpa.Eps:.3f}",
            f"{gpa.SR:.3f}",
            f"{gpa.Tout:.1f}",
            f"{gpa.Npol:.3f}",
            f"{gpa.Nef:.1f}",
            f"{gpa.Ner:.1f}",
            status_str
        ]
        table_data.append(row)
    
    # Заголовки столбцов
    columns = ['ГПА', 'Nst\n(об/мин)', 'P_in\n(МПа)', 'Qпроизв\n(тыс.м³/ч)', 
               'Eps\n(КПД)', 'SR\n(сжатие)', 'Tout\n(°C)', 'Npol\n(пол.КПД)', 
               'Nef\n(кВт)', 'Ner\n(кВт)', 'Статус']
    
    # Создание фигуры и осей
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Создание таблицы
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                     colWidths=[0.08, 0.12, 0.12, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    
    # Стилизация таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Форматирование заголовков - выбираем цвет в зависимости от алгоритма
    color_map = {
        'PSO': '#4472C4',  # Синий
        'GWO': '#70AD47',  # Зелёный
        'ACO': '#C55A11'   # Оранжевый
    }
    header_color = color_map.get(algorithm_name, '#4472C4')
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Чередование цветов строк
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    # Добавление общей информации
    summary_text = f"""
    АЛГОРИТМ: {algorithm_name}  |  НАКОПЛЕНО: {q_cumulative:.1f} тыс.м³  |  ЦЕЛЬ: {Q_TARGET_DAY:.1f} тыс.м³  |  ФУНКЦИОНАЛ: {total_cost:.4f}
    """
    
    fig.text(0.5, 0.95, summary_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), weight='bold')
    
    plt.title(f'Финальные параметры агрегатов ГПА (алгоритм: {algorithm_name})', 
              fontsize=14, weight='bold', pad=30)
    
    # Сохранение и отображение
    table_filename = os.path.join("results", f"final_parameters_{algorithm_name}_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(table_filename, dpi=150, bbox_inches='tight')
    print(f"\n>>> Таблица финальных параметров {algorithm_name} сохранена: {os.path.abspath(table_filename)}")
    plt.close()

# Создаём таблицы для PSO и GWO из результатов сравнения
print(f"\n{'='*100}")
print("СОЗДАНИЕ ФИНАЛЬНЫХ ТАБЛИЦ ПАРАМЕТРОВ")
print(f"{'='*100}")

create_final_parameters_table_from_results("PSO", pso_results, Q_TARGET_DAY)
create_final_parameters_table_from_results("GWO", gwo_results, Q_TARGET_DAY)

# Добавляем ACO в сравнение, если основной режим не ACO
gpa_instances_for_aco = [
    GPA(ID="1", Q_min_lim=210.0, Q_max_lim=800.0, Nst_min_lim=4200.0, Nst_max_lim=9900.0),
    GPA(ID="2", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8500.0),
    GPA(ID="3", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8200.0)
]
comp_instances_for_aco = [
    Compressor(Nst_nom=6000.0, Q_nom=600.0),
    Compressor(Nst_nom=5900.0, Q_nom=500.0),
    Compressor(Nst_nom=5700.0, Q_nom=500.0)
]
aco_results = run_simulation_and_collect_results("ACO", gpa_instances_for_aco, comp_instances_for_aco, pin_profile, 
                                                  Q_TARGET_DAY, SIMULATION_HOURS, Tin_ext)
# Создаём таблицу финальных параметров для ACO
create_final_parameters_table_from_results("ACO", aco_results, Q_TARGET_DAY)

# Собираем результаты PID для полного сравнения
gpa_instances_for_pid = [
    GPA(ID="1", Q_min_lim=210.0, Q_max_lim=800.0, Nst_min_lim=4200.0, Nst_max_lim=9900.0),
    GPA(ID="2", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8500.0),
    GPA(ID="3", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8200.0)
]
comp_instances_for_pid = [
    Compressor(Nst_nom=6000.0, Q_nom=600.0),
    Compressor(Nst_nom=5900.0, Q_nom=500.0),
    Compressor(Nst_nom=5700.0, Q_nom=500.0)
]
pid_results = run_simulation_and_collect_results("PID", gpa_instances_for_pid, comp_instances_for_pid, pin_profile, 
                                                  Q_TARGET_DAY, SIMULATION_HOURS, Tin_ext)
create_final_parameters_table_from_results("PID", pid_results, Q_TARGET_DAY)

# === СОЗДАНИЕ ТАБЛИЦЫ ГИПЕРПАРАМЕТРОВ ВСЕХ АЛГОРИТМОВ ===
def create_hyperparameters_table(regulators_dict, Q_TARGET_DAY):
    """
    Создаёт и сохраняет таблицу со всеми гиперпараметрами алгоритмов.
    """
    print(f"\n{'='*100}")
    print("СОЗДАНИЕ ТАБЛИЦЫ ГИПЕРПАРАМЕТРОВ")
    print(f"{'='*100}\n")
    
    # Подготовка данных
    table_data = []
    
    # PID
    pid_reg = regulators_dict["PID"]
    table_data.append([
        "PID",
        f"{pid_reg.Kp}",
        f"{pid_reg.Ki}",
        f"{pid_reg.Kd}",
        "-",
        "-",
        "-",
        f"{pid_reg.nst_step_max}"
    ])
    
    # PSO
    pso_reg = regulators_dict["PSO"]
    table_data.append([
        "PSO",
        f"{pso_reg.n_particles}",
        f"{pso_reg.iters}",
        f"{pso_reg.options['c1']}",
        f"{pso_reg.options['c2']}",
        f"{pso_reg.options['w']}",
        "-",
        f"{pso_reg.delta_nst_max}"
    ])
    
    # GWO
    gwo_reg = regulators_dict["GWO"]
    table_data.append([
        "GWO",
        f"{gwo_reg.n_wolves}",
        f"{gwo_reg.iters}",
        "-",
        "-",
        "-",
        "-",
        f"{gwo_reg.max_nst_step}"
    ])
    
    # ACO
    aco_reg = regulators_dict["ACO"]
    table_data.append([
        "ACO",
        f"{aco_reg.n_ants}",
        f"{aco_reg.iters}",
        f"{aco_reg.alpha}",
        f"{aco_reg.beta}",
        f"{aco_reg.rho}",
        f"{aco_reg.q}",
        f"{aco_reg.max_nst_step}"
    ])
    
    # Создание таблицы
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    columns = ['Алгоритм', 'Pop/Кол-во', 'Итерации', 'c1/Alpha', 'c2/Beta', 'w/Rho', 'q', 'Max Step']
    
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.14])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Стилизация заголовков
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#1F4E78')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Цветные строки для каждого алгоритма
    colors = ['#FFC000', '#4472C4', '#70AD47', '#C55A11']  # PID, PSO, GWO, ACO
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(colors[i-1])
            table[(i, j)].set_text_props(weight='bold', color='white', fontsize=11)
    
    plt.title('Гиперпараметры всех алгоритмов оптимизации', fontsize=15, weight='bold', pad=20)
    
    # Сохранение
    table_filename = os.path.join("results", f"hyperparameters_all_algorithms_{Q_TARGET_DAY:.0f}.png")
    plt.savefig(table_filename, dpi=150, bbox_inches='tight')
    print(f">>> Таблица гиперпараметров сохранена: {os.path.abspath(table_filename)}\n")
    plt.close()
    
    # Сохранение в CSV
    csv_filename = os.path.join("results", f"hyperparameters_all_algorithms_{Q_TARGET_DAY:.0f}.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer_csv = __import__('csv').writer(csvfile, delimiter=';')
        
        writer_csv.writerow(['ГИПЕРПАРАМЕТРЫ ВСЕХ АЛГОРИТМОВ ОПТИМИЗАЦИИ'])
        writer_csv.writerow([f'Целевой расход: {Q_TARGET_DAY:.1f} тыс.м³/сутки'])
        writer_csv.writerow([])
        
        writer_csv.writerow(columns)
        for row in table_data:
            writer_csv.writerow(row)
        
        writer_csv.writerow([])
        writer_csv.writerow(['ОПИСАНИЕ ПАРАМЕТРОВ'])
        writer_csv.writerow(['PID: Kp, Ki, Kd - коэффициенты регулятора; nst_step_max - макс. шаг'])
        writer_csv.writerow(['PSO: n_particles - кол-во частиц; iters - итерации; c1,c2 - параметры; w - инерционный вес'])
        writer_csv.writerow(['GWO: n_wolves - кол-во волков; iters - итерации'])
        writer_csv.writerow(['ACO: n_ants - кол-во муравьёв; iters - итерации; alpha,beta,rho,q - параметры феромона'])
    
    print(f">>> CSV таблица гиперпараметров сохранена: {os.path.abspath(csv_filename)}\n")

# Создание таблицы гиперпараметров
create_hyperparameters_table(regulators, Q_TARGET_DAY)

# === СОЗДАНИЕ СРАВНИТЕЛЬНЫХ ГИСТОГРАММ ДЛЯ ВСЕХ 4 АЛГОРИТМОВ ===
plot_all_algorithms_comparison(pid_results, pso_results, gwo_results, aco_results, Q_TARGET_DAY)

print(f"\n{'='*100}")
print("СРАВНЕНИЕ ЗАВЕРШЕНО")
print(f"{'='*100}")

def tune_aco_hyperparameters(gpa_list, comp_list, Q_target):
    """
    Оптимизация гиперпараметров ACO методом сеточного перебора.
    Тестирует различные комбинации параметров и находит лучшую.
    
    Args:
        gpa_list: список объектов ГПА
        comp_list: список объектов компрессоров
        Q_target: целевой расход
    
    Returns:
        dict: лучшие найденные параметры и их стоимость
    """
    best_overall_cost = float('inf')
    best_params = {}
    results_list = []
    
    # Пространства для поиска
    alpha_space = [0.5, 1.0, 2.0]           # Важность феромона
    beta_space = [1.0, 2.0, 3.0]            # Важность привлекательности
    rho_space = [0.05, 0.1, 0.2, 0.3]       # Скорость испарения феромона
    n_ants_space = [20, 30, 40]             # Количество муравьёв
    
    print(f"\n{'='*100}")
    print("=== ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ ACO ===")
    print(f"{'='*100}")
    print(f"Всего комбинаций для тестирования: {len(alpha_space) * len(beta_space) * len(rho_space) * len(n_ants_space)}")
    print(f"{'='*100}\n")
    
    test_num = 0
    total_tests = len(alpha_space) * len(beta_space) * len(rho_space) * len(n_ants_space)
    
    for n_ants in n_ants_space:
        for alpha in alpha_space:
            for beta in beta_space:
                for rho in rho_space:
                    test_num += 1
                    print(f"[{test_num}/{total_tests}] Тестирование: n_ants={n_ants}, alpha={alpha}, beta={beta}, rho={rho}")
                    
                    try:
                        # Создаём регулятор с текущими параметрами
                        reg = AntColonyRegulator(
                            n_ants=n_ants, 
                            iters=40, 
                            alpha=alpha, 
                            beta=beta, 
                            rho=rho, 
                            q=1.0, 
                            max_nst_step=800.0
                        )
                        
                        # Запускаем оптимизацию на первом часу
                        reg.calculate(
                            gpa_list=gpa_list, 
                            comp_list=comp_list, 
                            Pin=pin_profile[0], 
                            Tin=Tin_ext, 
                            Q_target=Q_target,
                            prev_nst=None,
                            P_out_req=pin_profile[0] * 1.25
                        )
                        
                        # Получаем стоимость
                        current_cost = reg.best_cost if hasattr(reg, 'best_cost') else float('inf')
                        
                        # Сохраняем результат
                        result_entry = {
                            'n_ants': n_ants,
                            'alpha': alpha,
                            'beta': beta,
                            'rho': rho,
                            'cost': current_cost
                        }
                        results_list.append(result_entry)
                        
                        print(f"    ✓ Стоимость: {current_cost:.4f}")
                        
                        # Обновляем лучший результат
                        if current_cost < best_overall_cost:
                            best_overall_cost = current_cost
                            best_params = {
                                'n_ants': n_ants,
                                'alpha': alpha,
                                'beta': beta,
                                'rho': rho
                            }
                            print(f"    >>> НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ! Стоимость: {best_overall_cost:.4f}\n")
                    
                    except Exception as e:
                        print(f"    ✗ Ошибка: {str(e)}\n")
                        continue
    
    # === ВЫВОД РЕЗУЛЬТАТОВ ===
    print(f"\n{'='*100}")
    print("=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ACO ===")
    print(f"{'='*100}")
    print(f"\n✓ ЛУЧШИЕ ПАРАМЕТРЫ ACO:")
    print(f"  n_ants (кол-во муравьёв): {best_params['n_ants']}")
    print(f"  alpha (важность феромона): {best_params['alpha']}")
    print(f"  beta (важность привлекательности): {best_params['beta']}")
    print(f"  rho (скорость испарения): {best_params['rho']}")
    print(f"  Лучшая стоимость: {best_overall_cost:.4f}")
    print(f"{'='*100}\n")
    
    # Сортируем результаты и выводим ТОП-5
    results_list_sorted = sorted(results_list, key=lambda x: x['cost'])
    print("ТОП-5 ЛУЧШИХ КОМБИНАЦИЙ:\n")
    for i, result in enumerate(results_list_sorted[:5], 1):
        print(f"{i}. n_ants={result['n_ants']}, alpha={result['alpha']}, beta={result['beta']}, "
              f"rho={result['rho']} → Стоимость: {result['cost']:.4f}")
    
    print(f"\n{'='*100}\n")
    
    # Сохранение результатов в CSV
    csv_filename = os.path.join("results", f"aco_hyperparameter_optimization_{Q_TARGET_DAY:.0f}.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer_csv = __import__('csv').writer(csvfile, delimiter=';')
        
        writer_csv.writerow(['ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ ACO'])
        writer_csv.writerow([f'Целевой расход: {Q_TARGET_DAY:.1f} тыс.м³/сутки'])
        writer_csv.writerow([])
        
        writer_csv.writerow(['n_ants', 'alpha', 'beta', 'rho', 'Стоимость'])
        for result in results_list_sorted:
            writer_csv.writerow([
                result['n_ants'],
                result['alpha'],
                result['beta'],
                result['rho'],
                f"{result['cost']:.6f}"
            ])
        
        writer_csv.writerow([])
        writer_csv.writerow(['ЛУЧШИЕ ПАРАМЕТРЫ'])
        writer_csv.writerow(['Параметр', 'Значение'])
        writer_csv.writerow(['n_ants', best_params['n_ants']])
        writer_csv.writerow(['alpha', best_params['alpha']])
        writer_csv.writerow(['beta', best_params['beta']])
        writer_csv.writerow(['rho', best_params['rho']])
        writer_csv.writerow(['Лучшая стоимость', f"{best_overall_cost:.6f}"])
    
    print(f"Результаты оптимизации сохранены в: {os.path.abspath(csv_filename)}\n")
    
    return best_params

def tune_pso_hyperparameters(gpa_list, comp_list, Q_target):
    """Поиск оптимальных гиперпараметров PSO методом сеточного перебора."""
    best_overall_cost = float('inf')
    best_params = {}
    c1_space = [1.2, 1.49, 2.0]
    w_space = [0.5, 0.72, 0.9]
    print("\n=== Оптимизация гиперпараметров PSO ===")
    for c1 in c1_space:
        for w in w_space:
            print(f"Тестирование: c1={c1}, c2={c1}, w={w}")
            reg = PyswarmsRegulator(n_particles=30, iters=40, c1=c1, c2=c1, w=w)
            reg.calculate(gpa_list, comp_list, pin_profile[0], Tin_ext, Q_target)
            current_cost = reg.optimizer.best_cost if hasattr(reg.optimizer, 'best_cost') else getattr(reg, 'best_cost', float('inf'))
            if current_cost < best_overall_cost:
                best_overall_cost = current_cost
                best_params = {'c1': c1, 'c2': c1, 'w': w}
    print(f"Лучшие параметры: {best_params} с ценой {best_overall_cost:.4f}")
    return best_params

# Раскомментировать для запуска подбора гиперпараметров:
# tune_aco_hyperparameters(gpa_instances, comp_instances, Q_TARGET_DAY)  # Оптимизация гиперпараметров ACO
# tune_pso_hyperparameters(gpa_instances, comp_instances, Q_TARGET_DAY)  # Оптимизация гиперпараметров PSO