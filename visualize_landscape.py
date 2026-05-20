#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация 3D-ландшафта целевой функции с учетом изменения давления Pin.
Строит графики для характерных точек суточного профиля (ночь, день, вечер).
"""
import numpy as np
import matplotlib.pyplot as plt
from functional import objective_function
from gpa_model import GPA
from compressor_model import Compressor
from Nst_GWOreg import GreyWolfRegulator

# Импорт генератора профиля (убедитесь, что имя файла совпадает с вашим)
try:
    from pressure_imit import generate_pin_profile
except ImportError:
    from pressure_imit import generate_pin_profile

def visualize_3d_landscape_varying_pin():
    # 1. Инициализация моделей (согласовано с exec.py v3.0)
    gpa_instances = [
        GPA(ID="1", Q_min_lim=210.0, Q_max_lim=620.0, Nst_min_lim=3200.0, Nst_max_lim=5000.0),
        GPA(ID="2", Q_min_lim=210.0, Q_max_lim=900.0, Nst_min_lim=5200.0, Nst_max_lim=11900.0),
        GPA(ID="3", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8200.0)
    ]
    comp_instances = [
        Compressor(Nst_nom=4000.0, Q_nom=600.0),
        Compressor(Nst_nom=5900.0, Q_nom=500.0),
        Compressor(Nst_nom=5700.0, Q_nom=500.0)
    ]
    # Целевой расход (в оригинале 210, в exec.py часовой ~100-125. Оставляем 210 для теста ландшафта)
    Q_target = 100.0 
    Tin = 20.0

    # 2. Получаем профиль давления и выбираем репрезентативные точки
    # Генерируем полные сутки
    full_profile = generate_pin_profile(hours=24, mode="daily_cycle", seed=42)
    
    # Выбираем 3 характерных часа: 00:00 (мин/среднее), 12:00 (пик), 23:00 (спад)
    hours_to_check = [0, 12, 23]
    pins_to_check = [full_profile[h] for h in hours_to_check]
    
    print(f"Выбранные значения Pin для визуализации: {[f'{p:.2f}' for p in pins_to_check]} МПа")

    # 3. Цикл по выбранным давлениям
    for idx, Pin in enumerate(pins_to_check):
        # Требуемое давление на выходе (фиксированная степень сжатия сети)
        P_out_req = Pin * 1.25  
        
        print(f"\n=== Расчет ландшафта для Pin = {Pin:.2f} МПа (час {hours_to_check[idx]}) ===")
        
        # Комбинации пар для визуализации: (индексы варьируемых, индекс фиксируемого)
        pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]  
        
        for i1, i2, ifx in pairs:
            # Диапазоны оборотов (уменьшено до 30 точек для ускорения расчета 3D-сетки)
            nst_range1 = np.linspace(gpa_instances[i1].Nst_min_lim, gpa_instances[i1].Nst_max_lim, 30)
            nst_range2 = np.linspace(gpa_instances[i2].Nst_min_lim, gpa_instances[i2].Nst_max_lim, 30)
            NST1, NST2 = np.meshgrid(nst_range1, nst_range2)
            Z_cost = np.zeros(NST1.shape)
            
            fixed_nst = (gpa_instances[ifx].Nst_min_lim + gpa_instances[ifx].Nst_max_lim) / 2.0
            
            print(f"  -> Пара ГПА-{i1+1} & ГПА-{i2+1} (ГПА-{ifx+1} фикс = {fixed_nst:.0f})...")
            
            for r in range(len(nst_range1)):
                for c in range(len(nst_range2)):
                    nst_vals = np.zeros(3)
                    nst_vals[i1] = NST1[r, c]
                    nst_vals[i2] = NST2[r, c]
                    nst_vals[ifx] = fixed_nst
                    
                    Qprod_arr, Qg_arr, Nst_arr, Eps_arr = [], [], [], []
                    
                    for gpa, comp, nst in zip(gpa_instances, comp_instances, nst_vals):
                        comp.Nst_current = nst
                        
                        # === ИНТЕГРАЦИЯ НОВОЙ ФИЗИКИ КОМПРЕССОРА ===
                        # Передаем Pin и P_out_req в компрессор для корректного расчета
                        if hasattr(comp, 'set_network_conditions'):
                            comp.set_network_conditions(Pin, P_out_req)
                            comp_out = comp.get_state_with_network(nst, Pin, P_out_req)
                        else:
                            # Fallback для старой модели
                            comp_out = comp.get_state()
                        # ===========================================
                        
                        gpa.update_state(comp_out, Pin=Pin, Tin=Tin)
                        gpa.calc_production_flow()
                        gpa.calc_compression_power()
                        gpa.calc_available_power(TinD=Tin, Patm=101.325)
                        gpa.calc_fuel_flow(TinD=Tin)
                        
                        Qprod_arr.append(gpa.Qprod)
                        Qg_arr.append(gpa.Qtg)
                        Nst_arr.append(gpa.Nst)
                        Eps_arr.append(gpa.Eps)
                        
                    res = objective_function(
                        gpa_list=[g.ID for g in gpa_instances], Q_target=Q_target, E_target=1.5,
                        Qprod_i=np.array(Qprod_arr), Qg_i=np.array(Qg_arr),
                        Nst_i=np.array(Nst_arr), Eps_i=np.array(Eps_arr),
                        Q_min_lim_i=np.array([g.Q_min_lim for g in gpa_instances]),
                        Q_max_lim_i=np.array([g.Q_max_lim for g in gpa_instances]),
                        Nst_min_lim_i=np.array([g.Nst_min_lim for g in gpa_instances]),
                        Nst_max_lim_i=np.array([g.Nst_max_lim for g in gpa_instances])
                    )
                    # Логарифмическое масштабирование для наглядности
                    Z_cost[r, c] = np.log10(res["total"] + 1.0)

            # Визуализация
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(NST1, NST2, Z_cost, cmap='viridis', edgecolor='none', alpha=0.85)
            ax.set_title(f'Ландшафт (log10)\nPin={Pin:.2f} МПа | ГПА-{i1+1} vs ГПА-{i2+1}')
            ax.set_xlabel(f'Обороты ГПА-{i1+1} (об/мин)')
            ax.set_ylabel(f'Обороты ГПА-{i2+1} (об/мин)')
            ax.set_zlabel('log10(Cost)')
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.tight_layout()
            plt.show()

def tune_and_visualize_gwo(gpa_list, comp_list, Q_target):
        # Наборы гиперпараметров для тестирования
        wolves_variants = [10, 30, 50]
        iters_variants = [20, 50, 100]
    
        plt.figure(figsize=(10, 6))
    
        best_overall_score = float('inf')
        best_config = {}

        for n_wolves in wolves_variants:
            for iters in iters_variants:
                print(f"Тест: Волков={n_wolves}, Итераций={iters}")
            
                regulator = GreyWolfRegulator(n_wolves=n_wolves, iters=iters)
                # Запуск расчета (Pin=5.0, Tin=20.0)
                regulator.calculate(gpa_list, comp_list, 5.0, 20.0, Q_target)
            
                # Визуализация линии сходимости для данной конфигурации
                plt.plot(regulator.cost_history, label=f'Wolves:{n_wolves}, Iters:{iters}')
            
                final_score = regulator.cost_history[-1]
                if final_score < best_overall_score:
                    best_overall_score = final_score
                    best_config = {'wolves': n_wolves, 'iters': iters}

        plt.yscale('log') # Логарифмическая шкала, так как штрафы могут быть огромными
        plt.title('Сходимость алгоритма Grey Wolf Optimizer')
        plt.xlabel('Итерация')
        plt.ylabel('Значение целевой функции (log)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()

        print(f"Оптимальная настройка: {best_config} с результатом {best_overall_score}")

def plot_wolf_hierarchy(alpha, beta, delta):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Визуализируем положение трех лучших волков в пространстве оборотов ГПА
    ax.scatter(alpha[0], alpha[1], alpha[2], c='gold', s=200, label='Alpha (Leader)', marker='*')
    ax.scatter(beta[0], beta[1], beta[2], c='silver', s=150, label='Beta')
    ax.scatter(delta[0], delta[1], delta[2], c='brown', s=100, label='Delta')
    
    ax.set_xlabel('Nst ГПА-1')
    ax.set_ylabel('Nst ГПА-2')
    ax.set_zlabel('Nst ГПА-3')
    plt.legend()
    plt.title("Иерархия лидеров в пространстве решений")
    plt.show()

if __name__ == "__main__":
    # 1. Инициализация оборудования (параметры приведены в соответствие с exec.py)
    gpa_list = [
        GPA(ID="1", Q_min_lim=210.0, Q_max_lim=800.0, Nst_min_lim=4200.0, Nst_max_lim=9900.0),
        GPA(ID="2", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8500.0),
        GPA(ID="3", Q_min_lim=210.0, Q_max_lim=720.0, Nst_min_lim=4200.0, Nst_max_lim=8200.0)
    ]
    comp_list = [
        Compressor(Nst_nom=6000.0, Q_nom=600.0),
        Compressor(Nst_nom=5900.0, Q_nom=500.0),
        Compressor(Nst_nom=5700.0, Q_nom=500.0)
    ]
    Q_target = 100.0  # Целевой часовой расход, тыс.м³/ч (адаптируйте под ваш сценарий)

    # 2. Вызов процедуры оптимизации гиперпараметров и отрисовки сходимости
    tune_and_visualize_gwo(gpa_list, comp_list, Q_target)

    # 3. Вызов отрисовки иерархии волков
    # Примечание: требует явной передачи координат alpha, beta, delta
    # Текущая реализация calculate() возвращает только alpha_pos.
    # Для демонстрации используются заглушки. Замените на реальные данные при необходимости.
    demo_alpha = np.array([4500.0, 4600.0, 4400.0])
    demo_beta  = demo_alpha + np.array([150.0, -120.0, 80.0])
    demo_delta = demo_alpha + np.array([-200.0, 250.0, -150.0])
    plot_wolf_hierarchy(demo_alpha, demo_beta, demo_delta)