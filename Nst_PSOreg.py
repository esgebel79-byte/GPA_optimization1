#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSO-регулятор на базе библиотеки pyswarms.
v1.1
"""

import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history, plot_contour
from pyswarms.utils.plotters.formatters import Mesher
from typing import List, Optional
from Nst_SHOPreg import NstRegulatorBase
from gpa_model import GPA
from compressor_model import Compressor
from functional import objective_function

class PyswarmsRegulator(NstRegulatorBase):
    def __init__(self, n_particles: int = 30, iters: int = 100, c1=1.49, c2=1.49, w=0.72, delta_nst_max: float = 700.0):
        self.n_particles = n_particles
        self.iters = iters
        self.options = {"c1": c1, "c2": c2, "w": w}
        self.delta_nst_max = delta_nst_max  # Максимальное допустимое изменение оборотов за шаг
        self.optimizer = None

    # Удалён @staticmethod, так как метод вызывается через self и требует корректного связывания аргументов
    def _evaluate_particle(self, nst_vals: np.ndarray, gpa_list, comp_list, Pin, Tin, Q_target, P_out_req) -> float:
        """Расчёт функционала для одной частицы (одного набора оборотов) с учетом сети."""
        Qprod_arr, Qg_arr, Nst_arr, Eps_arr = [], [], [], []
        
        for gpa, comp, nst in zip(gpa_list, comp_list, nst_vals):
            comp.Nst_current = nst
            
            # --- АДАПТАЦИЯ ПОД НОВУЮ ФИЗИКУ КОМПРЕССОРА ---
            if hasattr(comp, 'set_network_conditions'):
                comp.set_network_conditions(Pin, P_out_req)
                comp_out = comp.get_state_with_network(nst, Pin, P_out_req)
            else:
                comp_out = comp.get_state()
            # -----------------------------------------------
            
            gpa.Q_min_lim = 210.0 * 60.0 / 1000.0 * 1.05
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
            gpa_list=[g.ID for g in gpa_list], Q_target=Q_target, E_target=1.5,
            Qprod_i=np.array(Qprod_arr), Qg_i=np.array(Qg_arr),
            Nst_i=np.array(Nst_arr), Eps_i=np.array(Eps_arr),
            Q_min_lim_i=np.array([g.Q_min_lim for g in gpa_list]),
            Q_max_lim_i=np.array([g.Q_max_lim for g in gpa_list]),
            Nst_min_lim_i=np.array([g.Nst_min_lim for g in gpa_list]),
            Nst_max_lim_i=np.array([g.Nst_max_lim for g in gpa_list])
        )
        return res["total"]

    def _make_fitness_func(self, gpa_list, comp_list, Pin, Tin, Q_target, P_out_req):
        """Создаёт векторизованную функцию пригодности для pyswarms."""
        def fitness(pos: np.ndarray) -> np.ndarray:
            n_part, dim = pos.shape
            costs = np.empty(n_part)
            bounds_min = [g.Nst_min_lim for g in gpa_list]
            bounds_max = [g.Nst_max_lim for g in gpa_list]
            
            for i in range(n_part):
                # Явное ограничение оборотов до передачи в модель
                nst_clipped = np.clip(pos[i], bounds_min, bounds_max)
                costs[i] = self._evaluate_particle(nst_clipped, gpa_list, comp_list, Pin, Tin, Q_target, P_out_req)
            return costs
        return fitness

    def calculate(self, gpa_list, comp_list, Pin, Tin, Q_target, prev_nst=None, P_out_req=None, **kwargs) -> np.ndarray:
        """
        Расчет оптимальных оборотов.
        P_out_req: требуемое давление на выходе (МПа). Если None, используется Pin * 1.25 как заглушка.
        """
        dim = len(gpa_list)
        base_min = np.array([g.Nst_min_lim for g in gpa_list])
        base_max = np.array([g.Nst_max_lim for g in gpa_list])
        
        # Динамические границы с учётом ограничения на шаг
        if prev_nst is not None:
            prev_nst_arr = np.asarray(prev_nst).flatten()
            min_bound = np.maximum(base_min, prev_nst_arr - self.delta_nst_max)
            max_bound = np.minimum(base_max, prev_nst_arr + self.delta_nst_max)
        else:
            min_bound = base_min
            max_bound = base_max

        # Защита от вырождения границ (min > max)
        min_bound = np.minimum(min_bound, max_bound - 1.0)
        
        bounds = (min_bound, max_bound)
        
        # Если P_out_req не передан, используем типовую степень сжатия
        if P_out_req is None:
            P_out_req = Pin * 1.25
            
        fitness_func = self._make_fitness_func(gpa_list, comp_list, Pin, Tin, Q_target, P_out_req)
        
        self.optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles, dimensions=dim,
            options=self.options, bounds=bounds
        )
        best_cost, best_pos = self.optimizer.optimize(fitness_func, iters=self.iters)
        
        # Сохраняем для визуализации и отладки
        self.best_pos = best_pos
        self.best_cost = best_cost
        self._opt_context = {
            'gpa': gpa_list, 'comp': comp_list, 
            'Pin': Pin, 'Tin': Tin, 'Q_target': Q_target, 'P_out_req': P_out_req,
            'bounds_min': bounds[0], 'bounds_max': bounds[1]
        }
        
        return np.clip(best_pos, bounds[0], bounds[1])

    def plot_contour_2d(self, dim1_idx=0, dim2_idx=1):
        """Визуализация траектории частиц в 2D-проекции."""
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            print("Ошибка: оптимизатор не инициализирован.")
            return
        if not hasattr(self, 'best_pos') or self.best_pos is None:
            print("Ошибка: оптимизация не завершена или best_pos не сохранён.")
            return

        # pyswarms хранит историю как список массивов. Преобразуем в единый тензор.
        pos_hist = self.optimizer.pos_history
        if isinstance(pos_hist, list):
            pos_hist = np.array(pos_hist)
        if pos_hist.ndim < 3 or pos_hist.shape[-1] < 2:
            print("Недостаточно данных для 2D-визуализации.")
            return
            
        pos_2d = pos_hist[:, :, [dim1_idx, dim2_idx]]
        
        if not hasattr(self, '_opt_context'):
            print("Ошибка: контекст оптимизации отсутствует.")
            return

        ctx = self._opt_context
        # Mesher требует функцию, принимающую два 1D-массива (x1, x2)
        def fitness_2d_wrapper(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            n_part = len(x1)
            full_pos = np.tile(ctx['bounds_max'][np.newaxis, :], (n_part, 1))
            full_pos[:, dim1_idx] = x1
            full_pos[:, dim2_idx] = x2
            # Векторный расчёт стоимостей
            costs = np.array([
                self._evaluate_particle(p, ctx['gpa'], ctx['comp'], ctx['Pin'], ctx['Tin'], ctx['Q_target']) 
                for p in full_pos
            ])
            return costs

        mesher = Mesher(func=fitness_2d_wrapper)
        mark_coords = (self.best_pos[dim1_idx], self.best_pos[dim2_idx])
        
        try:
            plot_contour(pos_history=pos_2d, mesher=mesher, mark=mark_coords)
            plt.title(f"Траектория частиц PSO (ось {dim1_idx+1} vs {dim2_idx+1})")
            plt.show()
        except Exception as e:
            print(f"Встроенный plotter pyswarms вызвал ошибку: {e}")
            print("Используется резервный метод отрисовки через matplotlib.")
            self._plot_contour_fallback(pos_2d, mark_coords)

    def _plot_contour_fallback(self, pos_2d: np.ndarray, mark: tuple):
        """Надёжная отрисовка траектории стандартными средствами matplotlib."""
        fig, ax = plt.subplots(figsize=(8, 6))
        # Рисуем траектории частиц (последняя итерация выделена)
        for t in range(pos_2d.shape[0]):
            ax.scatter(pos_2d[t, :, 0], pos_2d[t, :, 1], c='blue', alpha=0.3, s=10)
        # Выделяем финальное положение
        ax.scatter(pos_2d[-1, :, 0], pos_2d[-1, :, 1], c='darkblue', alpha=0.8, s=30, label='Финальные позиции')
        # Отмечаем найденный оптимум
        ax.scatter([mark[0]], [mark[1]], c='red', s=100, marker='X', label='Лучшее решение')
        
        ax.set_xlabel(f'Измерение {0+1} (об/мин)')
        ax.set_ylabel(f'Измерение {1+1} (об/мин)')
        ax.set_title('Траектория роя частиц (резервный отрисовщик)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_history(self):
        if self.optimizer:
            plot_cost_history(self.optimizer.cost_history)
            plt.title("График сходимости PSO (Cost History)")
            plt.xlabel("Итерация")
            plt.ylabel("Стоимость (Cost)")
            plt.show()
