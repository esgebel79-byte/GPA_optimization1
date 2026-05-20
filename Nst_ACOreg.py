#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACO-регулятор (Ant Colony Optimization) для ГПА.
Имитирует поведение муравьёв при поиске оптимальных оборотов.
v1.0
"""
import numpy as np
from typing import Optional
from RegBase import NstRegulatorBase
from functional import objective_function


class AntColonyRegulator(NstRegulatorBase):
    def __init__(self, n_ants: int = 30, iters: int = 40, alpha: float = 1.0, 
                 beta: float = 2.0, rho: float = 0.1, q: float = 1.0, max_nst_step: float = 800.0):
        """
        Инициализация ACO-регулятора.
        
        Args:
            n_ants: количество муравьёв
            iters: количество итераций алгоритма
            alpha: важность феромона (обычно 1.0)
            beta: важность расстояния/привлекательности (обычно 2.0)
            rho: коэффициент испарения феромона (0-1)
            q: константа для обновления феромона
            max_nst_step: максимальное изменение оборотов за шаг
        """
        self.n_ants = n_ants
        self.iters = iters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_nst_step = max_nst_step
        self.cost_history = []
        self.best_cost = float("inf")
        self.best_solution = None

    def _evaluate_fitness(self, nst_vals, gpa_list, comp_list, Pin, Tin, Q_target, P_out_req=None):
        """Расчёт функционала для набора оборотов."""
        Qprod_arr, Qg_arr, Nst_arr, Eps_arr = [], [], [], []
        
        for gpa, comp, nst in zip(gpa_list, comp_list, nst_vals):
            comp.Nst_current = nst
            
            if hasattr(comp, 'set_network_conditions'):
                comp.set_network_conditions(Pin, P_out_req)
                comp_out = comp.get_state_with_network(nst, Pin, P_out_req)
            else:
                comp_out = comp.get_state()
            
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

    def _build_ant_solution(self, lb, ub, pheromone, attractiveness, n_segments=10):
        """
        Муравей строит решение путём выбора значений оборотов на основе феромонов и привлекательности.
        
        Args:
            lb: нижние границы оборотов
            ub: верхние границы оборотов
            pheromone: матрица феромонов [n_dimensions, n_segments]
            attractiveness: матрица привлекательности [n_dimensions, n_segments]
            n_segments: количество дискретных сегментов для каждого измерения
        """
        dim = len(lb)
        solution = np.zeros(dim)
        
        for i in range(dim):
            # Пробabilities based on pheromone and attractiveness
            tau = pheromone[i, :] ** self.alpha
            eta = attractiveness[i, :] ** self.beta
            probabilities = (tau * eta) / (tau * eta).sum()
            
            # Рулетка выбора сегмента
            segment_idx = np.random.choice(n_segments, p=probabilities)
            
            # Линейная интерполяция в пределах выбранного сегмента
            segment_min = lb[i] + (ub[i] - lb[i]) * segment_idx / n_segments
            segment_max = lb[i] + (ub[i] - lb[i]) * (segment_idx + 1) / n_segments
            
            solution[i] = np.random.uniform(segment_min, segment_max)
        
        return np.clip(solution, lb, ub)

    def calculate(self, gpa_list, comp_list, Pin, Tin, Q_target, prev_nst=None, P_out_req=None, **kwargs) -> np.ndarray:
        """
        Поиск оптимальных оборотов с использованием алгоритма муравьиной колонии.
        """
        dim = len(gpa_list)
        lb = np.array([g.Nst_min_lim for g in gpa_list])
        ub = np.array([g.Nst_max_lim for g in gpa_list])
        
        if P_out_req is None:
            P_out_req = Pin * 1.25

        # Инициализация феромонов и привлекательности
        n_segments = 10
        pheromone = np.ones((dim, n_segments))
        
        # Привлекательность (инверсия расстояния от центра диапазона)
        attractiveness = np.ones((dim, n_segments))
        for i in range(dim):
            for j in range(n_segments):
                segment_center = lb[i] + (ub[i] - lb[i]) * (j + 0.5) / n_segments
                attractiveness[i, j] = 1.0 / (1.0 + abs(segment_center - (lb[i] + ub[i]) / 2.0) / (ub[i] - lb[i]))
        
        self.best_cost = float("inf")
        self.best_solution = None
        self.cost_history = []
        
        # Основной цикл ACO
        for iteration in range(self.iters):
            ant_solutions = []
            ant_costs = []
            
            # Каждый муравей строит решение
            for ant in range(self.n_ants):
                solution = self._build_ant_solution(lb, ub, pheromone, attractiveness, n_segments)
                cost = self._evaluate_fitness(solution, gpa_list, comp_list, Pin, Tin, Q_target, P_out_req)
                
                ant_solutions.append(solution)
                ant_costs.append(cost)
            
            # Поиск лучшегосреди муравьёв
            best_ant_idx = np.argmin(ant_costs)
            best_iteration_cost = ant_costs[best_ant_idx]
            best_iteration_solution = ant_solutions[best_ant_idx]
            
            # Обновление глобального лучшего
            if best_iteration_cost < self.best_cost:
                self.best_cost = best_iteration_cost
                self.best_solution = best_iteration_solution.copy()
            
            self.cost_history.append(self.best_cost)
            
            # === ОБНОВЛЕНИЕ ФЕРОМОНОВ ===
            # Испарение феромонов
            pheromone *= (1.0 - self.rho)
            
            # Добавление нового феромона лучшего муравья
            for i in range(dim):
                best_segment = int((best_iteration_solution[i] - lb[i]) / (ub[i] - lb[i]) * n_segments)
                best_segment = np.clip(best_segment, 0, n_segments - 1)
                pheromone[i, best_segment] += self.q / best_iteration_cost
            
            # Добавление феромона глобальным лучшим
            for i in range(dim):
                global_segment = int((self.best_solution[i] - lb[i]) / (ub[i] - lb[i]) * n_segments)
                global_segment = np.clip(global_segment, 0, n_segments - 1)
                pheromone[i, global_segment] += self.q / (2.0 * self.best_cost)  # Дополнительный бонус
            
            # Ограничение феромонов от стагнации
            pheromone = np.clip(pheromone, 0.1, 10.0)
        
        # === ОГРАНИЧЕНИЕ НА МАКСИМАЛЬНОЕ ИЗМЕНЕНИЕ ОБОРОТОВ ===
        final_pos = np.clip(self.best_solution, lb, ub)
        
        if prev_nst is not None:
            prev_nst_arr = np.asarray(prev_nst).flatten()
            delta = final_pos - prev_nst_arr
            delta_clipped = np.clip(delta, -self.max_nst_step, self.max_nst_step)
            final_pos = prev_nst_arr + delta_clipped
            final_pos = np.clip(final_pos, lb, ub)
        
        return final_pos
