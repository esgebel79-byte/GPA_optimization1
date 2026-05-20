#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GWO-регулятор (Grey Wolf Optimizer) для ГПА.
Имитирует иерархию и охоту серых волков.
"""
import numpy as np
from typing import Optional
from Nst_SHOPreg import NstRegulatorBase
from functional import objective_function

class GreyWolfRegulator(NstRegulatorBase):
    def __init__(self, n_wolves: int = 30, iters: int = 40, max_nst_step: float = 800.0):
        self.n_wolves = n_wolves
        self.iters = iters
        self.max_nst_step = max_nst_step  # Макс. изменение оборотов за шаг, об/мин
        self.cost_history = []

    def _evaluate_fitness(self, pos, gpa_list, comp_list, Pin, Tin, Q_target, P_out_req=None):
        Qprod_arr, Qg_arr, Nst_arr, Eps_arr = [], [], [], []
        for gpa, comp, nst in zip(gpa_list, comp_list, pos):
            # Согласование с новой сетевой моделью компрессора
            if P_out_req is not None and hasattr(comp, 'set_network_conditions'):
                comp.set_network_conditions(Pin, P_out_req)
                comp_out = comp.get_state_with_network(nst, Pin, P_out_req)
            else:
                comp.Nst_current = nst
                comp_out = comp.get_state()
                
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

    def calculate(self, gpa_list, comp_list, Pin, Tin, Q_target, prev_nst=None, **kwargs) -> np.ndarray:
        P_out_req = kwargs.get('P_out_req')
        dim = len(gpa_list)
        lb = np.array([g.Nst_min_lim for g in gpa_list])
        ub = np.array([g.Nst_max_lim for g in gpa_list])

        positions = np.random.uniform(lb, ub, (self.n_wolves, dim))
        alpha_pos, alpha_score = np.zeros(dim), float("inf")
        beta_pos, beta_score = np.zeros(dim), float("inf")
        delta_pos, delta_score = np.zeros(dim), float("inf")

        self.cost_history = []  # Сброс истории перед новым запуском

        for t in range(self.iters):
            # Оценка приспособленности и обновление лидеров
            for i in range(self.n_wolves):
                positions[i] = np.clip(positions[i], lb, ub)
                fitness = self._evaluate_fitness(positions[i], gpa_list, comp_list, Pin, Tin, Q_target, P_out_req)
                
                if fitness < alpha_score:
                    alpha_score, alpha_pos = fitness, positions[i].copy()
                elif fitness < beta_score:
                    beta_score, beta_pos = fitness, positions[i].copy()
                elif fitness < delta_score:
                    delta_score, delta_pos = fitness, positions[i].copy()

            self.cost_history.append(alpha_score)  # Запись истории сходимости

            # Обновление положений
            a = 2 - t * (2 / self.iters)
            for i in range(self.n_wolves):
                for j in range(dim):
                    r1, r2 = np.random.random(), np.random.random()
                    A1, C1 = 2*a*r1 - a, 2*r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    r1, r2 = np.random.random(), np.random.random()
                    A2, C2 = 2*a*r1 - a, 2*r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    r1, r2 = np.random.random(), np.random.random()
                    A3, C3 = 2*a*r1 - a, 2*r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    positions[i, j] = (X1 + X2 + X3) / 3
            
            positions = np.clip(positions, lb, ub)

        # Финальное положение до применения лимитов динамики
        final_pos = np.clip(alpha_pos, lb, ub)

        # === ОГРАНИЧЕНИЕ НА МАКСИМАЛЬНОЕ ИЗМЕНЕНИЕ ОБОРОТОВ ===
        if prev_nst is not None:
            prev_nst = np.asarray(prev_nst).flatten()
            delta = final_pos - prev_nst
            # Ограничиваем приращение заданным шагом
            delta_clipped = np.clip(delta, -self.max_nst_step, self.max_nst_step)
            final_pos = prev_nst + delta_clipped
            # Повторное ограничение физическими границами (защита от выхода за пределы при обрезке шага)
            final_pos = np.clip(final_pos, lb, ub)

        return final_pos