# -*- coding: utf-8 -*-
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

"""
Улучшенная модель ЦБН (центробежного нагнетателя) с нелинейной характеристикой.
Основана на типовых зависимостях ε(Q, N) и η(Q, N) для центробежных компрессоров.

Согласно теории центробежных машин:
- Характеристика компрессора имеет параболическую форму
- Существует линия помпажа (минимальный устойчивый расход)
- КПД максимален в расчетной точке и падает при отклонении
- Степень сжатия зависит как от оборотов, так и от расхода
"""

@dataclass
class Compressor:
    """
    Улучшенная модель ЦБН с нелинейной характеристикой.
    Реализует зависимость ε(Q,N) и η(Q,N) на основе типовых характеристик.
    """
    def __init__(self, Nst_nom: float, Q_nom: float):
        self.Nst_nom = Nst_nom      # Номинальные обороты, об/мин (4800)
        self.Q_nom = Q_nom          # Номинальный расход на всасывании, м³/мин (363)
        self.Nst_current = Nst_nom
        
        # Параметры помпажной границы
        self.Qsr_min = 210.0        # Минимальный расход для безпомпажного режима, м³/мин
        
        # Параметры характеристики компрессора
        self.psi_nom = 0.55         # Номинальный коэффициент напора (typical: 0.45-0.65)
        self.eta_max = 0.82         # Максимальный политропный КПД
        self.a_char = 0.5           # Коэффициент формы характеристики (0.3-0.7)
        
        # Параметры для расчета рабочей точки
        self.P_in_current = 6.0     # Текущее давление на входе, МПа (будет обновлено)
        self.P_out_required = 7.5   # Требуемое давление на выходе, МПа (будет обновлено)
    
    def calculate_epsilon_from_rpm_and_flow(self, Nst: float, Q: float) -> float:
        """
        Расчет степени сжатия как функции оборотов И расхода.
        
        Согласно теории центробежных компрессоров:
        ε = f(N, Q) через коэффициент напора ψ и коэффициент расхода φ
        
        Характеристика имеет вид:
        - При Q → 0 (помпаж): ε максимальна
        - При Q → Q_max (дросселирование): ε → 1
        - При фиксированных N: зависимость ε(Q) параболическая
        
        Args:
            Nst: частота вращения, об/мин
            Q: объемный расход на всасывании, м³/мин
            
        Returns:
            Степень сжатия ε = P_out / P_in
        """
        if Nst <= 0 or Q <= 0:
            return 1.0
            
        N_red = Nst / self.Nst_nom          # Относительные обороты
        Q_red = Q / (self.Q_nom * N_red)    # Относительный расход
        
        # Коэффициент напора зависит от квадрата относительных оборотов
        psi = self.psi_nom * (N_red ** 2)
        
        # Характеристика компрессора (параболическая зависимость)
        # ε = 1 + ψ * (1 - a * (Q/Q_nom)^2)
        # где a — коэффициент формы характеристики
        epsilon = 1.0 + psi * (1.0 - self.a_char * (Q_red ** 2))
        
        # Ограничения: ε не может быть < 1 (компрессор не может снижать давление)
        return max(1.0, epsilon)
    
    def get_actual_flow_from_network(self, Nst: float, P_in: float, P_out_req: float) -> float:
        """
        Определение расхода из пересечения характеристики компрессора 
        и характеристики сети (рабочая точка).
        
        Рабочая точка находится там, где:
        ε_compressor(Q, N) = ε_network = P_out_req / P_in
        
        Используется простой итерационный метод поиска.
        
        Args:
            Nst: частота вращения, об/мин
            P_in: давление на входе, МПа
            P_out_req: требуемое давление на выходе (от системы), МПа
            
        Returns:
            Установившийся расход Q, м³/мин
        """
        if Nst <= 0 or P_in <= 0 or P_out_req <= 0:
            return self.Qsr_min * 1.1
        
        # Требуемая степень сжатия от системы
        epsilon_req = P_out_req / P_in
        
        # Начальное приближение: номинальный расход с учетом оборотов
        N_red = Nst / self.Nst_nom
        Q = self.Q_nom * N_red * 0.8  # Начинаем с 80% от номинала
        
        # Итерационный поиск рабочей точки (метод простой итерации)
        for iteration in range(30):
            epsilon_comp = self.calculate_epsilon_from_rpm_and_flow(Nst, Q)
            
            # Относительная ошибка
            rel_error = (epsilon_comp - epsilon_req) / epsilon_req
            
            # Корректировка расхода
            if abs(rel_error) < 0.001:  # Сходимость достигнута
                break
                
            if epsilon_comp > epsilon_req:
                # Компрессор может больше — увеличиваем расход (движемся вправо по характеристике)
                Q *= (1.0 + 0.1 * rel_error)
            else:
                # Компрессор не дотягивает — уменьшаем расход (движемся влево)
                Q *= (1.0 - 0.15 * abs(rel_error))
            
            # Ограничения расхода
            Q_min = self.Qsr_min * 1.05  # +5% запас от помпажа
            Q_max = self.Q_nom * N_red * 1.5  # Максимум 150% от номинала
            
            Q = max(Q_min, min(Q, Q_max))
        
        # Финальная проверка на помпаж
        if Q < self.Qsr_min:
            Q = self.Qsr_min * 1.1
        
        return Q
    
    def calculate_Npol_from_rpm_and_flow(self, Nst: float, Q: float) -> float:
        """
        Расчет политропного КПД как функции оборотов и расхода.
        
        КПД имеет колоколообразную зависимость с максимумом в расчетной точке.
        При отклонении от оптимального расхода (помпаж или дросселирование) КПД падает.
        
        Согласно теории: η = f(φ, ψ), где φ — коэффициент расхода
        
        Args:
            Nst: частота вращения, об/мин
            Q: объемный расход, м³/мин
            
        Returns:
            Политропный КПД (0.65 - 0.82)
        """
        if Nst <= 0 or Q <= 0:
            return 0.65
        
        N_red = Nst / self.Nst_nom
        Q_opt = self.Q_nom * N_red  # Оптимальный расход при данных оборотах
        Q_red = Q / Q_opt           # Относительный расход
        
        # Колоколообразная кривая (гауссиана) с максимумом при Q_red ≈ 1.0
        # Ширина кривой: σ ≈ 0.3 (типичное значение для ЦБН)
        sigma = 0.35
        eta = self.eta_max * np.exp(-0.5 * ((Q_red - 1.0) / sigma) ** 2)
        
        # Дополнительное падение КПД при отклонении оборотов от номинала
        rpm_factor = 1.0 - 0.08 * ((N_red - 1.0) ** 2)
        eta *= rpm_factor
        
        # Ограничения КПД
        return max(0.65, min(eta, self.eta_max))
    
    def check_surge(self, Q: float) -> bool:
        """
        Проверка режима на помпаж.
        
        Условие безопасной работы: Q / Q_sr_min >= 1.1 (запас 10%)
        
        Args:
            Q: текущий расход, м³/мин
            
        Returns:
            True если режим безопасен, False если близок к помпажу
        """
        return Q >= self.Qsr_min * 1.1
    
    def get_state_with_network(self, Nst: float, P_in: float, P_out_req: float) -> dict:
        """
        Полное состояние компрессора с учетом характеристики сети.
        
        Это основной метод для взаимодействия с GPA.
        Сначала определяется рабочая точка (расход), затем рассчитываются
        все параметры: ε, η, SR.
        
        Args:
            Nst: частота вращения, об/мин
            P_in: давление на входе, МПа
            P_out_req: требуемое давление на выходе (от системы), МПа
            
        Returns:
            Словарь с параметрами компрессора
        """
        # Сохраняем текущие условия для возможного использования
        self.P_in_current = P_in
        self.P_out_required = P_out_req
        
        # 1. Определение расхода из пересечения характеристик
        Q = self.get_actual_flow_from_network(Nst, P_in, P_out_req)
        
        # 2. Расчет степени сжатия для найденного расхода
        epsilon = self.calculate_epsilon_from_rpm_and_flow(Nst, Q)
        
        # 3. Расчет КПД для рабочей точки
        Npol = self.calculate_Npol_from_rpm_and_flow(Nst, Q)
        
        # 4. Расчет помпажного запаса
        SR = (Q - self.Qsr_min) / self.Qsr_min if self.Qsr_min > 0 else 0.0
        is_surge_safe = self.check_surge(Q)
        
        # 5. Относительные параметры
        N_red = Nst / self.Nst_nom if self.Nst_nom > 0 else 0.0
        
        return {
            'Nst_abs': Nst,
            'Nst_red': N_red,
            'epsilon': epsilon,
            'Npol': Npol,
            'Q': Q,  # м³/мин на всасывании
            'is_surge': not is_surge_safe,
            'SR': SR,
            'in_safe_zone': N_red > 0.88,  # Безопасная зона по оборотам
            'P_in': P_in,
            'P_out_actual': P_in * epsilon,
            'P_out_required': P_out_req
        }
    
    def get_state(self) -> dict:
        """
        Обратная совместимость: упрощенный метод get_state.
        Используется, когда параметры сети не переданы явно.
        """
        # Используем сохраненные или дефолтные значения
        return self.get_state_with_network(
            Nst=self.Nst_current,
            P_in=self.P_in_current,
            P_out_req=self.P_out_required
        )
    
    def set_network_conditions(self, P_in: float, P_out_req: float):
        """
        Установка условий сети для последующих расчетов.
        
        Args:
            P_in: давление на входе, МПа
            P_out_req: требуемое давление на выходе, МПа
        """
        self.P_in_current = P_in
        self.P_out_required = P_out_req