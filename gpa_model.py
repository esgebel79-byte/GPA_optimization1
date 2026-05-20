# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
import logging

"""
Класс GPA для представления параметров газоперекачивающего агрегата
v0.1
"""

from dataclasses import dataclass
from gpa_constants import ( COMPRESSIBILITY_FACTOR, REF_PRESSURE_MPA,
                            GAS_MOLAR_MASS_KG_KMOL, MECHANICAL_EFFICIENCY, REF_TEMPERATURE_K,
                            FUEL_HEAT_VALUE_MJ_NM3,
                            POLYTROPIC_EXPONENT_RATIO, GTU_NOMINAL_POWER_MW)

@dataclass
class GPA:
    # -----------------------------------------------------------------
    #  Класс для хранения параметров газоперекачивающего агрегата (ГПА).
    # -----------------------------------------------------------------
        
    ID: str                           # Идентификатор ГПА
    
    # === Паспортные и режимные коэффициенты ===
    kn0: float = 0.95                 # Коэффициент технического состояния ГТУ [0..1]
    Qm: float = 35.0                  # Низшая теплота сгорания топлива, МДж/нм³ (взято среднее от 32 до 38)
    
    SR: float = 0.0                   # Помпажный запас

    # === Режимные ограничения ГПА ===
    Q_min_lim: float = 0.0      # Минимальный допустимый расход, тыс.нм³/ч
    Q_max_lim: float = 0.0      # Максимальный допустимый расход, тыс.нм³/ч
    Nst_min_lim: float = 0.0    # Минимальные допустимые обороты, об/мин
    Nst_max_lim: float = 0.0    # Максимальные допустимые обороты, об/мин
    
        
    # === Аэродинамические параметры от модели компрессора ===
    Eps: float = 0.0                  # Степень сжатия (от компрессора)
    Nst: float = 0.0                  # ФАКТИЧЕСКАЯ Частота вращения (от компрессора) об/мин
    Q: float = 0.0                    # Расход газа, тыс. нм³/ч (от компрессора)

    # === Давления, МПа ===
    Pin: float = 0.0
    Pout: float = 0.0 # Функция от Pin*Eps
    
    # === Температуры, °C ===
    Tin: float = 0.0
    Tout: float = 0.0 # Функция 

    # === Расходы ===
    Qtg: float = 0.0                  # Расход топливного газа, тыс. нм³/ч
    Qprod : float = 0.0               # Расход газа коммерческий, тыс. нм³/ч
    
    # === Мощности и КПД ===
    Nef: float = 0.0                  # Эффективная мощность ГПА, кВт
    Npol: float = 0.0                 # Политропический КПД ЦБН (от компрессора)
    Ner: float = 0.0                  # Располагаемая мощность ГПА, кВт

    # -----------------------------------------------------------------
    # Методы расчёта
    # -----------------------------------------------------------------
    """
    def calc_polytropic_efficiency(self) -> float:
        #Расчёт политропного КПД ЦБН методом Шульца - это обратная задача
       # Температурный коэффициент политропы.
        P_in_abs = self.Pin + REF_PRESSURE_MPA  
        P_out_abs = self.Pin + REF_PRESSURE_MPA
        T_in_K = self.Tin + 273.15
        T_out_K = self.Tout + 273.15
        
        mt = math.log(T_out_K / T_in_K) / math.log(P_out_abs / P_in_abs)
        
        # Показатель псевдоизэнтропы (эмпирическая зависимость)
        kk = (4.16 + 0.0041 * ((self.Tin + self.Tout) / 2 - 10.0) + 3.93 * (GAS_MOLAR_MASS_KG_KMOL - 0.55) + 5 * (mt - 0.3))
        
        # Политропный КПД, % → перевод в доли единицы
        npol_percent = 100.0 / (kk * mt)
        self.Npol = npol_percent / 100.0
        return self.Npol
     """
    def update_state(self, comp_out: dict, Pin: float, Tin: float, P_out_req: float = None):
     
        self.Pin = Pin
        self.Tin = Tin
      
        self.Eps = comp_out['epsilon']
        self.Npol = comp_out['Npol']
        self.Q = comp_out['Q']  # в м³/мин
        self.Nst = comp_out.get('Nst_abs', 0.0)
        self.SR = comp_out.get('SR', 0.0)

        # Pout из степени сжатия
        self.Pout = self.Pin * self.Eps
    
        # Tout через баланс энергий
        T_in_K = self.Tin + 273.15
        m = POLYTROPIC_EXPONENT_RATIO
        if self.Npol > 0.0:
            # T_out = T_in * [1 + (ε^m - 1) / (m * η_pol)]
            self.Tout = T_in_K * (1.0 + (self.Eps ** m - 1.0) / (m * self.Npol)) - 273.15
        else:
            self.Tout = self.Tin  # Защита от некорректного КПД

    def calc_production_flow(self) -> float: #Расчёт коммерческого расхода газа
        T_in_K = self.Tin + 273.15  
        P_in_abs = self.Pin #абсолютное
       # Z_ref при стандартных условиях 0.998–1.0
        Z_ref = 0.998
    
        self.Qprod = self.Q * (293.15 / T_in_K) * (Z_ref / COMPRESSIBILITY_FACTOR) * 60 / 1000.0 * self.Npol
        return self.Qprod

    def calc_compression_power(self) -> float: #Расчёт мощности сжатия ЦБН (переводится в эффективную мощность ГПА)
        # Газовая постоянная
        Rg = 8.31 / GAS_MOLAR_MASS_KG_KMOL
        
        # Температурный коэффициент политропы
        T_in_K = self.Tin + 273.15
        m = POLYTROPIC_EXPONENT_RATIO
        
        if self.Npol <= 0.0:
            self.Nef = 0.0
            return
        # self.Q в м3/мин
        self.Nef = (self.Q * Rg * T_in_K * (self.Eps ** m - 1.0) / m) / (self.Npol * MECHANICAL_EFFICIENCY * 1000.0)  # кВт
    
    #Расчёт располагаемой мощности ГПА
    def calc_available_power(self, TinD: float, Patm: float):
        T_in_K = self.Tin + 273.15
        temp_corr = 1.0 - ((TinD + 273.15 - T_in_K) / (TinD + 273.15))
        p_corr = Patm / 101.325
        self.Ner = GTU_NOMINAL_POWER_MW * self.kn0 * temp_corr * p_corr  # МВт
    
    #Расход топлива, кг/с
    def calc_fuel_flow(self, TinD: float):
        kLoad = self.Nef / self.Ner if self.Ner > 0 else 0.0 #Коэфф. загрузки
        Nef_nom = 0.325 #Номинальный эффективный КПД
        load_corr = 1.0 - 0.15 * (1.0 - kLoad) ** 2
        temp_corr = 1.0 - 0.002 * max(0.0, TinD - 15.0)
        self.Nef_GTU = max(0.15, min(0.32, Nef_nom * load_corr * temp_corr))
        
        N_turb_MW = self.Nef / 1000.0
        self.Qtg = (N_turb_MW * 3600.0) / (self.Nef_GTU * FUEL_HEAT_VALUE_MJ_NM3)  # м3/мин
    
    #
    def get_metrics(self) -> dict:
        return {
            'Nef': self.Nef, 'Ner': self.Ner, 'Qtg': self.Qtg,
            'Eps': self.Eps, 'Npol': self.Npol,
            'Pout': self.Pout, 'Tout': self.Tout,
            'kLoad': self.Nef / self.Ner if self.Ner > 0 else 0.0,
            'Q': self.Qprod # в тыс. м3/час
        }
    
    # Выход за ограничения
    def check_constraints(self) -> dict:
        limits = {
            'Q_below_min': self.Q_min_lim > 0 and self.Q < self.Q_min_lim,
            'Q_above_max': self.Q_max_lim > 0 and self.Q > self.Q_max_lim,
            'Nst_below_min': self.Nst_min_lim > 0 and self.Nst < self.Nst_min_lim,
            'Nst_above_max': self.Nst_max_lim > 0 and self.Nst > self.Nst_max_lim
        }
        limits['is_within_limits'] = not any(
            [limits['Q_below_min'], limits['Q_above_max'], 
            limits['Nst_below_min'], limits['Nst_above_max']]
        )
        return limits