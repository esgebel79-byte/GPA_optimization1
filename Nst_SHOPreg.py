# -*- coding: utf-8 -*-
"""
ПИД-регулятор для распределения нагрузки между ГПА.
v2.7 (исправлена логика PV, устранено неконтролируемое накопление, 
убраны избыточные ограничения, добавлена проверка совместимости simple_pid)
"""
import numpy as np
from RegBase import NstRegulatorBase
from gpa_model import GPA
import numpy as np
from typing import Tuple

class PIDController:
    """Дискретный ПИД-регулятор с условным интегрированием и сбросом при смене знака ошибки."""
    
    def __init__(
        self,
        Kp: float = 50.0,
        Ki: float = 0.3,
        Kd: float = 1.5,
        setpoint: float = 0.0,
        sample_time: float = None,  # None = 1.0 (шаг моделирования 1 час)
        output_limits: Tuple[float, float] = (-800.0, 800.0),
        integral_limit: float = 200.0
    ):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.dt = 1.0 if sample_time is None else float(sample_time)
        self.output_min, self.output_max = output_limits
        self.integral_limit = integral_limit
        
        # Внутреннее состояние
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0

    def __call__(self, measurement: float) -> float:
        error = self.setpoint - measurement
        
        # 1. Сброс интегратора при смене знака ошибки
        if self._prev_error * error < 0:
            self._integral = 0.0
            
        # 2. Пропорциональная составляющая
        P = self.Kp * error
        
        # 3. Интегральная составляющая с ограничением
        self._integral += self.Ki * error * self.dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        I = self._integral
        
        # 4. Дифференциальная составляющая (по измерению, исключает derivative kick)
        D = -self.Kd * (measurement - self._prev_measurement) / self.dt
        
        # 5. Суммирование и ограничение выхода
        raw_output = P + I + D
        output = np.clip(raw_output, self.output_min, self.output_max)
        
        # 6. Anti-windup (conditional integration)
        # Если выход в насыщении и ошибка направлена в ту же сторону, откатываем приращение интегратора
        if (output >= self.output_max and error > 0) or \
           (output <= self.output_min and error < 0):
            self._integral -= self.Ki * error * self.dt
            
        self._prev_error = error
        self._prev_measurement = measurement
        return output

class PIDRegulator(NstRegulatorBase):
    def __init__(self, Kp: float = 100.0, Ki: float = 1.3, Kd: float = 0.0, 
                 nst_step_max: float = 500.0, integral_limit: float = 100.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.nst_step_max = nst_step_max
        self.pid_correction = 0.0
        self._prev_error = 0.0
        
        # Удалён импорт simple_pid. Используется собственная реализация:
        self.pid = PIDController(
            Kp=Kp, Ki=Ki, Kd=Kd,
            setpoint=0.0,
            sample_time=None,  # dt=1.0 соответствует часовому шагу CSV
            output_limits=(-nst_step_max, nst_step_max),
            integral_limit=integral_limit
        )
        # sample_time=None, proportional_on_measurement и auto_mode более не требуются

    def calculate(self, gpa_list, Q_target, Q_actual=0.0, prev_nst=None, **kwargs) -> np.ndarray:
        self.pid.setpoint = Q_target
        current_error = Q_target - Q_actual
        
     # if kwargs.get('full_reset', False):
     #     self.pid.reset()
            
        self._prev_error = current_error
        
        # Вызов регулятора. Знак ошибки обрабатывается внутри PIDController
        raw_pid_out = self.pid(Q_actual)
        self.pid_correction = raw_pid_out

        # 2. Распределение корректировки между ГПА
        nst_new = np.zeros(len(gpa_list))
        if prev_nst is None:
            prev_nst = np.array([g.Nst_min_lim for g in gpa_list])

        total_weight = sum(1.0 / max(0.1, g.Q_max_lim - g.Q_min_lim) for g in gpa_list)

        for i, gpa in enumerate(gpa_list):
            weight = (1.0 / max(0.1, gpa.Q_max_lim - gpa.Q_min_lim)) / total_weight
            delta = self.pid_correction * weight
            
            # Ограничение только по физическим пределам агрегата
            nst_new[i] = np.clip(prev_nst[i] + delta, gpa.Nst_min_lim, gpa.Nst_max_lim)

        return nst_new