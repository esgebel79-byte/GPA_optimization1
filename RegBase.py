# -*- coding: utf-8 -*-
"""
Базовый интерфейс для стратегий регулирования оборотов ГПА.
v1.2 (стабильная сигнатура)
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

class NstRegulatorBase(ABC):
    """Абстрактный базовый класс регуляторов. Все стратегии наследуют этот интерфейс."""
    @abstractmethod
    def calculate(
        self,
        gpa_list: List,
        comp_list: List,
        Q_target: float,
        prev_nst: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Расчёт оптимальных оборотов.
        **kwargs позволяет расширять сигнатуру дочерних классов без нарушения контракта.
        """
        pass