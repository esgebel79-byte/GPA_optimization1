#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор суточного профиля входного давления Pin для ГПА.
Основан на типовых колебаниях давления в магистральных газопроводах.
"""
import numpy as np
from typing import Optional

def generate_pin_profile(
    hours: int = 24,
    mode: str = "daily_cycle",
    base_pin: float = 5.8,
    amplitude: float = 0.6,
    noise_std: float = 0.04,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Возвращает массив давления Pin (МПа) для каждого часа симуляции.
    
    Параметры:
        hours: количество часов симуляции
        mode: тип профиля ("daily_cycle", "ramp", "step", "constant")
        base_pin: базовое давление, МПа
        amplitude: амплитуда колебаний, МПа
        noise_std: стандартное отклонение шума, МПа
        seed: seed для воспроизводимости
        
    Возвращает:
        np.ndarray формы (hours,)
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.linspace(0, 2 * np.pi, hours, endpoint=False)
    
    if mode == "daily_cycle":
        # Пик давления днём (часы 10-18), спад ночью
        # Сдвиг фазы на -π/2 делает максимум в t=π/2 (середина дня)
        pin = base_pin + amplitude * np.sin(t - np.pi / 2)
    elif mode == "ramp":
        # Плавное нарастание давления за сутки
        pin = np.linspace(base_pin - amplitude, base_pin + amplitude, hours)
    elif mode == "step":
        # Ступенчатое изменение (имитация переключения upstream-станции)
        pin = np.full(hours, base_pin)
        mid = hours // 2
        pin[mid:] += amplitude
    elif mode == "constant":
        pin = np.full(hours, base_pin)
    else:
        raise ValueError(f"Неподдерживаемый режим профиля: {mode}")
        
    # Добавление небольшого шума для имитации измерений/флуктуаций
    pin += np.random.normal(0, noise_std, hours)
    
    # Физические ограничения (типичный диапазон для МГ: 4.0 - 7.5 МПа)
    pin = np.clip(pin, 4.0, 7.5)
    
    return pin