import numpy as np
"""
Согласно формуле, в функционал от ГПА передается коммерческий расход Qprod[i],
Объемный расход топливного газа, кг/ч Gg[i]
g[i] - запас до верх/нижн огр. lolim[i], hilim[i] - верхние и нижние ограничения
v0.1
"""

def objective_function(
    gpa_list, Q_target, E_target,
    Qprod_i, Qg_i, Nst_i, Eps_i,
    Q_min_lim_i, Q_max_lim_i,
    Nst_min_lim_i, Nst_max_lim_i
):
    if Qprod_i is None or Qg_i is None or Nst_i is None:
        raise ValueError("Входные массивы параметров не могут быть None.")

    # Веса: высокий приоритет на достижение целевого расхода, топливный расход вторичен
    weights = {"w_Q": 500.0, "w_F": 1.0, "w_B": 5.0}

    n_gpa = len(gpa_list)
    if n_gpa == 0:
        return {"total": 0.0, "regulation_error": 0.0, "fuel_total": 0.0}

    q_term = 0.0
    f_term = 0.0
    b_term = 0.0
    total_penalty = 0.0

    Qprod_total = np.sum(Qprod_i)
    Qg_total = np.sum(Qg_i)
    Q_avg = float(np.mean(Qprod_i)) if n_gpa > 0 else 0.0

    # 1. Штраф за отклонение СУММАРНОГО расхода от целевого (Цель №1)
    q_total_err = Q_target - Qprod_total
    q_term += weights["w_Q"] * (q_total_err ** 2)

    for i in range(n_gpa):
        q_i = Qprod_i[i]
        n_i = Nst_i[i]
        penalty_i = 0.0

        # 2. Минимизация расхода топлива (Цель №2)
        f_term += weights["w_F"] * Qg_i[i]

        # Балансировка нагрузки между агрегатами (стабилизирует оптимизацию)
        b_term += weights["w_B"] * ((q_i - Q_avg) ** 2)

        # Штрафы за выход за рамки расхода
        if Q_max_lim_i is not None and q_i > Q_max_lim_i[i]:
            penalty_i += (q_i - Q_max_lim_i[i]) ** 2
        if Q_min_lim_i is not None and q_i < Q_min_lim_i[i]:
            penalty_i += (Q_min_lim_i[i] - q_i) ** 2

        # Штрафы за выход за рамки оборотов
        if Nst_max_lim_i is not None and n_i > Nst_max_lim_i[i]:
            penalty_i += (n_i - Nst_max_lim_i[i]) ** 2
        if Nst_min_lim_i is not None and n_i < Nst_min_lim_i[i]:
            penalty_i += (Nst_min_lim_i[i] - n_i) ** 2

        total_penalty += penalty_i

    total = q_term + f_term + b_term + total_penalty

    return {
        "total": total,
        "q_target": Q_target,
        "q_fact": Qprod_total,
        "regulation_error": q_total_err,
        "q_term": q_term,
        "f_term": f_term,
        "b_term": b_term,
        "constraint_term": total_penalty,
        "fuel_total": Qg_total
    }