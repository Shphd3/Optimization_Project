import time
import numpy as np
import matplotlib.pyplot as plt
# ============================================================
# Core forward / backward (classic, explicit, no shortcuts)
# ============================================================
def forward(
    x_t,
    beta,
    Q0,
    Alpha,
    L0,
    C0_all,
    Wq,
    We,
    Rsrp,
    Conn,
    Capa,
    ALL_Users_Traffic,
    ALL_Cells_Bw,
):
    # 1) RSRP + CIO
    rsrp_cio = Rsrp + x_t[:, np.newaxis]

    # 2) Softmax probability
    rsrp_norm = rsrp_cio - rsrp_cio.max(axis=0)
    exp_rsrp = np.exp(rsrp_norm * beta) * Conn
    sum_exp = exp_rsrp.sum(axis=0, keepdims=True)
    sum_exp[sum_exp == 0] = 1e-9
    Prob = exp_rsrp / sum_exp

    # 3) Load
    L_m_abs = np.sum((ALL_Users_Traffic * Prob) / Capa, axis=1)
    l_m_ratio = L_m_abs / ALL_Cells_Bw

    # 4) Costs
    qos_cost_m = func_Q(l_m_ratio, ALL_Cells_Bw, Q0, L0, Alpha)
    energy_cost_m = func_E(l_m_ratio, ALL_Cells_Bw, C0_all)

    # Total objective
    obj = np.sum(Wq * qos_cost_m + We * energy_cost_m)

    cache = {
        "Prob": Prob,
        "Load_m": l_m_ratio,
        "L_m_abs": L_m_abs,
        "Unit_Cost": Wq * qos_cost_m + We * energy_cost_m,
        "QoS_Cost": np.sum(Wq * qos_cost_m),
        "Energy_Cost": np.sum(We * energy_cost_m),
    }

    return obj, cache


def backward(
    x_t,
    cache,
    M,
    N,
    beta,
    Q0,
    Alpha,
    L0,
    C0_all,
    Wq,
    We,
    Conn,
    Capa,
    ALL_Users_Traffic,
    ALL_Cells_Bw,
):
    Prob = cache["Prob"]
    l_m_ratio = cache["Load_m"]

    # 1) Marginal cost w.r.t. load ratio
    grad_Q = grad_Q_to_l(l_m_ratio, ALL_Cells_Bw, Q0, L0, Alpha)
    grad_E = grad_E_to_l(l_m_ratio, ALL_Cells_Bw, C0_all)
    dJ_dl = Wq * grad_Q + We * grad_E  # (M,)

    # 2) Marginal cost U_mn = dJ/dl_m * dl_m/dProb_mn
    U_mn = dJ_dl[:, np.newaxis] * ALL_Users_Traffic / (
        Capa * ALL_Cells_Bw[:, np.newaxis]
    )

    # 3) Gradient w.r.t. X_i
    avg_U_n = np.sum(Prob * U_mn, axis=0)  # (N,)
    grad_obj_to_x = beta * np.sum(Prob * (U_mn - avg_U_n), axis=1)  # (M,)

    return grad_obj_to_x


# ============================================================
# Optimizers (classic + one improved/innovative)
# ============================================================
def grad_step(x_t, grad, lr=0.01):
    return x_t - lr * grad


def momentum_step(x_t, v_t, grad, lr=0.1, beta=0.9):
    v_new = beta * v_t + lr * grad
    x_new = x_t - v_new
    return x_new, v_new


def adagrad_step(x_t, G_t, grad, lr=0.1):
    eps = 1e-8
    G_new = G_t + grad**2
    x_new = x_t - lr * grad / (np.sqrt(G_new) + eps)
    return x_new, G_new


def rmsprop_step(x_t, G_t, grad, lr=0.1, beta=0.9):
    eps = 1e-8
    G_new = beta * G_t + (1 - beta) * grad**2
    x_new = x_t - lr * grad / (np.sqrt(G_new) + eps)
    return x_new, G_new


def adam_step(x_t, v_t, G_t, grad, t, lr=0.1, beta1=0.9, beta2=0.999):
    eps = 1e-8
    v_new = beta1 * v_t + (1 - beta1) * grad
    G_new = beta2 * G_t + (1 - beta2) * grad**2
    v_hat = v_new / (1 - beta1**t)
    G_hat = G_new / (1 - beta2**t)
    x_new = x_t - lr * v_hat / (np.sqrt(G_hat) + eps)
    return x_new, v_new, G_new


def lion_step(x_t, m_t, grad, lr=0.1, beta1=0.9, beta2=0.99):
    # Lion (sign-based momentum). Simple, fast, and robust.
    m_new = beta1 * m_t + (1 - beta1) * grad
    x_new = x_t - lr * np.sign(m_new)
    m_new = beta2 * m_new + (1 - beta2) * grad
    return x_new, m_new


# ============================================================
# Training loop (optimizer execution)
# ============================================================
def _clip_grad(grad, clip_norm):
    if clip_norm is None:
        return grad
    norm = np.linalg.norm(grad)
    if norm == 0:
        return grad
    scale = min(1.0, clip_norm / norm)
    return grad * scale


def run_optimizer(
    optimizer_name,
    x0,
    beta,
    Q0,
    Alpha,
    L0,
    C0_all,
    Wq,
    We,
    Rsrp,
    Conn,
    Capa,
    ALL_Users_Traffic,
    ALL_Cells_Bw,
    max_iteration=20000,
    tolerance=1e-3,
    tolerance_obj=1e-1,
    tolerance_grad=1.0,
    lr=0.1,
    normalize_x=True,
    clip_grad_norm=None,
    print_every=100,
    optimizer_cfg=None,
    weight_decay=0.0,
):
    if optimizer_cfg is None:
        optimizer_cfg = {}

    opt_name = optimizer_name.lower()
    decoupled_wd = optimizer_cfg.get("decoupled_wd", None)
    if opt_name.endswith("w"):
        base_opt = opt_name[:-1]
        if decoupled_wd is None:
            decoupled_wd = True
    else:
        base_opt = opt_name
        if decoupled_wd is None:
            decoupled_wd = False

    M = Rsrp.shape[0]
    N = Rsrp.shape[1]

    x_t = x0.copy()
    v_t = np.zeros_like(x_t)
    G_t = np.zeros_like(x_t)
    m_t = np.zeros_like(x_t)

    obj_pre = float("inf")
    start_time = time.time()

    history = {
        "x_path": [],
        "load": [],
        "obj": [],
        "obj_Q": [],
        "obj_E": [],
        "unit_cost": [],
        "grad_norm": [],
        "net_capa": [],
    }

    for i in range(1, max_iteration + 1):
        obj, cache = forward(
            x_t,
            beta,
            Q0,
            Alpha,
            L0,
            C0_all,
            Wq,
            We,
            Rsrp,
            Conn,
            Capa,
            ALL_Users_Traffic,
            ALL_Cells_Bw,
        )

        net_capa = network_capa_estimate(
            cache["Prob"], Capa, Conn, ALL_Cells_Bw
        )

        grad = backward(
            x_t, 
            cache,
            M,
            N,
            beta,
            Q0,
            Alpha,
            L0,
            C0_all,
            Wq,
            We,
            Conn,
            Capa,
            ALL_Users_Traffic,
            ALL_Cells_Bw,
        )

        wd = optimizer_cfg.get("weight_decay", weight_decay)
        grad_update = grad
        if wd and not decoupled_wd:
            grad_update = grad + wd * x_t
        grad_update = _clip_grad(grad_update, clip_grad_norm)

        if base_opt == "gd":
            x_new = grad_step(x_t, grad_update, lr=lr)
        elif base_opt == "momentum":
            x_new, v_t = momentum_step(
                x_t, v_t, grad_update, lr=lr, beta=optimizer_cfg.get("beta", 0.9)
            )
        elif base_opt == "adagrad":
            x_new, G_t = adagrad_step(x_t, G_t, grad_update, lr=lr)
        elif base_opt == "rmsprop":
            x_new, G_t = rmsprop_step(
                x_t, G_t, grad_update, lr=lr, beta=optimizer_cfg.get("beta", 0.9)
            )
        elif base_opt == "adam":
            x_new, v_t, G_t = adam_step(
                x_t,
                v_t,
                G_t,
                grad_update,
                i,
                lr=lr,
                beta1=optimizer_cfg.get("beta1", 0.9),
                beta2=optimizer_cfg.get("beta2", 0.999),
            )
        elif base_opt == "lion":
            x_new, m_t = lion_step(
                x_t,
                m_t,
                grad_update,
                lr=lr,
                beta1=optimizer_cfg.get("beta1", 0.9),
                beta2=optimizer_cfg.get("beta2", 0.99),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        if wd and decoupled_wd:
            x_new = x_new * (1 - lr * wd)

        if normalize_x:
            x_new = x_new - np.mean(x_new)

        x_diff = np.linalg.norm(x_new - x_t)
        grad_norm = np.linalg.norm(grad_update)

        obj_new, cache_new = forward(
            x_new,
            beta,
            Q0,
            Alpha,
            L0,
            C0_all,
            Wq,
            We,
            Rsrp,
            Conn,
            Capa,
            ALL_Users_Traffic,
            ALL_Cells_Bw,
        )
        net_capa_new = network_capa_estimate(
            cache_new["Prob"], Capa, Conn, ALL_Cells_Bw
        )

        should_stop = (
            x_diff < tolerance
            or abs(obj_pre - obj_new) < tolerance_obj
            or grad_norm < tolerance_grad
        )

        history["x_path"].append(x_new.copy())
        history["load"].append(cache_new["Load_m"])
        history["obj"].append(obj_new)
        history["obj_Q"].append(cache_new["QoS_Cost"])
        history["obj_E"].append(cache_new["Energy_Cost"])
        history["unit_cost"].append(cache_new["Unit_Cost"])
        history["grad_norm"].append(grad_norm)
        history["net_capa"].append(net_capa_new)

        if print_every and i % print_every == 0:
            end_time = time.time()
            print(
                f"[{optimizer_name}] Round {i} | "
                f"obj={obj_new:.4f}, max_load={np.max(cache_new['Load_m']):.4f}, "
                f"net_capa={net_capa_new:.4f}, x_diff={x_diff:.4f}, "
                f"grad_norm={grad_norm:.4f}, time={end_time - start_time:.4f}s"
            )
            start_time = time.time()

        x_t = x_new
        obj_pre = obj_new
        if should_stop:
            break

    history["x_final"] = x_t.copy()
    return history


def run_classic(
    x0,
    beta,
    Q0,
    Alpha,
    L0,
    C0_all,
    Wq,
    We,
    Rsrp,
    Conn,
    Capa,
    ALL_Users_Traffic,
    ALL_Cells_Bw,
    **kwargs,
):
    # Classic baseline: Adam with standard settings
    history = run_optimizer(
        "adam",
        x0,
        beta,
        Q0,
        Alpha,
        L0,
        C0_all,
        Wq,
        We,
        Rsrp,
        Conn,
        Capa,
        ALL_Users_Traffic,
        ALL_Cells_Bw,
        **kwargs,
    )

    # Expose for visualization in show.py
    globals()["gd_path"] = history["x_path"]
    globals()["gd_load"] = history["load"]
    globals()["gd_obj"] = history["obj"]
    globals()["gd_obj_Q"] = history["obj_Q"]
    globals()["gd_obj_E"] = history["obj_E"]
    globals()["gd_unit_cost"] = history["unit_cost"]
    globals()["gd_net_capa"] = history["net_capa"]

    return history


# ============================================================
# Ablation & visualization (separate block)
# ============================================================
def run_ablation(
    x0,
    beta,
    Q0,
    Alpha,
    L0,
    C0_all,
    Wq,
    We,
    Rsrp,
    Conn,
    Capa,
    ALL_Users_Traffic,
    ALL_Cells_Bw,
    optimizer_names=None,
    normalize_flags=None,
    weight_decay_flags=None,
    **kwargs,
):
    if optimizer_names is None:
        optimizer_names = [
            "gd",
            "momentum",
            "adagrad",
            "rmsprop",
            "adam",
            "adamw",
            "lion",
            "lionw",
        ]
    if normalize_flags is None:
        normalize_flags = [True, False]
    if weight_decay_flags is None:
        weight_decay_flags = [0.0, 1e-3]

    results = {}
    for name in optimizer_names:
        for norm_flag in normalize_flags:
            for wd in weight_decay_flags:
                wd_tag = "wd" if wd > 0 else "no_wd"
                tag = f"{name}_{wd_tag}_norm" if norm_flag else f"{name}_{wd_tag}_no_norm"
                results[tag] = run_optimizer(
                    name,
                    x0,
                    beta,
                    Q0,
                    Alpha,
                    L0,
                    C0_all,
                    Wq,
                    We,
                    Rsrp,
                    Conn,
                    Capa,
                    ALL_Users_Traffic,
                    ALL_Cells_Bw,
                    normalize_x=norm_flag,
                    weight_decay=wd,
                    **kwargs,
                )
    return results


def plot_ablation_obj(results, title="Objective Comparison"):
    plt.figure(figsize=(10, 5))
    for name, hist in results.items():
        plt.plot(hist["obj"], label=name)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_ablation_max_load(results, title="Max Load Comparison"):
    plt.figure(figsize=(10, 5))
    for name, hist in results.items():
        max_load = [np.max(l) for l in hist["load"]]
        plt.plot(max_load, label=name)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Max Load")
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_ablation_qe(results, title="QoS / Energy Comparison"):
    plt.figure(figsize=(10, 5))
    for name, hist in results.items():
        plt.plot(hist["obj_Q"], label=f"{name}-Q")
        plt.plot(hist["obj_E"], linestyle="--", label=f"{name}-E")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Placeholder for manual runs; keep empty to avoid side effects on import.
    pass