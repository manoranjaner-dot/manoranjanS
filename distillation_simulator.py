#!/usr/bin/env python3
"""
Distillation Column Simulator

Features
--------
1) Binary mixture McCabe-Thiele simulation:
   - Theoretical stages via stepping method
   - Feed tray location
   - Reflux and operating lines
   - Optional tray efficiency for actual trays
   - Plot generation

2) Multicomponent shortcut (FUG: Fenske-Underwood-Gilliland):
   - Minimum stages (Fenske)
   - Minimum reflux ratio (Underwood, simplified)
   - Theoretical stages estimate (Gilliland)
   - Actual stages with efficiency

3) Column profile estimates:
   - Rectifying/stripping tray temperatures
   - Pressure drop per tray and total pressure drop
   - Top/bottom product summaries
   - Reboiler duty and steam flow estimate

IMPORTANT
---------
This is an engineering shortcut simulator for preliminary design/screening.
For detailed design, use rigorous VLE and MESH-equation simulation tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

try:
    import numpy as np
except Exception:  # optional dependency fallback
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:  # optional dependency fallback
    plt = None


# -------------------------------
# Data classes
# -------------------------------

@dataclass
class Component:
    name: str
    molecular_weight: float
    cp_liq_kj_per_kmolK: float
    latent_heat_kj_per_kmol: float
    antoine_A: float
    antoine_B: float
    antoine_C: float
    relative_volatility_ref: Optional[float] = None


@dataclass
class FeedCondition:
    pressure_kpa: float
    temperature_C: float
    flow_kmol_h: float
    mole_fractions: Dict[str, float]
    q_value: float = 1.0


@dataclass
class ColumnSpec:
    distillate_rate_kmol_h: float
    distillate_key_fraction: float
    bottoms_key_fraction: float
    reflux_ratio: Optional[float] = None
    tray_efficiency: float = 0.7
    pressure_drop_per_tray_kpa: float = 0.5
    top_pressure_kpa: float = 101.3
    reflux_subcooling_C: float = 2.0
    steam_latent_heat_kj_per_kg: float = 2200.0


@dataclass
class SimulationResult:
    mode: str
    theoretical_trays: float
    actual_trays: float
    feed_tray: Optional[int]
    total_pressure_drop_kpa: float
    top_temperature_C: float
    bottom_temperature_C: float
    total_temperature_difference_C: float
    distillate_flow_kmol_h: float
    bottoms_flow_kmol_h: float
    distillate_composition: Dict[str, float]
    bottoms_composition: Dict[str, float]
    reflux_flow_kmol_h: float
    reflux_temperature_C: float
    product_draw_kmol_h: float
    rectifying_tray_temps_C: List[float] = field(default_factory=list)
    stripping_tray_temps_C: List[float] = field(default_factory=list)
    rectifying_tray_pressures_kpa: List[float] = field(default_factory=list)
    stripping_tray_pressures_kpa: List[float] = field(default_factory=list)
    reboiler_duty_kj_h: float = 0.0
    steam_flow_kg_h: float = 0.0
    reboiler_inlet_temp_C: float = 0.0
    reboiler_outlet_temp_C: float = 0.0
    minimum_reflux_ratio: Optional[float] = None
    minimum_stages: Optional[float] = None


# -------------------------------
# Utility thermodynamics
# -------------------------------

def normalize_composition(z: Dict[str, float]) -> Dict[str, float]:
    s = sum(z.values())
    if s <= 0:
        raise ValueError("Composition sum must be positive.")
    return {k: v / s for k, v in z.items()}


def bubble_point_binary_T_C(x1: float, P_kpa: float, c1: Component, c2: Component) -> float:
    """
    Ideal Raoult-law binary bubble point using Antoine equation.
    Antoine form used: log10(Psat_mmHg) = A - B/(C + T_C)
    """
    P_mmHg = P_kpa * 760.0 / 101.325

    def f(T):
        p1 = 10 ** (c1.antoine_A - c1.antoine_B / (c1.antoine_C + T))
        p2 = 10 ** (c2.antoine_A - c2.antoine_B / (c2.antoine_C + T))
        return x1 * p1 + (1 - x1) * p2 - P_mmHg

    lo, hi = -50.0, 250.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def y_equilibrium_binary(x: float, alpha: float) -> float:
    return (alpha * x) / (1 + (alpha - 1) * x)


# -------------------------------
# Core simulator
# -------------------------------

class DistillationColumnSimulator:
    def __init__(self, components: List[Component], feed: FeedCondition, spec: ColumnSpec):
        self.components = components
        self.feed = feed
        self.spec = spec
        self.feed.mole_fractions = normalize_composition(self.feed.mole_fractions)

    def run(self, make_plot: bool = True, plot_path: str = "mccabe_thiele.png") -> SimulationResult:
        if len(self.components) == 2:
            return self._run_binary(make_plot, plot_path)
        return self._run_multicomponent_shortcut()

    def _run_binary(self, make_plot: bool, plot_path: str) -> SimulationResult:
        c_light = self.components[0]
        c_heavy = self.components[1]
        key = c_light.name

        zF = self.feed.mole_fractions[key]
        D = self.spec.distillate_rate_kmol_h
        F = self.feed.flow_kmol_h
        B = F - D

        xD = self.spec.distillate_key_fraction
        xB = self.spec.bottoms_key_fraction

        if not (0 < xB < zF < xD < 1):
            raise ValueError("Expected xB < zF < xD and all between 0 and 1 for binary separation.")

        # overall/light-key balance
        if B <= 0:
            raise ValueError("Bottoms flow must be positive. Check distillate rate.")

        alpha = c_light.relative_volatility_ref or 2.2

        R = self.spec.reflux_ratio if self.spec.reflux_ratio is not None else 1.8

        q = self.feed.q_value

        def rect_line(x):
            return (R / (R + 1)) * x + xD / (R + 1)

        if abs(q - 1.0) < 1e-6:
            x_int = zF
            y_int = rect_line(x_int)
        else:
            # intersection of q-line and rectifying line
            # q-line: y = q/(q-1) x - zF/(q-1)
            m_q = q / (q - 1)
            b_q = -zF / (q - 1)
            m_r = R / (R + 1)
            b_r = xD / (R + 1)
            x_int = (b_r - b_q) / (m_q - m_r)
            y_int = rect_line(x_int)

        # stripping line through (xB,xB) and feed intersection
        m_s = (y_int - xB) / (x_int - xB)
        b_s = xB - m_s * xB

        def strip_line(x):
            return m_s * x + b_s

        # McCabe-Thiele stepping
        x = xD
        stages = 0
        feed_stage = None
        xs = [xD]
        ys = [xD]

        while x > xB and stages < 500:
            y = ys[-1]
            # horizontal to equilibrium: invert y = alpha x /(1 + (alpha-1)x)
            x_eq = y / (alpha - y * (alpha - 1))
            x_eq = max(0.0, min(1.0, x_eq))
            xs.append(x_eq)
            ys.append(y)

            # vertical to operating line
            if x_eq >= x_int:
                y_new = rect_line(x_eq)
            else:
                if feed_stage is None:
                    feed_stage = stages + 1
                y_new = strip_line(x_eq)
            xs.append(x_eq)
            ys.append(y_new)

            x = x_eq
            stages += 1

            if x <= xB:
                break

        theoretical = float(stages)
        actual = theoretical / max(self.spec.tray_efficiency, 1e-6)

        T_top = bubble_point_binary_T_C(xD, self.spec.top_pressure_kpa, c_light, c_heavy)
        P_bottom = self.spec.top_pressure_kpa + actual * self.spec.pressure_drop_per_tray_kpa
        T_bottom = bubble_point_binary_T_C(xB, P_bottom, c_light, c_heavy)

        reflux_flow = R * D
        product_draw = D

        cp_avg = 0.5 * (c_light.cp_liq_kj_per_kmolK + c_heavy.cp_liq_kj_per_kmolK)
        latent_avg = 0.5 * (c_light.latent_heat_kj_per_kmol + c_heavy.latent_heat_kj_per_kmol)

        Q_reboiler = B * latent_avg + B * cp_avg * max(T_bottom - self.feed.temperature_C, 0)
        steam_flow = Q_reboiler / self.spec.steam_latent_heat_kj_per_kg

        n_rect = max(1, int((feed_stage or int(theoretical / 2)) - 1))
        n_strip = max(1, int(theoretical - n_rect))

        rect_temps = [T_top + i * (((T_top + T_bottom) / 2 - T_top) / max(n_rect - 1, 1)) for i in range(n_rect)]
        strip_temps = [((T_top + T_bottom) / 2) + i * ((T_bottom - (T_top + T_bottom) / 2) / max(n_strip - 1, 1)) for i in range(n_strip)]

        rect_press = [self.spec.top_pressure_kpa + i * self.spec.pressure_drop_per_tray_kpa for i in range(n_rect)]
        strip_press_start = rect_press[-1] + self.spec.pressure_drop_per_tray_kpa
        strip_press = [strip_press_start + i * self.spec.pressure_drop_per_tray_kpa for i in range(n_strip)]

        if make_plot:
            self._plot_mccabe(alpha, xD, xB, zF, R, q, x_int, xs, ys, plot_path)

        return SimulationResult(
            mode="binary_mccabe_thiele",
            theoretical_trays=theoretical,
            actual_trays=actual,
            feed_tray=feed_stage,
            total_pressure_drop_kpa=actual * self.spec.pressure_drop_per_tray_kpa,
            top_temperature_C=T_top,
            bottom_temperature_C=T_bottom,
            total_temperature_difference_C=T_bottom - T_top,
            distillate_flow_kmol_h=D,
            bottoms_flow_kmol_h=B,
            distillate_composition={c_light.name: xD, c_heavy.name: 1 - xD},
            bottoms_composition={c_light.name: xB, c_heavy.name: 1 - xB},
            reflux_flow_kmol_h=reflux_flow,
            reflux_temperature_C=T_top - self.spec.reflux_subcooling_C,
            product_draw_kmol_h=product_draw,
            rectifying_tray_temps_C=rect_temps,
            stripping_tray_temps_C=strip_temps,
            rectifying_tray_pressures_kpa=rect_press,
            stripping_tray_pressures_kpa=strip_press,
            reboiler_duty_kj_h=Q_reboiler,
            steam_flow_kg_h=steam_flow,
            reboiler_inlet_temp_C=T_bottom - 10,
            reboiler_outlet_temp_C=T_bottom + 5,
        )

    def _run_multicomponent_shortcut(self) -> SimulationResult:
        # Assume first is light key and second is heavy key for shortcut demo
        lk = self.components[0]
        hk = self.components[1]

        z = self.feed.mole_fractions
        key_lk = lk.name
        key_hk = hk.name

        xD_lk = self.spec.distillate_key_fraction
        xB_lk = self.spec.bottoms_key_fraction

        F = self.feed.flow_kmol_h
        D = self.spec.distillate_rate_kmol_h
        B = F - D

        alpha_lk_hk = (lk.relative_volatility_ref or 2.5) / (hk.relative_volatility_ref or 1.0)
        alpha_lk_hk = max(alpha_lk_hk, 1.05)

        # Fenske minimum stages
        xD_hk = max(1 - xD_lk, 1e-6)
        xB_hk = max(1 - xB_lk, 1e-6)
        Nmin = math.log((xD_lk / xB_lk) * (xB_hk / xD_hk)) / math.log(alpha_lk_hk)
        Nmin = max(Nmin, 1.0)

        # Underwood simplified estimate
        Rmin = max(0.2, (alpha_lk_hk * xD_lk / (alpha_lk_hk - 1)) - 1)
        R = self.spec.reflux_ratio if self.spec.reflux_ratio is not None else 1.4 * Rmin
        R = max(R, 1.05 * Rmin)

        # Gilliland correlation (Eduljee explicit approximation)
        X = (R - Rmin) / (R + 1)
        Y = 1 - math.exp((1 + 54.4 * X) * (X - 1) / (11 + 117.2 * X))
        N = (Nmin + Y) / (1 - Y)

        theoretical = float(max(N, Nmin + 1))
        actual = theoretical / max(self.spec.tray_efficiency, 1e-6)
        feed_tray = int(0.45 * theoretical)

        P_top = self.spec.top_pressure_kpa
        total_dp = actual * self.spec.pressure_drop_per_tray_kpa
        P_bottom = P_top + total_dp

        # crude temperature bracket from key components
        T_top = bubble_point_binary_T_C(xD_lk, P_top, lk, hk)
        T_bottom = bubble_point_binary_T_C(xB_lk, P_bottom, lk, hk)

        # distribute non-keys by relative volatility heuristic
        xD = {}
        xB = {}
        remaining_D = 1.0
        remaining_B = 1.0
        xD[key_lk] = xD_lk
        xB[key_lk] = xB_lk
        remaining_D -= xD_lk
        remaining_B -= xB_lk

        others = [c for c in self.components if c.name != key_lk]
        if others:
            z_other = sum(z.get(c.name, 0) for c in others)
            for c in others:
                frac = (z.get(c.name, 0) / z_other) if z_other > 0 else 1 / len(others)
                xD[c.name] = max(1e-8, remaining_D * frac * (1.2 if (c.relative_volatility_ref or 1) > 1 else 0.8))
                xB[c.name] = max(1e-8, remaining_B * frac * (0.8 if (c.relative_volatility_ref or 1) > 1 else 1.2))

            # renormalize
            sD = sum(xD.values())
            sB = sum(xB.values())
            xD = {k: v / sD for k, v in xD.items()}
            xB = {k: v / sB for k, v in xB.items()}

        reflux_flow = R * D

        cp_avg = sum(c.cp_liq_kj_per_kmolK for c in self.components) / len(self.components)
        latent_avg = sum(c.latent_heat_kj_per_kmol for c in self.components) / len(self.components)
        Q_reboiler = B * latent_avg + B * cp_avg * max(T_bottom - self.feed.temperature_C, 0)
        steam_flow = Q_reboiler / self.spec.steam_latent_heat_kj_per_kg

        n_rect = max(1, int(feed_tray - 1))
        n_strip = max(1, int(theoretical - n_rect))
        rect_temps = [T_top + i * (((T_top + T_bottom) / 2 - T_top) / max(n_rect - 1, 1)) for i in range(n_rect)]
        strip_temps = [((T_top + T_bottom) / 2) + i * ((T_bottom - (T_top + T_bottom) / 2) / max(n_strip - 1, 1)) for i in range(n_strip)]

        rect_press = [P_top + i * self.spec.pressure_drop_per_tray_kpa for i in range(n_rect)]
        strip_press_start = rect_press[-1] + self.spec.pressure_drop_per_tray_kpa
        strip_press = [strip_press_start + i * self.spec.pressure_drop_per_tray_kpa for i in range(n_strip)]

        return SimulationResult(
            mode="multicomponent_FUG_shortcut",
            theoretical_trays=theoretical,
            actual_trays=actual,
            feed_tray=feed_tray,
            total_pressure_drop_kpa=total_dp,
            top_temperature_C=T_top,
            bottom_temperature_C=T_bottom,
            total_temperature_difference_C=T_bottom - T_top,
            distillate_flow_kmol_h=D,
            bottoms_flow_kmol_h=B,
            distillate_composition=xD,
            bottoms_composition=xB,
            reflux_flow_kmol_h=reflux_flow,
            reflux_temperature_C=T_top - self.spec.reflux_subcooling_C,
            product_draw_kmol_h=D,
            rectifying_tray_temps_C=rect_temps,
            stripping_tray_temps_C=strip_temps,
            rectifying_tray_pressures_kpa=rect_press,
            stripping_tray_pressures_kpa=strip_press,
            reboiler_duty_kj_h=Q_reboiler,
            steam_flow_kg_h=steam_flow,
            reboiler_inlet_temp_C=T_bottom - 10,
            reboiler_outlet_temp_C=T_bottom + 5,
            minimum_reflux_ratio=Rmin,
            minimum_stages=Nmin,
        )

    def _plot_mccabe(
        self,
        alpha: float,
        xD: float,
        xB: float,
        zF: float,
        R: float,
        q: float,
        x_int: float,
        xs: List[float],
        ys: List[float],
        path: str,
    ):
        if plt is None:
            print("matplotlib not installed, skipping McCabe-Thiele plot.")
            return
        x = [i / 499 for i in range(500)]
        y_eq = [y_equilibrium_binary(v, alpha) for v in x]
        y_diag = x
        y_rect = [(R / (R + 1)) * v + xD / (R + 1) for v in x]

        if abs(q - 1.0) < 1e-6:
            y_q = [math.nan for _ in x]
        else:
            y_q = [(q / (q - 1)) * v - zF / (q - 1) for v in x]

        plt.figure(figsize=(8, 8))
        plt.plot(x, y_eq, label="Equilibrium curve")
        plt.plot(x, y_diag, "k--", label="y=x")
        plt.plot(x, y_rect, label="Rectifying line")
        if abs(q - 1.0) >= 1e-6:
            plt.plot(x, y_q, label="q-line")
        else:
            plt.axvline(zF, linestyle=":", label="q-line (vertical)")

        # stripping line from x_int to xB
        y_int = (R / (R + 1)) * x_int + xD / (R + 1)
        m_s = (y_int - xB) / (x_int - xB)
        b_s = xB - m_s * xB
        y_strip = [m_s * v + b_s for v in x]
        plt.plot(x, y_strip, label="Stripping line")

        plt.plot(xs, ys, "r-", linewidth=1.5, label="Stage stepping")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("x (liquid mole fraction of light key)")
        plt.ylabel("y (vapor mole fraction of light key)")
        plt.title("McCabe-Thiele Diagram")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


def print_result(r: SimulationResult):
    print("\n===== DISTILLATION SIMULATION RESULTS =====")
    print(f"Mode: {r.mode}")
    print(f"Theoretical trays: {r.theoretical_trays:.2f}")
    print(f"Actual trays: {r.actual_trays:.2f}")
    print(f"Feed tray: {r.feed_tray}")
    print(f"Top temperature (C): {r.top_temperature_C:.2f}")
    print(f"Bottom temperature (C): {r.bottom_temperature_C:.2f}")
    print(f"Column ΔT (C): {r.total_temperature_difference_C:.2f}")
    print(f"Total pressure drop (kPa): {r.total_pressure_drop_kpa:.2f}")

    print("\n--- Top Distillate ---")
    print(f"Distillate flow (kmol/h): {r.distillate_flow_kmol_h:.2f}")
    print(f"Distillate composition: {r.distillate_composition}")
    print(f"Reflux flow (kmol/h): {r.reflux_flow_kmol_h:.2f}")
    print(f"Reflux temperature (C): {r.reflux_temperature_C:.2f}")
    print(f"Product draw flow (kmol/h): {r.product_draw_kmol_h:.2f}")

    print("\n--- Bottom Product ---")
    print(f"Bottoms flow (kmol/h): {r.bottoms_flow_kmol_h:.2f}")
    print(f"Bottoms composition: {r.bottoms_composition}")

    print("\n--- Reboiler ---")
    print(f"Reboiler duty (kJ/h): {r.reboiler_duty_kj_h:,.2f}")
    print(f"Steam flow required (kg/h): {r.steam_flow_kg_h:,.2f}")
    print(f"Reboiler inlet temperature (C): {r.reboiler_inlet_temp_C:.2f}")
    print(f"Reboiler outlet temperature (C): {r.reboiler_outlet_temp_C:.2f}")

    if r.minimum_reflux_ratio is not None:
        print("\n--- Shortcut References ---")
        print(f"Minimum reflux ratio Rmin: {r.minimum_reflux_ratio:.3f}")
        print(f"Minimum stages Nmin: {r.minimum_stages:.3f}")


if __name__ == "__main__":
    # Example binary system: Benzene / Toluene
    comps = [
        Component("Benzene", 78.11, 136.0, 30700.0, 6.90565, 1211.033, 220.79, 2.4),
        Component("Toluene", 92.14, 158.0, 33300.0, 6.95464, 1344.8, 219.48, 1.0),
    ]

    feed = FeedCondition(
        pressure_kpa=101.3,
        temperature_C=90.0,
        flow_kmol_h=100.0,
        mole_fractions={"Benzene": 0.50, "Toluene": 0.50},
        q_value=1.0,
    )

    spec = ColumnSpec(
        distillate_rate_kmol_h=45.0,
        distillate_key_fraction=0.95,
        bottoms_key_fraction=0.05,
        reflux_ratio=2.0,
        tray_efficiency=0.72,
        pressure_drop_per_tray_kpa=0.6,
        top_pressure_kpa=101.3,
    )

    sim = DistillationColumnSimulator(comps, feed, spec)
    result = sim.run(make_plot=True, plot_path="mccabe_thiele.png")
    print_result(result)
    print("\nMcCabe-Thiele plot saved to: mccabe_thiele.png")
