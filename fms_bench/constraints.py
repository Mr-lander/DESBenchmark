"""Energy and carbon accounting for DESBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .distributions import hour_to_tou_price
from .models import Machine


@dataclass
class EnergyModel:
    """Compute energy use and TOU electricity cost over time intervals."""

    peak_window: Tuple[int, int]
    peak_price: float
    offpeak_price: float

    def _tou_cost(self, total_power_kw: float, start_hour: float, end_hour: float) -> float:
        if total_power_kw <= 0.0 or end_hour <= start_hour:
            return 0.0

        cost = 0.0
        t = start_hour
        while t < end_hour:
            next_boundary = min(end_hour, float(int(t) + 1))
            # Guard against numerical corner cases where t is exactly an integer boundary.
            if next_boundary <= t:
                next_boundary = end_hour
            segment_hours = next_boundary - t
            price = hour_to_tou_price(t, self.peak_window, self.peak_price, self.offpeak_price)
            cost += total_power_kw * segment_hours * price
            t = next_boundary
        return cost

    def interval_energy(
        self,
        machines: Mapping[int, Machine],
        start_hour: float,
        end_hour: float,
    ) -> Dict[str, float]:
        """Return interval energy/cost while machine states are constant."""
        if end_hour <= start_hour:
            return {"energy_kwh": 0.0, "energy_cost": 0.0}

        duration = end_hour - start_hour
        total_power_kw = 0.0
        for machine in machines.values():
            if machine.status == "busy":
                total_power_kw += machine.active_power_kw
            elif machine.status == "setup":
                total_power_kw += machine.setup_power_kw
            else:
                total_power_kw += machine.idle_power_kw

        energy_kwh = total_power_kw * duration
        energy_cost = self._tou_cost(total_power_kw, start_hour, end_hour)
        return {"energy_kwh": energy_kwh, "energy_cost": energy_cost}


@dataclass
class CarbonTracker:
    """Track carbon emissions under an optional weekly cap."""

    kg_per_kwh: float
    weekly_cap_tons: Optional[float] = None
    total_kg: float = 0.0

    def reset(self) -> None:
        self.total_kg = 0.0

    def record(self, energy_kwh: float) -> float:
        emitted_kg = max(0.0, energy_kwh) * self.kg_per_kwh
        self.total_kg += emitted_kg
        return emitted_kg

    def status(self) -> Dict[str, float | bool | None]:
        cap_kg = None if self.weekly_cap_tons is None else self.weekly_cap_tons * 1000.0
        remaining = None if cap_kg is None else cap_kg - self.total_kg
        exceeded = False if remaining is None else remaining < 0.0
        return {
            "carbon_total_kg": self.total_kg,
            "carbon_cap_kg": cap_kg,
            "carbon_remaining_kg": remaining,
            "carbon_cap_exceeded": exceeded,
        }
