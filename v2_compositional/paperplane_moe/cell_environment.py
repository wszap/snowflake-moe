# -*- coding: utf-8 -*-
"""Explicit, unit-bearing contract for a future cell environment model.

This module is deliberately independent from the neural forward pass.  It does
not turn optimizer statistics into concentrations and it does not claim to be a
whole-cell simulator.  Its first job is narrower: make omitted physical and
chemical state impossible to hide behind organelle names.

Conventions
-----------
* concentration: millimolar (mM)
* compartment volume: femtolitre (fL)
* membrane potential: millivolt (mV), inside relative to outside
* temperature: kelvin (K)
* pressure: pascal (Pa)
* amount returned by :meth:`ChemicalCompartment.amount_fmol`: femtomole

For an ideal dilute solution, 1 mM contributes approximately 1 mOsm/L per
independent particle.  ``ideal_osmolarity_mOsm_L`` is therefore a bookkeeping
approximation, not an activity-coefficient or Donnan-equilibrium calculation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence, Tuple


ENVIRONMENT_AXES: Tuple[str, ...] = (
    "boundary_and_geometry",
    "temperature",
    "ph",
    "ionic_composition",
    "osmotic_state",
    "membrane_potential",
    "nutrient_sources",
    "electron_acceptor",
    "redox_state",
    "energy_currency",
    "waste_products",
    "signals",
    "mechanics",
    "spatial_transport",
    "time",
    "stochasticity",
)

VALID_ROLES = {
    "ion", "buffer", "nutrient", "electron_acceptor", "redox",
    "energy", "waste", "signal", "solvent", "other",
}


def _finite_nonnegative(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) >= 0.0


@dataclass
class ChemicalCompartment:
    """Finite, well-mixed chemical compartment with explicit units.

    ``charge`` is the signed ionic charge number used only for charge-density
    bookkeeping.  Real protonation states depend on pH; callers must supply the
    charge convention used by their model rather than relying on hidden values.
    """

    name: str
    volume_fL: float
    concentrations_mM: Dict[str, float]
    charge: Dict[str, float]
    roles: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    fixed_charge_mM: float = 0.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("compartment name is empty")
        if not (math.isfinite(self.volume_fL) and self.volume_fL > 0.0):
            errors.append(f"{self.name}: volume_fL must be finite and > 0")
        for species, concentration in self.concentrations_mM.items():
            if not _finite_nonnegative(concentration):
                errors.append(f"{self.name}/{species}: concentration must be finite and >= 0")
            if species not in self.charge:
                errors.append(f"{self.name}/{species}: explicit charge is missing")
            unknown = set(self.roles.get(species, ())) - VALID_ROLES
            if unknown:
                errors.append(f"{self.name}/{species}: unknown roles {sorted(unknown)}")
        extra_charge = set(self.charge) - set(self.concentrations_mM)
        if extra_charge:
            errors.append(f"{self.name}: charge supplied for absent species {sorted(extra_charge)}")
        return errors

    def amount_fmol(self, species: str) -> float:
        """Return amount in fmol; 1 mM * 1 fL = 1e-3 fmol."""
        return float(self.concentrations_mM.get(species, 0.0)) * self.volume_fL * 1e-3

    def ideal_osmolarity_mOsm_L(self) -> float:
        return sum(float(v) for v in self.concentrations_mM.values())

    def net_charge_density_mEq_L(self) -> float:
        mobile = sum(
            float(self.charge[name]) * float(concentration)
            for name, concentration in self.concentrations_mM.items()
        )
        return mobile + float(self.fixed_charge_mM)

    def has_role(self, role: str) -> bool:
        return any(role in self.roles.get(name, ()) for name in self.concentrations_mM)


@dataclass
class CellEnvironment:
    """Minimum state required before calling a model a chemical cell environment."""

    extracellular: ChemicalCompartment
    intracellular: ChemicalCompartment
    temperature_K: float
    extracellular_pH: float
    intracellular_pH: float
    membrane_potential_mV: float
    redox_ratio: float
    mechanical_pressure_Pa: float
    time_s: float
    boundary_model: str
    electron_acceptor: str
    signal_model: str
    stochasticity_model: str
    spatial_model: str
    organism_profile: str = "abstract"
    light_umol_m2_s: float = 0.0

    def audit(self) -> dict:
        errors = self.extracellular.validate() + self.intracellular.validate()
        ext, cyt = self.extracellular, self.intracellular
        all_roles = {
            role
            for compartment in (ext, cyt)
            for roles in compartment.roles.values()
            for role in roles
        }
        species = set(ext.concentrations_mM) | set(cyt.concentrations_mM)

        axes = {
            "boundary_and_geometry": bool(self.boundary_model.strip())
                and ext.volume_fL > 0.0 and cyt.volume_fL > 0.0,
            "temperature": math.isfinite(self.temperature_K) and self.temperature_K > 0.0,
            "ph": all(math.isfinite(v) and 0.0 <= v <= 14.0
                      for v in (self.extracellular_pH, self.intracellular_pH)),
            "ionic_composition": "ion" in all_roles,
            "osmotic_state": all(
                math.isfinite(compartment.ideal_osmolarity_mOsm_L())
                and compartment.ideal_osmolarity_mOsm_L() > 0.0
                for compartment in (ext, cyt)),
            "membrane_potential": math.isfinite(self.membrane_potential_mV),
            "nutrient_sources": ext.has_role("nutrient"),
            "electron_acceptor": bool(self.electron_acceptor.strip())
                and self.electron_acceptor in species,
            "redox_state": math.isfinite(self.redox_ratio) and self.redox_ratio > 0.0
                and "redox" in all_roles,
            "energy_currency": {"ATP", "ADP", "Pi"}.issubset(cyt.concentrations_mM)
                and all("energy" in cyt.roles.get(name, ()) for name in ("ATP", "ADP", "Pi")),
            "waste_products": ext.has_role("waste") or cyt.has_role("waste"),
            "signals": bool(self.signal_model.strip()) and "signal" in all_roles,
            "mechanics": _finite_nonnegative(self.mechanical_pressure_Pa),
            "spatial_transport": bool(self.spatial_model.strip()),
            "time": _finite_nonnegative(self.time_s),
            "stochasticity": bool(self.stochasticity_model.strip()),
        }
        missing = [axis for axis in ENVIRONMENT_AXES if not axes[axis]]
        if missing:
            errors.append("missing environment axes: " + ", ".join(missing))

        osmotic_gap = cyt.ideal_osmolarity_mOsm_L() - ext.ideal_osmolarity_mOsm_L()
        adenylates = [cyt.concentrations_mM.get(name, 0.0) for name in ("ATP", "ADP", "AMP")]
        adenylate_total = sum(adenylates)
        energy_charge = ((adenylates[0] + 0.5 * adenylates[1]) / adenylate_total
                         if adenylate_total > 0.0 else None)

        warnings: list[str] = []
        if abs(ext.net_charge_density_mEq_L()) > 1.0:
            warnings.append("extracellular net charge exceeds 1 mEq/L; counter-ions may be missing")
        if abs(cyt.net_charge_density_mEq_L()) > 1.0:
            warnings.append("intracellular net charge exceeds 1 mEq/L; fixed charge/counter-ions may be missing")
        if abs(osmotic_gap) > 50.0:
            warnings.append("ideal osmolarity gap exceeds 50 mOsm/L; volume regulation is not represented")
        if self.organism_profile == "photoautotroph" and self.light_umol_m2_s <= 0.0:
            errors.append("photoautotroph profile requires positive light flux")

        return {
            "complete": not errors,
            "axes": axes,
            "missing": missing,
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "external_ideal_osmolarity_mOsm_L": ext.ideal_osmolarity_mOsm_L(),
                "internal_ideal_osmolarity_mOsm_L": cyt.ideal_osmolarity_mOsm_L(),
                "osmotic_gap_mOsm_L": osmotic_gap,
                "external_net_charge_mEq_L": ext.net_charge_density_mEq_L(),
                "internal_net_charge_mEq_L": cyt.net_charge_density_mEq_L(),
                "adenylate_energy_charge": energy_charge,
            },
        }


def audit_taxonomic_coherence(components: Iterable[str], profile: str) -> dict:
    """Check obvious cross-kingdom category errors without judging ML utility.

    ``abstract`` permits all components but only supports an analogy-level claim.
    Other profiles intentionally enforce conservative textbook boundaries.
    """

    components = {str(c).strip().lower() for c in components}
    allowed_profiles = {"abstract", "bacterium", "animal", "plant"}
    if profile not in allowed_profiles:
        raise ValueError(f"unknown profile {profile!r}; expected {sorted(allowed_profiles)}")

    incompatible: list[str] = []
    warnings: list[str] = []
    if "bloodstream" in components:
        warnings.append("bloodstream is organism/tissue-level transport, not an intracellular organelle")
    if profile == "bacterium":
        for name in ("mitochondrion", "chloroplast", "bloodstream"):
            if name in components:
                incompatible.append(f"bacterium profile cannot contain {name}")
    elif profile == "animal":
        if "chloroplast" in components:
            incompatible.append("animal profile cannot contain chloroplast")
        if "lac_operon" in components:
            incompatible.append("animal profile cannot claim a bacterial lac-operon mechanism")
    elif profile == "plant":
        if "lac_operon" in components:
            incompatible.append("plant profile cannot claim a bacterial lac-operon mechanism")
        if "bloodstream" in components:
            incompatible.append("plant cell profile cannot contain a bloodstream organelle")

    return {
        "profile": profile,
        "components": sorted(components),
        "coherent": not incompatible,
        "claim_level": "analogy_only" if profile == "abstract" else "profile_checked",
        "incompatible": incompatible,
        "warnings": warnings,
    }


def roles(**items: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    """Small helper for readable, immutable role declarations in configs/tests."""
    return {name: tuple(values) for name, values in items.items()}


__all__ = [
    "ENVIRONMENT_AXES", "ChemicalCompartment", "CellEnvironment",
    "audit_taxonomic_coherence", "roles",
]
