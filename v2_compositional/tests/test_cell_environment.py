import unittest

from paperplane_moe.cell_environment import (
    ChemicalCompartment,
    CellEnvironment,
    audit_taxonomic_coherence,
    roles,
)


def _fixture():
    ext_c = {
        "Na": 140.0, "K": 5.0, "Cl": 145.0, "glucose": 5.0,
        "O2": 0.2, "lactate": 1.0, "signal": 0.001,
    }
    ext_q = {"Na": 1.0, "K": 1.0, "Cl": -1.0, "glucose": 0.0,
             "O2": 0.0, "lactate": -1.0, "signal": 0.0}
    ext = ChemicalCompartment(
        "medium", 1_000.0, ext_c, ext_q,
        roles=roles(
            Na=("ion",), K=("ion",), Cl=("ion",),
            glucose=("nutrient",), O2=("electron_acceptor",),
            lactate=("waste",), signal=("signal",),
        ),
        fixed_charge_mM=1.0,
    )

    cyt_c = {
        "Na": 10.0, "K": 140.0, "Cl": 100.0, "ATP": 3.0,
        "ADP": 1.0, "AMP": 0.2, "Pi": 2.0, "NADH": 0.1,
        "NAD": 1.0, "glucose": 1.0, "O2": 0.05, "lactate": 2.0,
        "signal": 0.001,
    }
    cyt_q = {
        "Na": 1.0, "K": 1.0, "Cl": -1.0, "ATP": -4.0,
        "ADP": -3.0, "AMP": -2.0, "Pi": -2.0, "NADH": -2.0,
        "NAD": -1.0, "glucose": 0.0, "O2": 0.0, "lactate": -1.0,
        "signal": 0.0,
    }
    mobile_charge = sum(cyt_c[name] * cyt_q[name] for name in cyt_c)
    cyt = ChemicalCompartment(
        "cytosol", 1.0, cyt_c, cyt_q,
        roles=roles(
            Na=("ion",), K=("ion",), Cl=("ion",),
            ATP=("energy",), ADP=("energy",), AMP=("energy",), Pi=("energy",),
            NADH=("redox",), NAD=("redox",), glucose=("nutrient",),
            O2=("electron_acceptor",), lactate=("waste",), signal=("signal",),
        ),
        fixed_charge_mM=-mobile_charge,
    )
    return CellEnvironment(
        extracellular=ext,
        intracellular=cyt,
        temperature_K=310.15,
        extracellular_pH=7.4,
        intracellular_pH=7.2,
        membrane_potential_mV=-70.0,
        redox_ratio=0.1,
        mechanical_pressure_Pa=101_325.0,
        time_s=0.0,
        boundary_model="finite lipid membrane with explicit transporters",
        electron_acceptor="O2",
        signal_model="explicit extracellular ligand",
        stochasticity_model="chemical Langevin",
        spatial_model="well-mixed compartments with explicit membrane flux",
        organism_profile="abstract",
    )


class CellEnvironmentTests(unittest.TestCase):
    def test_complete_contract_covers_all_axes(self):
        report = _fixture().audit()
        self.assertTrue(report["complete"], report)
        self.assertEqual(report["missing"], [])
        self.assertGreater(report["metrics"]["adenylate_energy_charge"], 0.5)

    def test_amount_conversion_has_units(self):
        pool = ChemicalCompartment("x", 2.0, {"A": 100.0}, {"A": 0.0})
        self.assertAlmostEqual(pool.amount_fmol("A"), 0.2)

    def test_missing_energy_is_explicit(self):
        env = _fixture()
        del env.intracellular.concentrations_mM["ATP"]
        del env.intracellular.charge["ATP"]
        report = env.audit()
        self.assertFalse(report["complete"])
        self.assertIn("energy_currency", report["missing"])

    def test_cross_kingdom_mix_is_analogy_only(self):
        mixed = {"bloodstream", "chloroplast", "mitochondrion", "lac_operon"}
        abstract = audit_taxonomic_coherence(mixed, "abstract")
        self.assertTrue(abstract["coherent"])
        self.assertEqual(abstract["claim_level"], "analogy_only")
        animal = audit_taxonomic_coherence(mixed, "animal")
        self.assertFalse(animal["coherent"])
        self.assertTrue(any("chloroplast" in item for item in animal["incompatible"]))


if __name__ == "__main__":
    unittest.main()
