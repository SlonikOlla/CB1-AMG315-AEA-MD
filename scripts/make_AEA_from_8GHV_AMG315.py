from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolDescriptors
from rdkit.Geometry import Point3D
import numpy as np
from pathlib import Path

amg_file = Path(
    "structures/reference_endocannabinoid_CB1/"
    "AMG315_ZI5_aligned_to_frozen_CB1.sdf"
)

aea_file = Path(
    "ligands/endocannabinoids/AEA_generated_min.sdf"
)

out_file = Path(
    "structures/reference_endocannabinoid_CB1/"
    "AEA_from_8GHV_AMG315_aligned_to_frozen_CB1.sdf"
)

# Read heavy-atom molecules
amg = Chem.SDMolSupplier(str(amg_file), removeHs=True)[0]
aea = Chem.SDMolSupplier(str(aea_file), removeHs=True)[0]

if amg is None or aea is None:
    raise RuntimeError("Could not read AMG315 or AEA")

# Find exact common scaffold
mcs = rdFMCS.FindMCS(
    [amg, aea],
    atomCompare=rdFMCS.AtomCompare.CompareElements,
    bondCompare=rdFMCS.BondCompare.CompareOrder,
    ringMatchesRingOnly=True,
    completeRingsOnly=True,
    matchValences=True,
    timeout=60
)

core = Chem.MolFromSmarts(mcs.smartsString)

amg_match = amg.GetSubstructMatch(core)
aea_match = aea.GetSubstructMatch(core)

print("AMG315 heavy atoms:", amg.GetNumHeavyAtoms())
print("AEA heavy atoms:", aea.GetNumHeavyAtoms())
print("Mapped heavy atoms:", len(aea_match))

if len(aea_match) != aea.GetNumHeavyAtoms():
    raise RuntimeError(
        "Not all AEA heavy atoms map onto AMG315"
    )

# Copy AEA chemistry
pose = Chem.Mol(aea)

# Create a new conformer using experimental AMG315 coordinates
conf = Chem.Conformer(pose.GetNumAtoms())
conf.Set3D(True)

amg_conf = amg.GetConformer()

for aea_idx, amg_idx in zip(aea_match, amg_match):
    p = amg_conf.GetAtomPosition(amg_idx)
    conf.SetAtomPosition(
        aea_idx,
        Point3D(p.x, p.y, p.z)
    )

pose.RemoveAllConformers()
pose.AddConformer(conf, assignId=True)

# Add hydrogens to the experimental heavy-atom geometry
poseH = Chem.AddHs(pose, addCoords=True)

Chem.SanitizeMol(poseH)

print("Formula before H optimization:",
      rdMolDescriptors.CalcMolFormula(poseH))
print("Total atoms:", poseH.GetNumAtoms())
print("Heavy atoms:", poseH.GetNumHeavyAtoms())

# Optimize hydrogens ONLY.
# Every heavy atom is fixed at its experimental-template position.
props = AllChem.MMFFGetMoleculeProperties(
    poseH,
    mmffVariant="MMFF94s"
)

if props is None:
    raise RuntimeError("MMFF properties could not be generated")

ff = AllChem.MMFFGetMoleculeForceField(
    poseH,
    props,
    confId=0
)

for atom in poseH.GetAtoms():
    if atom.GetAtomicNum() > 1:
        ff.AddFixedPoint(atom.GetIdx())

ff.Initialize()

e_before = ff.CalcEnergy()
status = ff.Minimize(maxIts=2000)
e_after = ff.CalcEnergy()

print("MMFF H-only minimization status:", status)
print("Energy before:", e_before)
print("Energy after:", e_after)

# Verify that heavy atoms did not move
final_conf = poseH.GetConformer()

sq = []

for aea_idx, amg_idx in zip(aea_match, amg_match):
    pa = final_conf.GetAtomPosition(aea_idx)
    pm = amg_conf.GetAtomPosition(amg_idx)

    d2 = (
        (pa.x - pm.x)**2 +
        (pa.y - pm.y)**2 +
        (pa.z - pm.z)**2
    )
    sq.append(d2)

rmsd = np.sqrt(np.mean(sq))
maxdev = np.sqrt(np.max(sq))

print("Mapped heavy-atom RMSD to AMG315 template:", rmsd)
print("Maximum heavy-atom deviation:", maxdev)

poseH.SetProp(
    "_Name",
    "AEA_from_8GHV_AMG315_aligned_to_frozen_CB1"
)

w = Chem.SDWriter(str(out_file))
w.write(poseH)
w.close()

# Final integrity check
check = Chem.SDMolSupplier(
    str(out_file),
    removeHs=False
)[0]

if check is None:
    raise RuntimeError("Could not re-read final AEA SDF")

print("Final formula:",
      rdMolDescriptors.CalcMolFormula(check))
print("Final heavy atoms:", check.GetNumHeavyAtoms())
print("Final total atoms:", check.GetNumAtoms())
print("Saved:", out_file)
print("Re-read final SDF: OK")
