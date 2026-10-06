from Bio.PDB import PDBParser, Superimposer, PDBIO, Select
import numpy as np
from pathlib import Path

ref_pdb = Path("structures/CB1_final/CB1_WT_9B54_model10_PPM_clean.pdb")
mob_pdb = Path("structures/reference_endocannabinoid_CB1/8GHV.pdb")
out_dir = Path("structures/reference_endocannabinoid_CB1")
out_dir.mkdir(parents=True, exist_ok=True)

parser = PDBParser(QUIET=True)

ref = parser.get_structure("REF", ref_pdb)
mob = parser.get_structure("MOB", mob_pdb)

ref_chain = ref[0]["A"]
mob_chain = mob[0]["D"]

# Use shared CB1 CA atoms only.
# Restrict to residues present in our frozen model, 108-411.
ref_atoms = []
mob_atoms = []
used = []

for resnum in range(108, 412):
    if resnum not in ref_chain or resnum not in mob_chain:
        continue

    rr = ref_chain[resnum]
    mr = mob_chain[resnum]

    if "CA" not in rr or "CA" not in mr:
        continue

    # Require same residue identity
    if rr.get_resname() != mr.get_resname():
        continue

    ref_atoms.append(rr["CA"])
    mob_atoms.append(mr["CA"])
    used.append(resnum)

print("Matched CA atoms:", len(used))
print("Residue range:", min(used), "-", max(used))

sup = Superimposer()
sup.set_atoms(ref_atoms, mob_atoms)

print("CA RMSD before/fit RMSD:", sup.rms)

# Apply transformation to entire mobile structure,
# including AMG315 / ZI5
sup.apply(list(mob.get_atoms()))

class ChainDandZI5(Select):
    def accept_chain(self, chain):
        return 1 if chain.id == "D" else 0

io = PDBIO()
io.set_structure(mob)

aligned_all = out_dir / "8GHV_chainD_aligned_to_frozen_CB1.pdb"
io.save(str(aligned_all), ChainDandZI5())

class ZI5Only(Select):
    def accept_residue(self, residue):
        return 1 if residue.get_resname() == "ZI5" else 0

aligned_zi5 = out_dir / "AMG315_ZI5_aligned_to_frozen_CB1.pdb"
io.save(str(aligned_zi5), ZI5Only())

# quantify aligned receptor CA deviations
dists = []
for resnum in used:
    ref_ca = ref_chain[resnum]["CA"].coord
    mob_ca = mob_chain[resnum]["CA"].coord
    dists.append(np.linalg.norm(ref_ca - mob_ca))

dists = np.array(dists)

print("Post-alignment CA RMSD:", np.sqrt(np.mean(dists**2)))
print("Median CA deviation:", np.median(dists))
print("Max CA deviation:", np.max(dists))
print("Saved:", aligned_all)
print("Saved:", aligned_zi5)
