"""Data audit and feasibility gate for false-discovery-controlled HER screening.

Inputs:
  mamun2019_H_reactions.csv  direct Catalysis-Hub GraphQL export of pubId MamunHighT2019,
                             reactants H2gas, products Hstar (8,856 records)
  hydrogen-unrelaxed.db      initial structures with structure-type metadata (ASE sqlite),
                             used only to cross-check the structure-type rule

Outputs: audit_surfaces.csv (one row per surface) and audit_summary.json.
Usage:   python3 -I audit_mamun_h.py <export.csv> <hydrogen-unrelaxed.db> <outdir>
"""
import ast
import collections as C
import csv
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

DG_CORR = 0.24          # eV, Delta ZPE - T Delta S for H*, Norskov et al. 2005
DELTAS = (0.10, 0.15, 0.20, 0.30)


def per_h_energy(rec):
    e = float(rec["reaction_energy"])
    if rec["equation"] == "H2(g) + 2* -> 2H*":
        return e / 2.0
    if rec["equation"] == "0.5H2(g) + * -> H*":
        return e
    raise ValueError(rec["equation"])


def elements(formula):
    return sorted(set(re.findall(r"[A-Z][a-z]?", formula)))


def main(h_csv, udb, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10 ** 9)
    raw = list(csv.DictReader(open(h_csv)))
    rows, no_structure = [], 0
    for r in raw:
        systems = {x["name"]: x["aseId"] for x in json.loads(r["reactionSystems"])}
        if "Hstar" not in systems:          # no adsorbate structure stored: excluded
            no_structure += 1
            continue
        rows.append({"pub_id": "MamunHighT2019", "dft_functional": r["dftFunctional"],
                     "dft_code": r["dftCode"], "equation": r["Equation"], "coverages": r["coverages"],
                     "reaction_energy": r["reactionEnergy"], "hstar_id": systems["Hstar"],
                     "chemical_composition": r["chemicalComposition"],
                     "surface_composition": r["surfaceComposition"], "facet": r["facet"],
                     "sites": r["sites"]})
    audit = {"n_records": len(raw), "n_records_without_adsorbate_structure": no_structure,
             "n_records_used": len(rows)}
    audit["pub_id"] = dict(C.Counter(r["pub_id"] for r in rows))
    audit["functional"] = dict(C.Counter(r["dft_functional"] for r in rows))
    audit["code"] = dict(C.Counter(r["dft_code"] for r in rows))
    audit["equations"] = dict(C.Counter(r["equation"] for r in rows))
    audit["coverages"] = dict(C.Counter(r["coverages"] for r in rows))

    # de-duplicate on the relaxed adsorbate-structure id (last atoms id)
    uniq, dup_check = {}, []
    for r in rows:
        key = r["hstar_id"]
        e = per_h_energy(r)
        if key in uniq:
            dup_check.append(abs(uniq[key]["E"] - e))
            continue
        uniq[key] = {"surface": r["chemical_composition"], "slab": r["surface_composition"],
                     "facet": r["facet"], "site": json.loads(r["sites"])["H"], "E": e}
    audit["n_unique_configurations"] = len(uniq)
    audit["n_duplicates_removed"] = len(rows) - len(uniq)
    audit["max_abs_diff_duplicates_eV"] = max(dup_check) if dup_check else None

    # structure-type metadata from the unrelaxed database (read only)
    con = sqlite3.connect(f"file:{udb}?mode=ro", uri=True)
    meta = C.defaultdict(C.Counter)
    for (kvp,) in con.execute("select key_value_pairs from systems"):
        d = json.loads(kvp)
        meta[d["slab_name"]][(d.get("SB_symbol"), d.get("Pearson_symbol"), d.get("metalA"), d.get("metalB"))] += 1

    surf = C.defaultdict(list)
    for v in uniq.values():
        surf[(v["surface"], v["slab"], v["facet"])].append(v)

    table, n_meta_conflict, n_meta_missing, n_rule_mismatch = [], 0, 0, 0
    for (chem, slab, facet), confs in sorted(surf.items()):
        m = meta.get(slab)
        if not m:
            n_meta_missing += 1
            sb = pearson = mA = mB = None
        else:
            if len(m) > 1:
                n_meta_conflict += 1
            sb, pearson, mA, mB = m.most_common(1)[0][0]
        rule = "A1" if len(elements(slab)) == 1 else ("L12" if "3" in slab else "L10")
        if sb != rule:
            n_rule_mismatch += 1
        emin = min(c["E"] for c in confs)
        dg = emin + DG_CORR
        els = elements(slab)
        table.append({"surface": slab, "formula": chem, "facet": facet, "structure": sb,
                      "pearson": pearson, "metalA": mA, "metalB": mB if mB else "",
                      "system": "-".join(els), "n_sites": len(confs),
                      "n_unlabelled_sites": sum(1 for c in confs if "|" not in c["site"]),
                      "Emin_eV": round(emin, 6), "dGH_eV": round(dg, 6)})
    audit["n_surfaces"] = len(table)
    audit["n_surfaces_no_metadata"] = n_meta_missing
    audit["n_surfaces_metadata_conflict"] = n_meta_conflict
    audit["n_structure_rule_mismatch"] = n_rule_mismatch
    audit["structure_counts"] = dict(C.Counter(t["structure"] for t in table))
    audit["structure_by_facet"] = {f"{k[0]}|{k[1]}": v for k, v in
                                   C.Counter((t["structure"], t["facet"]) for t in table).items()}
    audit["n_chemical_systems"] = len({t["system"] for t in table})
    audit["n_elements"] = len({e for t in table for e in t["system"].split("-")})
    audit["sites_per_surface"] = dict(sorted(C.Counter(t["n_sites"] for t in table).items()))
    dgs = sorted(t["dGH_eV"] for t in table)
    q = lambda p: dgs[min(len(dgs) - 1, int(p * len(dgs)))]
    audit["dGH_quantiles_eV"] = {"min": dgs[0], "q10": q(.1), "q25": q(.25), "median": q(.5),
                                 "q75": q(.75), "q90": q(.9), "max": dgs[-1]}

    # feasibility: prevalence of in-window surfaces, overall and by structure type
    prev = {}
    for d in DELTAS:
        hit = [t for t in table if abs(t["dGH_eV"]) <= d]
        prev[f"{d:.2f}"] = {"n_hits": len(hit), "prevalence": round(len(hit) / len(table), 4),
                            "by_structure": dict(C.Counter(t["structure"] for t in hit)),
                            "n_systems_with_hit": len({t["system"] for t in hit})}
    audit["window"] = prev

    # attainability: conformal p-values are bounded below by 1/(n_cal+1); BH at level a
    # with m candidates can select k units only if 1/(n_cal+1) <= a*k/m, i.e.
    # k >= k_min = ceil(m / (a*(n_cal+1))). Compare k_min with the expected number of
    # in-window candidates in the pool (prevalence * m). Pools are the held-out structure
    # types; calibration sizes are the planned 30 % of the remaining surfaces.
    gate = {}
    struct_n = C.Counter(t["structure"] for t in table)
    for held in ("L10", "L12"):
        m = struct_n[held]
        n_cal = int(0.3 * (len(table) - m))
        for d in DELTAS:
            hits_pool = sum(1 for t in table if t["structure"] == held and abs(t["dGH_eV"]) <= d)
            for a in (0.1, 0.2, 0.3):
                kmin = math.ceil(m / (a * (n_cal + 1)))
                gate[f"pool={held},delta={d:.2f},alpha={a}"] = {
                    "m": m, "n_cal": n_cal, "k_min": kmin, "hits_in_pool": hits_pool,
                    "attainable": hits_pool >= kmin}
    audit["attainability_gate"] = gate
    audit["gate_all_attainable"] = all(v["attainable"] for v in gate.values())

    with open(out / "audit_surfaces.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    with open(out / "audit_summary.json", "w") as fh:
        json.dump(audit, fh, indent=1)
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
