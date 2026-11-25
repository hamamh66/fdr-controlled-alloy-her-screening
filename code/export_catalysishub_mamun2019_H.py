"""Export MamunHighT2019 hydrogen-adsorption records from Catalysis-Hub (author: H. Hamam).

Requires a personal Catalysis-Hub API key (https://api-catalysis-hub.slac.stanford.edu/auth/login),
entered at a hidden prompt; the key is never printed or written to disk.
Pagination is ordered by reaction id; the script stops if the number of distinct
records differs from the total reported by the API.
Usage: python3 export_catalysishub_mamun2019_H.py <output_csv>
"""
import json, sys, time, getpass, requests, pandas as pd

API = "https://api-catalysis-hub.slac.stanford.edu/graphql"
Q = """query($after: String) {
  reactions(pubId: "MamunHighT2019", reactants: "H2gas", products: "Hstar",
            order: "id", first: 200, after: $after) {
    totalCount
    pageInfo { hasNextPage endCursor }
    edges { node { id Equation sites coverages reactants products facet
                   reactionEnergy dftCode dftFunctional
                   surfaceComposition chemicalComposition
                   reactionSystems { name aseId } } }
  }
}"""


def main(out_csv):
    key = getpass.getpass("Catalysis-Hub API key (hidden): ")
    rows, seen, after, page, total = [], set(), "", 0, None
    while True:
        for attempt in range(5):
            r = requests.post(API, json={"query": Q, "variables": {"after": after}},
                              headers={"X-API-Key": key}, timeout=60)
            if r.status_code == 200 and "errors" not in r.json():
                break
            print(f"  retry {attempt+1}: HTTP {r.status_code} {r.text[:150]}")
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError("Export stopped: repeated API errors.")
        data = r.json()["data"]["reactions"]
        total = data["totalCount"]
        new = 0
        for e in data["edges"]:
            n = e["node"]
            if n["id"] in seen:
                continue
            seen.add(n["id"]); new += 1
            n["reactionSystems"] = json.dumps(n["reactionSystems"])
            rows.append(n)
        page += 1
        print(f"page {page}: +{new} new, {len(rows)} / {total} distinct records")
        if not data["pageInfo"]["hasNextPage"]:
            break
        after = data["pageInfo"]["endCursor"]
    if len(rows) != total:
        raise RuntimeError(f"Incomplete export: {len(rows)} distinct of {total}.")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} distinct records to {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1])
