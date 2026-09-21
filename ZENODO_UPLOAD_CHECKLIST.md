# Zenodo v1.0.0 publication checklist

This file is operational only; it can be removed after the Zenodo record is published.

1. Create a **new** Zenodo deposit (do not create a new version of the earlier apparent-criticality record).
2. Upload the GitHub release archive `instrumental-spillover-networks-v1.0.0.zip`.
3. Use the metadata in `.zenodo.json` / `ZENODO_METADATA.md`.
4. Resource type: **Software**.
5. Version: **1.0.0**.
6. Access: **Open**.
7. Primary license: **MIT**. The repository additionally states CC BY 4.0 for documentation/frozen derived outputs.
8. Add the GitHub repository as a related identifier:
   `https://github.com/mauricio-herrera/instrumental-spillover-networks`.
9. Treat `https://github.com/mauricio-herrera/apparent-criticality-cryptocon` only as **Related work**, not as a previous version.
10. Reserve/mint the DOI and publish the record.
11. Insert the minted DOI into:
    - `manuscript/paper2_jbes.tex` — Data and code availability;
    - `CITATION.cff`;
    - README citation section.
12. Recompile the manuscript and verify the DOI link before journal submission.
