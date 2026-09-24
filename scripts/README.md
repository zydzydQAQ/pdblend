# Maintenance entrypoints

- `profile/`: profile merge, publication and workload demand audit.
- `campaign/`: GPU lease queue and campaign preparation.
- `results/`: result export, compact storage and evidence-aware cleanup.

Root dated entrypoints still referenced by frozen or pending jobs remain available. A stable entrance may delegate to one of those files until its final live reference is retired. Do not remove dated files solely on their date.
