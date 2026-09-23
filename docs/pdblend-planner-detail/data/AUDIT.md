# Planner figure: CPU synthetic worked example

This is a calculation from the repository synthetic profile, not a GPU measurement.
8 GPUs, TP=PP=1; min_M=2 (explicit figure configuration); parked=L1; H=60 s; margin=5%.
Arrival rate 12 req/s; input lengths 256/512/2048/4096 equally likely (mean 1728, p95 4096); output 128 initially, then 256.
SLO: TTFT 1 s, TPOT 20 ms; safety 0.85 -> feasibility thresholds 850 ms / 17 ms.

| Output | Candidate | P/D/M/L1 | tau | Power W | TTFT ms | TPOT ms | M miss % | Feasible |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 128 | A | 0/0/5/3 | 0 | 772.137190 | 214.189940 | 16.027939 | 0.328474 | True |
| 128 | B | 0/0/4/4 | 0 | 713.551714 | 218.068582 | 17.499742 | 3.805764 | False |
| 128 | C | 2/1/2/3 | 1024 | 711.279577 | 319.169833 | 15.537256 | 1.308914 | True |
| 128 | D | 2/2/2/2 | 1024 | 768.204268 | 317.166275 | 13.775466 | 1.308914 | True |
| 128 | E_tau4096 | 2/1/2/3 | 4096 | 721.730960 | 312.213810 | 17.302583 | 42.424298 | False |
| 256 | A | 0/0/5/3 | 0 | 822.620969 | 216.112458 | 17.950456 | 2.118228 | False |
| 256 | B | 0/0/4/4 | 0 | 770.181474 | 221.083790 | 20.514950 | 48.978096 | False |
| 256 | C | 2/1/2/3 | 1024 | 780.836642 | 326.066822 | 22.434244 | 5.672893 | False |
| 256 | D | 2/2/2/2 | 1024 | 819.755515 | 319.199604 | 15.567026 | 5.672893 | True |
| 256 | E_tau4096 | 2/1/2/3 | 4096 | 785.384540 | 314.556839 | 21.649889 | 99.987013 | False |

Full enumeration chooses C initially and D after output length doubles.
Initial feasible candidates: 158; changed feasible candidates: 55.

## Initial A -> C

Esw=34 J; E_A=46328.231396 J; E_C including switch=42710.774600 J.
Saving=3617.456796 J; gain=7.808321% > 5%: switch.
Additional unmodeled energy must be strictly below 1301.045226 J/60s (21.684087 W average; 3.614015 J per PD request) for gain to remain strictly >5%.
This is an algebraic sensitivity bound. No separately measured incremental KV-energy term exists in this synthetic planner score.

## C -> D after output doubles

Re-evaluated C: 780.836642 W, 22.434244 ms TPOT >17 ms target (also >20 ms SLO).
D: 819.755515 W, 15.567026 ms TPOT; Esw=3.4 J.
Current C horizon=46850.198491 J; D with switch=49188.730908 J; gain=-4.991510%.
C is infeasible, so planner changes to D before checking the energy-saving hysteresis. Extra D capacity is required even though power rises.

## Appendix: min_M=0 audit

Removing the M-floor changes the full optimum: initially pure P2/D2/M0/L1×4 at 663.121615 W; after output doubles pure P2/D3/M0/L1×3 at 777.544566 W. Thus the min_M=2 constraint must be visible in the figure.

All assertions passed. See example.json for exact intermediate pool power, utilization, batch, transfer arithmetic, and source hashes.
