# GPU-Accelerated Satellite Simulation: evidence register

| Evidence | Origin | Strength |
| :-- | :-- | :-- |
| Project purpose, stated status and technology list | User-supplied project list, 17 November 2026 | Owner statement; no new execution |
| Repository file inventory | [Ace-Alch/GPU-Accelerated-Satellite-Sim](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim), main | Paths and blob identifiers inspected |
| Previous README | `README-before-rebuild.md` | Historical documentation; claims require the listed checks |

## Evidence limits

The inspected CPU source calculates gravity from one black-hole center. It does not sum gravity between every satellite pair. The physics work scales with satellite count and the graphics work includes pixel-by-satellite loops. The previous generic all-pairs O(N²) description does not describe this source. No speedup value is asserted here without its run conditions.

## Missing material

- Raw benchmark timings
- Exact build and hardware settings
- Per-author contribution
- Repeatable speedup calculation
