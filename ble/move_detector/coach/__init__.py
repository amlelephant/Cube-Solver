"""coach/ — the analytics layers over a decoded, timestamped move list.

Layer map is LAUNCH_ROADMAP.md 3.2:
  L1 timing   (timing.py)   onset timestamps only; survives evening light
  L2 phases   (phases.py)   replay from the known scramble; not yet built
  L3 cases/algorithms       needs L2 + the user library; not yet built
  L4 insights               rules over L1-L3 aggregates; not yet built

Everything here is a pure function over plain data so the same code runs
in the offline harness and in the server worker.
"""
