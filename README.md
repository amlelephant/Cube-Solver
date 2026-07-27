# Cube Solver

A computer-vision and machine-learning pipeline for verifying a Rubik's
Cube solve from webcam video alone — no smart cube required. It locates
the cube, reads its state, and checks the result against the group
theory of the cube.

Two paths, both webcam-based:

- **Scan verification** (`cv/`) — scan the cube before and after a solve;
  detection (YOLOv8), color classification (CNN + HSV ensemble), and a
  group-theory solve check confirm the end state is legitimately reachable
  from the scanned scramble.
- **Move-by-move verification** (`ble/move_detector/`) — detect and
  classify individual moves from video during the solve itself, then
  reconstruct the most likely true move sequence with a group-theoretic
  beam search against the known start and end states.

A full writeup is in progress.
