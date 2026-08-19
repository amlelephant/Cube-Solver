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

## Demo

![Live move-recognition demo: true move vs. model call, with substitutions, misses and spurious calls flagged, ending on the accuracy breakdown](media/demo_solve.gif)

This gif is an excerpt from an entire solve. The stats for that solve are listed at the end of the gif. This solve is about median accuracy for full light, so it is a good representation of how the pipeline performs. The full, uncropped, full-quality clip is at
[`media/demo_solve_verification.mp4`](media/demo_solve_verification.mp4).

Both are of a held-out solve (`ble/move_detector/`) with a live
move-recognition log burned into the frame: the true move at the timestamp
the smart cube reported it, the model's call alongside it, and any spurious
(phantom) call flagged in red. The session (solve_20260805_155829_solve)
was never seen by the evaluated checkpoint and is the median-accuracy
daytime solve among the paper's held-out set.

This is the raw per-move recognition stage, before the group-theoretic
state-reconstruction pass that can still correct some of what's flagged
wrong here. The full writeup (`paper/`) is in progress and will cover the
end-to-end numbers, including how accuracy decays with solve speed and
lighting.
