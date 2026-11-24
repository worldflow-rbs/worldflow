# Blue block

This synthetic tabletop scene is the bundled inference example. Its image and query points were generated together for this repository. The `sim7` source selects the simulated-scene conditioning and scale.

`frame.png` is a 480 × 288 RGB image. The 64 queries in `queries.npz` sample the visible top and front faces of the blue block. Each row of `points_xyz` (float32, meters) matches a row of `pixels_uv` (integer column and row, zero-based, measured on the original image).

The camera axes point right, down, and forward. With focal lengths of 400 pixels and principal point (239.5, 143.5), each 3D point projects to its listed pixel after rounding.
