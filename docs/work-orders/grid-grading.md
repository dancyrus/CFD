# Work order: graded tensor-product grid

GOAL
The plume domain is too short to read - you can see the Mach disk and
half of one shock cell. Fix it with a graded tensor-product grid, not by
enlarging the uniform grid (that costs ~14x the wall time).

WHY THIS APPROACH
Every shipping interactive fluid tool uses a fixed uniform or statically
stretched grid with geometry as a cheap-to-overwrite mask. Basilisk, the
reference adaptive-quadtree CFD code, shipped its GPU backend with
Cartesian and multigrid only. IDEFIX chose static stretching over AMR
explicitly for cost and complexity. On a uniform or graded grid, "the
user drew a wall" is a write to an array with no structure to
invalidate - that property is what makes the sandbox possible.

1. SOLVER TAKES ARBITRARY CELL EDGES
cfd-core accepts a list of cell edge positions and must not know why
they are spaced that way. Four numerics requirements:

  a. Use the properly weighted non-uniform 3-point stencil. A naive
     non-uniform stencil is formally FIRST order and injects a fraction
     (r-1) of full upwind dissipation - at growth ratio 1.2 that is 20%
     of upwind smearing through the plume. With the weighted stencil
     that term vanishes identically at any ratio.
  b. Slopes need the full unequal-spacing formula INCLUDING the cell's
     own value term. Distance-weighting alone is still only first order.
     minmod and MC need no change on non-uniform grids; van Leer and van
     Albada do. We use minmod, so the limiter function is unchanged.
  c. Do NOT pick a cell-centre radius. Reconstruction wants the volume
     centroid, the p/r balance wants the arithmetic mean of face radii,
     and they differ ~4%. Since (1/r)d(rp)/dr - p/r is identically
     dp/dr, discretize the pressure part as plain gradient minus
     geometric flux. Exact on any radial grid. Ref arXiv 1701.04834.
  d. Free-stream preservation is not structurally at risk here - that
     concern applies to mapped curvilinear grids, not tensor-product
     Cartesian. Still re-run T1 on a deliberately graded grid as the
     guard, plus the full ladder.

2. SPACING RULE LIVES IN THE GEOMETRY LAYER, NEVER IN cfd-core
For each axis independently, find the span where solid exists in the
rasterized field, hold base resolution across it plus a margin, then
grade geometrically beyond. Growth ratio 1.05 as margin, not as a
correctness requirement. Must work on arbitrary drawn geometry - do not
reference nozzle exit planes or exit radii anywhere.

3. PLUME LENGTH CONTROL
Compact / Standard / Long (~40 exit radii, 4-5 shock cells). Default
Standard. Show estimated time to steady state per option.

4. EXPLICITLY NOT DOING
No refinement boxes, no quadtree adaptivity, no sparse allocation.
Record why in the work order: with a global timestep, refining a box
halves dt everywhere for benefit nowhere; sub-stepping fixes that but
breaks positivity at the coarse-fine interface, which is where the Mach
disk sits. Sparse allocation treats inactive regions as vacuum, so
pressure waves cannot cross them - fatal for compressible flow.

5. REPORT
Cells, steps/s and seconds-to-steady before and after, on this machine.

NEXT JOB AFTER THIS (note it in the work order, do not build it)
Cut cells with state redistribution. Worth more than this task: the
staircase wall costs 13% of thrust, that error is FIRST order in cell
size, and cut cells make it second order. The rasterizer already
computes exact sub-cell fractions that the solver discards by
thresholding at 0.5.
