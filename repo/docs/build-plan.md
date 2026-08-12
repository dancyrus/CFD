# Build Plan — the human's runbook

What you type, in what order, what you check, and what you do when it breaks. The
session briefs live in `docs/sessions/`; you do not paste long prompts, you tell each
session which brief to read.

---

## 1. Schedule, honestly

**Three hours gets you something on screen and moving. It does not get you a
validated nozzle.**

| Milestone | P50 | Confidence |
|---|---|---|
| UI running, geometry editor working, mock plume animating | **2h 15m** | 85% |
| Drawn geometry with real compressible flow moving through it | **3h 10m** | 50% |
| A nozzle plume a propulsion engineer would not laugh at | **3h 50m** | 80% CI 3h05m – 6h30m |

Probability of finishing everything in 2h35m: under 10%.

What naive estimates leave out: cold compiles in fresh worktrees (each is a fresh
checkout with an empty `target/`, and eframe is a 2–5 minute build), egui API churn,
rayon borrow errors, agent context compaction on long sessions, and **you** as a
serial resource — four sessions × roughly eight interrupts each × a minute or two of
reading and pasting is 30–60 minutes of wall clock that is not agent time. They also
budget zero minutes for "it runs but the physics is wrong," which is found by looking,
is serial, and is where the variance lives.

The plan below is built so anything after minute 120 is demoable. That is worth more
than the three-hour number.

---

## 2. Why this decomposition

The solver is the critical path and adding sessions does not shorten it — flux,
reconstruction, limiter, axisymmetric source, wall, inlet, outflow and timestep all
have to be right simultaneously before anything looks like a nozzle. Three moves cut it:

**The coordinator writes HLLC and MUSCL itself.** They are the two most memorized
functions in numerical CFD, take about twelve minutes, and have cheap self-contained
tests. Putting them in the blocking phase converts the seam between the two solver
sessions from an agreement about behaviour into a shared compiled artifact —
contracts by signature drift at merge, contracts by implementation cannot. It also
means that at minute 150, when the plume is wrong, you rule out the flux solver in
ten seconds.

**The coordinator writes the RK2 orchestration.** `cfd-core/src/step.rs` calls
functions in `kernel.rs` and `physics.rs` that do not exist yet. Session A implements
one file, session B the other, neither ever opens `step.rs`. The merge becomes
textually conflict-free and the orchestration is already correct. This single
structural decision turns a 60-minute integration into a 25-minute one.

**The solver splits in two, and both halves test independently.** That is the test
that matters and it passes: B's axisymmetric source has a well-balanced test, its
quasi-1D init validates against the b4wind inversion, its inlet ghost state is
assertable, and its wall needs only `hllc_flux`, which the coordinator already wrote.
B never waits for A.

**The coordinator writes the acceptance tests before the solver exists.** Session A
gets a red-to-green target instead of inventing its own definition of done.

---

## 3. Preflight — the day before, not on the clock

- `rustc --version` prints 1.75 or later. `cargo` and `git` work.
- About 25 GB free disk. Four worktrees, four `target/` directories.
- Five terminal windows you can leave open.

Do **not** try to avoid the cold builds with a shared `CARGO_TARGET_DIR`. Concurrent
cargo invocations take a file lock on the build directory and serialize, which is
exactly what you paid for parallelism to avoid. Instead keep `eframe` in `cfd-ui`
only, so one worktree pays, and start `cargo build` in each worktree the moment it
exists.

---

## 4. The sequence

### T+0 — coordinator

```bash
mkdir -p ~/cfd-poc && cd ~/cfd-poc
git init -b main
# copy this docs/ tree and CLAUDE.md in
printf 'target/\n' > .gitignore
git add -A && git commit -m "plan"
claude
```

Prompt:

> Read `CLAUDE.md`, then `docs/physics-reference.md`, then `docs/contract.md`, then
> `docs/sessions/00-coordinator.md`. Follow the coordinator brief exactly, in order.
> Stop when `cargo test -p cfd-contract` is green.

**Wait about 50 minutes.** Then run `cargo test -p cfd-contract`. If it fails, paste
the entire error back into the session. Do not edit anything yourself.

### T+50 — fan out

```bash
cd ~/cfd-poc
git add -A && git commit -m "frozen contract, kernels, orchestration, mock"
git worktree add ../cfd-A -b core-kernel
git worktree add ../cfd-B -b core-physics
git worktree add ../cfd-C -b geom
git worktree add ../cfd-D -b ui
```

The commit must come first — worktrees share the git object store but not untracked
files. `claude --worktree <name>` would do the `git worktree add` for you and enforces
isolation (it blocks edits targeting the main checkout), but doing it by hand lets you
pre-warm the build and keeps branch names predictable.

Four terminals, warming the compiler while you type:

```bash
cd ~/cfd-A && cargo build -p cfd-core & claude    # terminal 2
cd ~/cfd-B && cargo build -p cfd-core & claude    # terminal 3
cd ~/cfd-C && cargo build -p cfd-geom & claude    # terminal 4
cd ~/cfd-D && cargo build -p cfd-ui   & claude    # terminal 5
```

The prompt for each is one line:

| Terminal | Prompt |
|---|---|
| 2 | `Read CLAUDE.md, docs/physics-reference.md, docs/contract.md, then docs/sessions/A-core-kernel.md. Follow that brief exactly. You are session A.` |
| 3 | `... docs/sessions/B-core-physics.md ... You are session B.` |
| 4 | `... docs/sessions/C-geom.md ... You are session C.` |
| 5 | `... docs/sessions/D-ui.md ... You are session D.` |

### T+50 to T+155 — you wait

Two jobs only:

1. When a session prints `CONTRACT CHANGE REQUEST`, decide. If you approve, paste the
   identical change into **all five** sessions yourself. Never let a session edit the
   contract in its own worktree — that creates a conflict in the one file that must
   never conflict.
2. Every 25 minutes: "one line — what is done, what is left, are you blocked?"

**T+120 checkpoint.** In terminal 5: `cargo run -p cfd-ui`. You should see a moving
mock plume and be able to drag geometry. If you cannot, go to §5 now, not at minute 180.

### T+155 — merge

In each session terminal: "commit everything with `git add -A && git commit -m
'<crate> done'`."

Then in terminal 1, where the coordinator is still alive:

> Read `docs/sessions/E-integration.md` and follow it.

**What you check after each merge:** the last line says `test result: ok`. That is the
whole check. If it says `FAILED` or `error[E0...]`, paste the entire output into the
coordinator session with: "this failed after merging `<branch>`, fix it, do not modify
cfd-contract, and do not fix a failing physics test by loosening a tolerance." If git
says `CONFLICT`, paste that too. Do not resolve anything yourself.

---

## 5. The abort ladder

Every rung is a **flag written at minute 25**, not code written at minute 150. That
is the entire plan. Flags you have to write under pressure are flags you do not get.

**Hard gate at T+150:** `cargo test -p cfd-core -- --ignored t4_sod`.

### If Sod passes but there is no plume

| Rung | Cost | What | Say to the session |
|---|---|---|---|
| 1 | **0 min** | `Geometry::Planar`. Drop the 1/r terms. A planar nozzle with staircase walls still gives a supersonic jet, shock diamonds and a Mach reflection on the centreline. Qualitatively right, quantitatively wrong — thrust is per unit depth. Costs nothing because it is exactly session A's deliverable, already tested. The best insurance in the plan, and free | "Set `Geometry::Planar` and badge the UI 'PLANAR (2D) MODE — thrust is per unit depth, not a rocket.'" |
| 2 | 5 min | `Reconstruction::FirstOrder` + `FluxMode::Hll` + CFL 0.2. Ugly, diffuse, extremely robust. Smeared shock cells beat no shock cells | "Switch to first order, HLL, CFL 0.2, and tell me what the plume does." |
| 3 | 15 min | `WallMode::ColumnReflect`. If the immersed boundary leaks, replace it: per axial column find the open radius and apply a reflecting condition at that one index. You lose arbitrary drawn geometry (only shapes star-convex in r survive) but the nozzle works and wall mass flux is exactly zero | "Implement `WallMode::ColumnReflect` in physics.rs and switch to it." |
| 4 | 10 min | Delete the plume. Domain ends at the exit plane, 256×128, converges in about 1,500 steps — six seconds either machine. A correct, fast, quasi-1D-validated nozzle interior with a real Mach contour. Boring. Alive. Demoable | "Cut the domain at the exit plane and drop to 256×128." |

### If Sod fails at T+150

The kernel is broken and you will not fix it in thirty minutes.

**Rung 0 — `SolverKind::Mock`.** Ship the mock demo: real geometry editor, real
contour math, real rasterizer, real UI, analytic plume. It looks like a nozzle. It is
not a simulation and must be watermarked **ANALYTIC PREVIEW — NOT A CFD SOLUTION**.
This costs zero minutes at T+150 precisely because it was built in the coordinator
phase. That is the whole reason `MockSolver` is in the blocking block instead of cut.

### Hard stop at T+210

Whatever is green at 210 is the demo. Merge it, run it, stop typing.

The failure mode that actually kills this is not a broken solver — it is you at minute
200 letting one session keep chasing a NaN while the other three sit merged, green and
unshipped.

---

## 6. What this PoC will and will not do

**Will:** correct area-ratio physics, exit Mach and thrust coefficient; sharp captured
shocks; expansion fans; the 2D divergence loss quasi-1D misses (1–3% on C_f, the "sim
beat the textbook" moment); an under-expanded plume with shock-cell structure; a
free-expansion cone approaching vacuum; drawn geometry and a parametric nozzle through
the same solver; an altitude slider whose transient you watch, 5–15 s on the PC and
12–40 s on the M1.

**Will not:** boundary layer, wall heat flux, viscous Isp loss, turbulence,
separation. Frozen γ underpredicts Isp by a few percent. No entrainment, so the plume
edge stays razor-sharp and shock cells persist further downstream than in a
photograph. Mass flow carries ±5% from the staircase wall at the shipping grid. You
get the Mach disk plus half the first reflected cell, not four diamonds — that was a
deliberate 3.2× cut of the plume domain to protect throat resolution, because the
downstream diamonds are the least trustworthy pixels on the screen and throat mass
flow is the number an engineer checks.

The demo case (γ = 1.24, ε = 8, p₀ = 5 MPa) is chosen so its perfect-expansion point
sits at about 2.3 km altitude. That puts **the entire slider range inside the regime
an inviscid solver can be trusted for** — mildly over-expanded at sea level through
under-expanded above, never crossing the separation threshold. Not an accident, and
worth preserving when the numbers get tuned.

---

## 7. Open items this plan does not close

- **Drawn geometry has no acceptance test.** T0–T8 all cover the parametric path. An
  arbitrary user stroke is covered by nothing, which is why drawn-geometry mode carries
  a "sandbox — qualitative only" badge. Closing it needs a drawn-wedge test against
  θ-β-M — affordable, but not in three hours.
- **Grid convergence.** One resolution, no refinement study. The measure grid exists so
  you can run the same case twice and see whether the number moves.
- **The 1D-vs-2D expectation table** in `physics-reference.md` §11 is theory, not
  measurement. Until T8 runs on real output, "0.3–1.0% lower mass flow is physics, 2%
  off is a bug" is a prediction.
- **Case save/load is cut.** Everything is driven from sliders. First thing to add back.
- **Local timestepping is cut**, replaced by the turbo control. It is the biggest
  single steady-state win and worth 30 minutes on the next pass.
