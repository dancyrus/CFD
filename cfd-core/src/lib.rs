//! Solver crate. This file and `step.rs` were frozen during the parallel
//! build phase — the coordinator wrote them; sessions A and B implement
//! `kernel.rs` and `physics.rs` respectively and never opened the frozen
//! files. That is what made the merge textually conflict-free. The freeze is
//! over; changes follow the CLAUDE.md contract-change rule (mirror
//! docs/contract.md and docs/physics-reference.md in the same commit, re-run
//! the full acceptance ladder).
//!
//! `forces.rs` is the body-force half of the engineering output: `Report`'s
//! exit-plane control volume is meaningless for arbitrary drawn geometry, so
//! surface pressure integrals over immersed bodies live there.
//!
//! Internal array layout (private to this crate): every solver-internal slice
//! — `Cons`, `Prim`, `solid`, `mask`, `rhs` — is padded, `grid.glen()` long,
//! indexed with `Grid::gidx`. Ghost cells never cross a crate boundary.

pub mod forces;
pub mod kernel;
pub mod physics;

mod mock;
mod step;

pub use mock::MockSolver;
pub use step::EulerSolver;
