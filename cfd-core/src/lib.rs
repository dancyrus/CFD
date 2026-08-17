//! Solver crate. **This file and `step.rs` are frozen** — the coordinator
//! wrote them; sessions A and B implement `kernel.rs` and `physics.rs`
//! respectively and never open the frozen files. That is what makes the
//! merge textually conflict-free.
//!
//! Internal array layout (private to this crate): every solver-internal slice
//! — `Cons`, `Prim`, `solid`, `mask`, `rhs` — is padded, `grid.glen()` long,
//! indexed with `Grid::gidx`. Ghost cells never cross a crate boundary.

pub mod kernel;
pub mod monitor;
pub mod physics;

mod mock;
mod step;

pub use mock::MockSolver;
pub use step::EulerSolver;
