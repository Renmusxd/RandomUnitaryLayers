mod multidefect;
mod singledefect;
pub mod utils;

use crate::multidefect::*;
use crate::singledefect::*;
use pyo3::prelude::*;

#[pymodule]
fn py_entropy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SingleDefectState>()?;
    m.add_class::<MultiDefectState>()?;
    Ok(())
}
