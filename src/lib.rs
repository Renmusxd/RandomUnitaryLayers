mod multidefect;
mod singledefect;
pub mod utils;

use pyo3::prelude::*;
use singledefect::*;

#[pymodule]
fn py_entropy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SingleDefectState>()?;
    m.add_class::<ThreadedSingleDefectStates>()?;
    Ok(())
}
