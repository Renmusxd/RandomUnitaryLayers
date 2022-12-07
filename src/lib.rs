#[cfg(feature = "internaldof")]
pub mod genericrandomcircuit;
pub mod multidefect;
pub mod singledefect;
pub mod utils;

#[cfg(feature = "internaldof")]
use crate::genericrandomcircuit::*;
use crate::multidefect::*;
use crate::singledefect::*;
use pyo3::prelude::*;

#[pymodule]
fn py_entropy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SingleDefectState>()?;
    m.add_class::<MultiDefectState>()?;
    m.add_class::<MultidefectPureState>()?;
    #[cfg(feature = "internaldof")]
    m.add_class::<GenericMultiDefectState>()?;
    Ok(())
}
