use pyo3::prelude::*;

#[pymodule]
fn _core(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
