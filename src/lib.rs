use pyo3::prelude::*;

// One Rust file per module, each exposing `register`: adding a module touches a single line here.
mod example;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    example::register(module)?;
    Ok(())
}
