use pyo3::prelude::*;

// Kept free of PyO3 types so it can be unit-tested and reused without a Python interpreter.
pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

#[pyfunction]
#[pyo3(name = "add")]
fn add_py(a: f64, b: f64) -> f64 {
    add(a, b)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(add_py, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2.0, 3.0), 5.0);
    }
}
