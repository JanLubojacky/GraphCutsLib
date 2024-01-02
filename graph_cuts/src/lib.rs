use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

mod edmons_karp;
mod graph;
mod utils;

#[pyfunction]
fn segment<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<f64>,
    mask_fg: PyReadonlyArray2<f64>,
    mask_bg: PyReadonlyArray2<f64>,
    sigma: Option<f64>, // optional sigma for the pixel similarity equation
    neighborhood_sz: Option<usize>, // optional neighborhood size for the pixel similarity equation
) -> &'py PyArray2<usize> {
    // Convert PyArrays to ndarrays
    let img_arr = img.as_array();
    let mask_fg_arr = mask_fg.as_array();
    let mask_bg_arr = mask_bg.as_array();

    // construct the graph
    let sigma = match sigma {
        Some(sigma) => sigma,
        None => 20.0,
    };
    let neighborhood_sz = match neighborhood_sz {
        Some(size) => {
            if size == 4 || size == 8 {
                size
            } else {
                panic!("neighborhood_sz must be 4 or 8")
            }
        },
        None => 4,
    };
    let mut g = utils::construct_graph(img_arr, mask_fg_arr, mask_bg_arr, sigma, neighborhood_sz);

    // run the max-flow algorithm (edmonds-karp)
    edmons_karp::edmons_karp(&mut g);

    // get the foreground mask from the min-cut
    let (height, width) = img_arr.dim();
    let segmentation_mask = utils::get_foreground_mask(&g, height, width);

    segmentation_mask.into_pyarray(py)
}

// A Python module implemented in Rust.
#[pymodule]
fn graph_cuts(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(segment, m)?)?;
    Ok(())
}
