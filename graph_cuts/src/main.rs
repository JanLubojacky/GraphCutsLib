// use ndarray::prelude::*;
use ndarray::{Array2, s};

mod edmons_karp;
mod graph;
mod utils;

// run some tests with an example image
fn main() {

    let (dim1, dim2) = (1000, 1000);

    let mut img = Array2::<f64>::zeros((dim1, dim2));
    let mut mask_fg = Array2::<f64>::zeros((dim1, dim2));
    let mut mask_bg = Array2::<f64>::zeros((dim1, dim2));

    img.slice_mut(s![0..dim1, 0]).fill(10.0);
    img.slice_mut(s![0, 0..dim2]).fill(10.0);

    mask_bg[[0, 0]] = 1.0;
    mask_fg[[dim1-1, dim2-1]] = 1.0;

    // println!("img: {:?}", img);
    // println!("mask_fg: {:?}", mask_fg);
    // println!("mask_bg: {:?}", mask_bg);

    // return;

    // let dim = 10;

    // let mut img = Array2::<f64>::zeros((dim, dim));
    // let mut mask_fg = Array2::<f64>::zeros((dim, dim));
    // let mut mask_bg = Array2::<f64>::zeros((dim, dim));
    //
    // img.slice_mut(s![3..7, 3..7]).fill(255.0);
    // mask_fg[[4, 5]] = 1.0;
    // mask_fg[[5, 4]] = 1.0;
    //
    // mask_bg[[0, 0]] = 1.0;
    // mask_bg[[dim-1, dim-1]] = 1.0;

    // retype to array view
    let img = img.view();
    let mask_fg = mask_fg.view();
    let mask_bg = mask_bg.view();

    println!("constructing graph");

    let mut g = utils::construct_graph(img, mask_fg, mask_bg, 30.0, 8);

    // g.print_graph();

    println!("graph constructed");

    println!("running max flow algorithm");

    edmons_karp::edmons_karp(&mut g);

    println!("max flow computed");

    println!("computing segmentation mask");

    let segmentation_mask = utils::get_foreground_mask(&g, dim1, dim2);

    // println!("segmentation mask: {:?}", segmentation_mask);
}
