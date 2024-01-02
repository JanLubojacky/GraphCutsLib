use crate::edmons_karp;
use core::f64;

use ndarray::{Array2, ArrayView2};

use crate::graph::Graph;

// compute the affinity between two pixels (range [0,1])
// and discretize it to the range [0,100]
fn pixel_affinity(pixel1: f64, pixel2: f64, sigma: f64) -> f64 {
    // compute the affinity between two pixels
    let diff = pixel1 - pixel2;
    let diff_squared = diff.powi(2) / (2.0 * sigma.powi(2));
    let mut affinity = (-diff_squared).exp();

    // discretize affinity
    affinity = (affinity * 100.0).round();

    return affinity;
}

// given
//  - an image
//  - a foreground mask
//  - a background mask
// construct a graph for graph cuts segmentation
pub fn construct_graph(
    image: ArrayView2<f64>,
    fg_mask: ArrayView2<f64>,
    bg_mask: ArrayView2<f64>,
    sigma: f64,
    neighborhood_sz: usize,
) -> Graph {
    let (height, width) = image.dim();

    let num_vertices = height * width + 2;

    // construct the graph
    let mut g = Graph::new(num_vertices);

    // cast to i32
    let height = height as i32;
    let width = width as i32;

    let neighbourhood_indices = match neighborhood_sz {
        4 => vec![(0, -1), (-1, 0)],
        8 => vec![(0, -1), (-1, 0), (1, 0), (0, 1)],
        _ => panic!("neighborhood_sz must be 4 or 8"),
    };

    // create edges
    for i in 0..height {
        for j in 0..width {
            let pixel = image[[i as usize, j as usize]];
            let u = (i * width + j) as usize; // this vertex

            // if in the foreground mask connect to the source
            if fg_mask[[i as usize, j as usize]] == 1.0 {
                g.add_edge(g.source, u, f64::INFINITY);
            }
            // if in the background mask connect to the sink
            if bg_mask[[i as usize, j as usize]] == 1.0 {
                g.add_edge(u, g.sink, f64::INFINITY);
            }

            if fg_mask[[i as usize, j as usize]] == 1.0 && bg_mask[[i as usize, j as usize]] == 1.0
            {
                panic!(
                    "ERROR: Pixel {}{} is both in foreground and background!",
                    i, j
                );
            }


            for &(ii, jj) in neighbourhood_indices.iter() {

                let i2 = i + ii;
                let j2 = j + jj;

                // skip the current pixel
                if ii == 0 && jj == 0 {
                    continue;
                }

                // check bounds
                if i2 < 0 || i2 >= height || j2 < 0 || j2 >= width {
                    continue;
                }

                let v = (i2 * width + j2) as usize; // target vertex

                // check if edge already exists
                if g.adj_mat.contains_key(&[u, v]) {
                    continue;
                }

                // create an edge between the current pixel
                let pixel2 = image[[i2 as usize, j2 as usize]];
                let affinity = pixel_affinity(pixel, pixel2, sigma);

                if affinity > 0.0 {
                    g.add_edge(u, v, affinity);
                }
            }
        }
    }

    return g;
}

// given a graph where the max flow has been computed
// find the foreground mask via copmuting a min cut
// in the residual graph
pub fn get_foreground_mask(g: &Graph, height: usize, width: usize) -> Array2<usize> {
    let visited: Vec<bool> = edmons_karp::min_cut(g);

    // create a 2d array from the visited array
    let mut mask = Array2::<usize>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            mask[[y, x]] = visited[y * width + x] as usize;
        }
    }

    return mask;
}
