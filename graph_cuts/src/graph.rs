use std::collections::HashMap;

#[derive(Debug)]
pub struct Vertex {
    pub flow: f64,
    pub ub: f64,
}

impl Clone for Vertex {
    fn clone(&self) -> Self {
        Self {
            flow: self.flow,
            ub: self.ub,
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            flow: -1.0,
            ub: -1.0,
        }
    }
}

pub struct Graph {
    pub num_vertices: usize,
    pub adj_list: Vec<Vec<usize>>,
    pub adj_mat: HashMap<[usize; 2], Vertex>,
    pub source: usize,
    pub sink: usize,
}

impl Graph {
    // constructor
    pub fn new(num_vertices: usize) -> Self {
        Self {
            num_vertices,
            adj_list: vec![Vec::new(); num_vertices],
            adj_mat: HashMap::new(),
            source: num_vertices - 2,
            sink: num_vertices - 1,
        }
    }

    // initialize edge from u to v with capacity
    pub fn add_edge(&mut self, u: usize, v: usize, capacity: f64) {

        self.adj_list[u].push(v);
        self.adj_list[v].push(u);

        self.adj_mat.insert([u, v], Vertex {
            flow: 0.0,
            ub: capacity,
        });
        self.adj_mat.insert([v, u], Vertex {
            flow: 0.0,
            ub: capacity,
        });
    }

    pub fn print_graph(&self) {
        // print the adjacency list
        for (u, adj) in self.adj_list.iter().enumerate() {
            println!("{}: {:?}", u, adj);
        }
        // print the adjacency matrix
        for (u, adj) in self.adj_mat.iter().enumerate() {
            println!("{}: {:?}", u, adj);
        }
    }
}
