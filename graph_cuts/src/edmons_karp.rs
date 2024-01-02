use std::collections::VecDeque;

use crate::graph::Graph;

fn update_flow(g: &mut Graph, path: &VecDeque<usize>, bottleneck: f64) {
    // update the flow along the path
    for (i, u) in path.iter().enumerate() {
        let v = path[i + 1];
        // update the flow along the edge
        g.adj_mat.get_mut(&[*u, v]).unwrap().flow += bottleneck;
        g.adj_mat.get_mut(&[v, *u]).unwrap().flow -= bottleneck;

        if v == g.sink {
            break;
        }
    }
}

// Define a struct to hold both path and bottleneck
#[derive(Debug)]
struct PathWithBottleneck {
    path: VecDeque<usize>,
    bottleneck: f64,
}

// finds the shortest path from source to sink via BFS
//
// returns
//  - the path as a sequence of vertices
//  - the bottleneck of the path
fn find_agumenting_path(g: &mut Graph) -> Option<PathWithBottleneck> {
    let mut visited = vec![false; g.num_vertices];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut predecessors: Vec<i32> = vec![-1; g.num_vertices];

    // start at the source
    queue.push_back(g.source);

    while !queue.is_empty() {
        // pop the next vertex
        let u = queue.pop_front().unwrap();

        // skip if already visited
        if visited[u] {
            continue;
        }

        visited[u] = true;

        // visit all neighbors of u
        for &v_to in g.adj_list[u].iter() {

            let edge = &g.adj_mat[&[u, v_to]];

            if edge.flow == edge.ub {
                continue; // edge is saturated, don't add to queue
            }

            if v_to == g.sink {
                predecessors[v_to] = u as i32;
                queue.clear(); // empty the queue
                break; // found the sink, we are done
            }

            if !visited[v_to] {
                queue.push_back(v_to);
                predecessors[v_to] = u as i32;
            }
        }
    }

    let mut u = g.sink;
    // No path found
    if predecessors[u] == -1 {
        return None;
    }

    // Reconstruct the path from sink to source
    let mut path = VecDeque::new();
    let mut bottleneck: f64 = f64::INFINITY;

    path.push_front(u as usize);

    while u != g.source {
        u = predecessors[u] as usize;

        // get the bottleneck for this
        let edge = &g.adj_mat[&[u, path[0]]];
        let edge_capacity = edge.ub - edge.flow; // flow that can be added

        // update the bottleneck
        bottleneck = bottleneck.min(edge_capacity);

        path.push_front(u);
    }
    Some(PathWithBottleneck { path, bottleneck })
}

// given a graph solve the max flow problem via the min-cut algorithm
pub fn edmons_karp(g: &mut Graph) {
    loop {
        match find_agumenting_path(g) {
            Some(p_and_b) => {
                update_flow(g, &p_and_b.path, p_and_b.bottleneck);
            }
            None => {
                // no path found, we are done
                break;
            }
        }
    }
}

// run bfs from the source
// if a vertex is visited then it is in the source set
// otherwise it is in the sink set
pub fn min_cut(g: &Graph) -> Vec<bool> {
    let mut visited = vec![false; g.num_vertices];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut predecessors: Vec<i32> = vec![-1; g.num_vertices];

    // start at the source
    queue.push_back(g.source);
    while !queue.is_empty() {
        let u = queue.pop_front().unwrap();

        if visited[u] { // skip if already visited
            continue;
        }

        visited[u] = true;

        // visit all neighbors of u
        for &v_to in g.adj_list[u].iter() {
            if g.adj_mat[&[u, v_to]].flow == g.adj_mat[&[u, v_to]].ub {
                continue; // edge is saturated, don't add to queue
            }
            if visited[v_to] {
                continue; // already visited, don't add to queue
            }
            if !visited[v_to] {
                queue.push_back(v_to);
                predecessors[v_to] = u as i32;
            }
        }
    }

    return visited;
}
