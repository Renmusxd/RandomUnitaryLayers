use crate::utils::get_purity_iterator;
use ndarray::{s, ArrayView1, ArrayView2};
use ndarray_linalg::QR;
use num_complex::Complex;
use num_traits::{One, Zero};
use numpy::ndarray::{Array1, Array2, Array3, Axis};
use numpy::{Complex64, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::cmp::min;

#[pyclass]
pub struct GenericMultiDefectState {
    l: usize,
    n: usize,
    ds: Vec<usize>,
    cmat: Array2<usize>,
    num_experiments: usize,
    // DN x L x 2
    states: Option<Array3<usize>>,
    // State definition and buffer
    // First array is probability weights
    // Amplitudes array is (experiments, mixed, hilbert_space)
    amplitudes: Option<(Array1<f64>, Array3<Complex<f64>>, Array3<Complex<f64>>)>,
    rngs: Option<Vec<SmallRng>>,
    layer_connections: Option<Vec<(usize, usize)>>,
    connections: Option<PrecomputedConnections>,
    parallel_mats: bool,
}

struct PrecomputedConnections {
    connection_lookup: Array2<usize>,
    connection_to_state_to_group_pointers: Array3<usize>,
    connection_to_groups: Array2<usize>,
    n_sector_sizes: Array1<usize>,
}

#[pymethods]
impl GenericMultiDefectState {
    #[new]
    fn new(
        l: usize,
        n: usize,
        mut ds: Vec<usize>,
        num_experiments: Option<usize>,
        layer: Option<Vec<(usize, usize)>>,
        seeds: Option<Vec<u64>>,
        parallel_matrix_mul: Option<bool>,
    ) -> Self {
        let parallel_mats = parallel_matrix_mul.unwrap_or(true);
        while ds.len() <= n {
            ds.push(0)
        }
        let cmat = make_cmat(l, n, &ds);
        let mut head = Default::default();
        let mut vecstates = Default::default();
        make_states(n, l, n, &ds, &mut head, &mut vecstates);
        debug_assert_eq!(cmat[(l, n)], vecstates.len());

        let hilbert_d = vecstates.len();
        let mut states = Array3::zeros((hilbert_d, l, 2));
        states
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(vecstates.into_par_iter())
            .for_each(|(mut state, vecstate)| {
                state
                    .axis_iter_mut(Axis(0))
                    .zip(vecstate.into_iter())
                    .for_each(|(mut s, vs)| {
                        s[0] = vs.0;
                        s[1] = vs.1;
                    })
            });
        let states = Some(states);

        let num_experiments = num_experiments.unwrap_or(1);
        let rngs = if let Some(seeds) = seeds {
            seeds.into_iter().map(SmallRng::seed_from_u64).collect()
        } else {
            (0..num_experiments)
                .map(|_| SmallRng::from_entropy())
                .collect()
        };
        let rngs = Some(rngs);

        let layer_connections = layer.unwrap_or_else(|| {
            (0..l / 2)
                .map(|i| (2 * i, 2 * i + 1))
                .chain((0..l / 2 - 1).map(|i| (2 * i + 1, 2 * i + 2)))
                .collect()
        });
        let layer_connections = Some(layer_connections);

        let num_mixed = 1;

        let probs = Array1::zeros((num_mixed,));
        let amplitudes = Array3::zeros((num_experiments, num_mixed, hilbert_d));
        let buffer = amplitudes.clone();
        let mut s = Self {
            l,
            n,
            ds,
            cmat,
            num_experiments,
            states,
            amplitudes: Some((probs, amplitudes, buffer)),
            rngs,
            layer_connections,
            connections: None,
            parallel_mats,
        };

        let (_, a) = s.get_amplitudes_mut();
        for i in 0..num_experiments {
            for j in 0..num_mixed {
                a[(i, j, 0)] = Complex::one();
            }
        }
        s
    }

    fn get_states(&self, py: Python) -> Py<PyArray3<usize>> {
        self.get_raw_states().clone().to_pyarray(py).to_owned()
    }

    /// Get a random unitary of size `n`
    fn get_random_unitary(&mut self, py: Python, n: usize) -> Py<PyArray2<Complex64>> {
        let mut rng = rand::thread_rng();
        random_unitary(n, &mut rng).to_pyarray(py).to_owned()
    }

    fn apply_layer(&mut self) {
        let mut rngs = self.rngs.take().unwrap();
        let layer_conn = self.layer_connections.take().unwrap();
        layer_conn.iter().copied().for_each(|(a, b)| {
            let unitaries: Vec<Vec<Option<Array2<Complex<f64>>>>> = rngs
                .par_iter_mut()
                .map(|rng| {
                    (0..self.n + 1)
                        .map(|n_sector| {
                            let un = self.states_in_sector(n_sector);
                            if un > 0 {
                                Some(random_unitary(un, rng))
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .collect();
            let uij = |r: usize, n_sector: usize, pos: (usize, usize)| -> Complex<f64> {
                unitaries[r][n_sector]
                    .as_ref()
                    .map(|u| u[(pos.0, pos.1)])
                    .expect("Sector should not be accessible")
            };
            self.apply_matrix(a, b, uij);
        });

        self.layer_connections = Some(layer_conn);
        self.rngs = Some(rngs);
    }

    fn states_in_sector(&self, n_sector: usize) -> usize {
        let max_single_n = min(self.ds.len() - 1, n_sector);
        let min_single_n = n_sector - max_single_n;
        (min_single_n..max_single_n + 1)
            .map(|m| self.ds[max_single_n - m] * self.ds[m])
            .sum()
    }

    /// Get the mean purity across all experiments.
    pub fn get_mean_purity(&self) -> f64 {
        let purities = self.get_purity_iterator();
        purities.sum::<f64>() / (self.num_experiments as f64)
    }

    fn get_precompute_size(&self) -> usize {
        let states = self.states.as_ref().unwrap().shape()[0];
        let layers = self.layer_connections.as_ref().unwrap().len();

        let pointers_size = layers * states * 2 * std::mem::size_of::<usize>();
        let groups_size = layers * states * std::mem::size_of::<usize>();
        let connections_size = self.l * self.l * std::mem::size_of::<usize>();
        let sectors_size = (self.n + 1) * std::mem::size_of::<usize>();
        pointers_size + groups_size + connections_size + sectors_size
    }

    fn precompute_connections(&mut self) {
        if self.connections.is_none() {
            let states = self.states.take().unwrap();
            let layers = self.layer_connections.take().unwrap();

            // For each connection, for each state, point to (start, self_pos).
            let mut state_pointers = Array3::zeros((layers.len(), states.shape()[0], 2));
            let mut state_groups = Array2::zeros((layers.len(), states.shape()[0]));
            let mut connections = Array2::zeros((self.l, self.l)) + usize::MAX;
            let mut sector_sizes = Array1::zeros((self.n + 1,));
            sector_sizes
                .iter_mut()
                .enumerate()
                .for_each(|(n_sector, s)| *s = self.states_in_sector(n_sector));

            layers
                .iter()
                .copied()
                .enumerate()
                .for_each(|(conn_i, (a, b))| connections[(a, b)] = conn_i);
            layers
                .iter()
                .copied()
                .zip(
                    state_pointers
                        .axis_iter_mut(Axis(0))
                        .zip(state_groups.axis_iter_mut(Axis(0))),
                )
                .par_bridge()
                .for_each(|((a, b), (mut pointers, mut groups))| {
                    let mut group_start_pos = 0;
                    states
                        .axis_iter(Axis(0))
                        .enumerate()
                        .map(|(state_index, state)| {
                            self.get_relevant_states_for_n_sector(
                                state_index,
                                state.axis_iter(Axis(0)).map(|t| (t[0], t[1])),
                                a,
                                b,
                            )
                        })
                        .enumerate()
                        .for_each(|(state_index, (state_rel_index, relevant_indices))| {
                            // We always go in order
                            // Check if this is the first time we've seen this group.
                            if state_rel_index == 0 {
                                debug_assert_eq!(state_index, relevant_indices[0]);
                                pointers[(state_index, 0)] = group_start_pos;
                                relevant_indices.iter().copied().enumerate().for_each(
                                    |(i, rel_state)| {
                                        groups[group_start_pos + i] = rel_state;
                                    },
                                );
                                group_start_pos += relevant_indices.len();
                            } else {
                                let my_group_start = pointers[(relevant_indices[0], 0)];
                                pointers[(state_index, 0)] = my_group_start;
                            }
                            pointers[(state_index, 1)] = state_rel_index;
                        })
                });
            self.connections = Some(PrecomputedConnections {
                connection_lookup: connections,
                connection_to_state_to_group_pointers: state_pointers,
                connection_to_groups: state_groups,
                n_sector_sizes: sector_sizes,
            });

            self.states = Some(states);
            self.layer_connections = Some(layers);
        }
    }

    fn get_probabilities_and_amplitudes(
        &self,
        py: Python,
    ) -> (Py<PyArray1<f64>>, Py<PyArray3<Complex<f64>>>) {
        let (ps, amps) = self.get_amplitudes();
        let ps = ps.clone().to_pyarray(py).to_owned();
        let amps = amps.clone().to_pyarray(py).to_owned();
        (ps, amps)
    }

    fn apply_matrix_to_number_sector(
        &mut self,
        a: usize,
        b: usize,
        n_sector: usize,
        matrix: PyReadonlyArray2<Complex64>,
    ) -> PyResult<()> {
        let matrix = matrix.to_owned_array();
        let states_in_sectors = self.states_in_sector(n_sector);
        if matrix.shape() != [states_in_sectors, states_in_sectors] {
            return Err(PyValueError::new_err(format!(
                "Matrix must be of size {:?} for n sector {}",
                [states_in_sectors, states_in_sectors],
                n_sector,
            )));
        }

        self.apply_matrix(a, b, |_, n, pos| {
            if n == n_sector {
                matrix[pos]
            } else if pos.0 == pos.1 {
                Complex::one()
            } else {
                Complex::zero()
            }
        });

        Ok(())
    }
}

struct Dets {
    replica: usize,
    state_index: usize,
}

fn run_on_each_state_parallel<F>(
    amps: &Array3<Complex<f64>>,
    buff: &mut Array3<Complex<f64>>,
    states: &Array3<usize>,
    f: F,
) where
    F: Fn(
            (
                Dets,
                ArrayView2<usize>,
                ArrayView1<Complex<f64>>,
                &mut Complex<f64>,
            ),
        ) + Send
        + Sync
        + Copy,
{
    amps.axis_iter(Axis(0))
        .into_par_iter()
        .zip(buff.axis_iter_mut(Axis(0)).into_par_iter())
        .enumerate()
        .for_each(|(r, (amps, mut buff))| {
            // Iterate mixed states
            amps.axis_iter(Axis(0))
                .into_par_iter()
                .zip(buff.axis_iter_mut(Axis(0)).into_par_iter())
                .for_each(|(amps, mut buff)| {
                    // Iterate quantum states
                    buff.axis_iter_mut(Axis(0))
                        .into_par_iter()
                        .zip(states.axis_iter(Axis(0)).into_par_iter())
                        .enumerate()
                        .map(|(state_index, (buff, s))| {
                            (
                                Dets {
                                    replica: r,
                                    state_index,
                                },
                                s,
                                amps,
                                buff.into_iter().next().unwrap(),
                            )
                        })
                        .for_each(f)
                });
        });
}

fn run_on_each_state<F>(
    amps: &Array3<Complex<f64>>,
    buff: &mut Array3<Complex<f64>>,
    states: &Array3<usize>,
    f: F,
) where
    F: Fn(
            (
                Dets,
                ArrayView2<usize>,
                ArrayView1<Complex<f64>>,
                &mut Complex<f64>,
            ),
        ) + Send
        + Sync
        + Copy,
{
    amps.axis_iter(Axis(0))
        .zip(buff.axis_iter_mut(Axis(0)))
        .enumerate()
        .for_each(|(r, (amps, mut buff))| {
            // Iterate mixed states
            amps.axis_iter(Axis(0))
                .zip(buff.axis_iter_mut(Axis(0)))
                .for_each(|(amps, mut buff)| {
                    // Iterate quantum states
                    buff.axis_iter_mut(Axis(0))
                        .zip(states.axis_iter(Axis(0)))
                        .enumerate()
                        .map(|(state_index, (buff, s))| {
                            (
                                Dets {
                                    replica: r,
                                    state_index,
                                },
                                s,
                                amps,
                                buff.into_iter().next().unwrap(),
                            )
                        })
                        .for_each(f)
                });
        });
}

impl GenericMultiDefectState {
    /// Applies the matrix unitary, represented by a function f(replica, n_sector, (r,c))
    fn apply_matrix<F>(&mut self, a: usize, b: usize, unitary: F)
    where
        F: Fn(usize, usize, (usize, usize)) -> Complex<f64> + Sync + Send,
    {
        let (probs, amps, mut buff) = self.amplitudes.take().unwrap();
        let states = self.states.take().unwrap();

        let conn = self.connections.as_ref().and_then(|conn| {
            if conn.connection_lookup[(a, b)] != usize::MAX {
                Some(conn)
            } else {
                None
            }
        });

        // Iterate experiments
        if let Some(connections) = conn {
            let connection_index = connections.connection_lookup[(a, b)];
            let group_pointers = connections.connection_to_state_to_group_pointers.slice(s![
                connection_index,
                ..,
                ..
            ]);
            let groups = connections
                .connection_to_groups
                .slice(s![connection_index, ..]);

            let f = |(index, s, amps, buff): (
                Dets,
                ArrayView2<usize>,
                ArrayView1<Complex<f64>>,
                &mut Complex<f64>,
            )| {
                let a_m = s[(a, 0)];
                let b_m = s[(b, 0)];
                let n_sector = a_m + b_m;

                let start_index = group_pointers[(index.state_index, 0)];
                let output_mat_index = group_pointers[(index.state_index, 1)];
                let n_sector_size = connections.n_sector_sizes[n_sector];
                let relevant_state_indices =
                    groups.slice(s![start_index..start_index + n_sector_size]);

                *buff = relevant_state_indices
                    .into_iter()
                    .copied()
                    .enumerate()
                    .map(|(mat_index, state_index)| {
                        let uij = unitary(index.replica, n_sector, (output_mat_index, mat_index));
                        uij * amps[state_index]
                    })
                    .sum::<Complex<f64>>();
            };

            if self.parallel_mats {
                run_on_each_state_parallel(&amps, &mut buff, &states, f);
            } else {
                run_on_each_state(&amps, &mut buff, &states, f);
            }
        } else {
            let f = |(index, s, amps, buff): (
                Dets,
                ArrayView2<usize>,
                ArrayView1<Complex<f64>>,
                &mut Complex<f64>,
            )| {
                let r = index.replica;
                let index = index.state_index;
                let a_m = s[(a, 0)];
                let b_m = s[(b, 0)];
                let n_sector = a_m + b_m;
                let (output_mat_index, relevant_state_indices) = self
                    .get_relevant_states_for_n_sector(
                        index,
                        s.axis_iter(Axis(0)).map(|t| (t[0], t[1])),
                        a,
                        b,
                    );

                *buff = relevant_state_indices
                    .into_iter()
                    .enumerate()
                    .map(|(mat_index, state_index)| {
                        let uij = unitary(r, n_sector, (output_mat_index, mat_index));
                        uij * amps[state_index]
                    })
                    .sum::<Complex<f64>>();
            };

            if self.parallel_mats {
                run_on_each_state_parallel(&amps, &mut buff, &states, f);
            } else {
                run_on_each_state(&amps, &mut buff, &states, f);
            }
        }
        self.states = Some(states);
        self.amplitudes = Some((probs, buff, amps));
    }

    fn get_relevant_states_for_n_sector<It>(
        &self,
        index: usize,
        state: It,
        a: usize,
        b: usize,
    ) -> (usize, Vec<usize>)
    where
        It: IntoIterator<Item = (usize, usize)>,
    {
        let mut state_array = state.into_iter().collect::<Vec<(usize, usize)>>();

        let a_m = state_array[a].0;
        let b_m = state_array[b].0;
        let n_sector = a_m + b_m;
        let max_single_n = min(self.ds.len() - 1, n_sector);
        let remaining_value = n_sector - max_single_n;

        // Iterates from (x, y) to (y, x) where x+y = n_sector and neither exceeds max.
        let mut output_mat_index = 0; // If we could precalculate this would be way cleaner.
        let ds = &self.ds;
        let res = (remaining_value..max_single_n + 1)
            .flat_map(|m| {
                let x_m = m;
                let y_m = n_sector - m;
                (0..ds[x_m])
                    .flat_map(move |x_s| (0..ds[y_m]).map(move |y_s| ((x_m, x_s), (y_m, y_s))))
            })
            .enumerate()
            .map(|(mat_index, ((x_m, x_s), (y_m, y_s)))| {
                state_array[a] = (x_m, x_s);
                state_array[b] = (y_m, y_s);
                let state_index = self.get_index_for_state(state_array.iter().copied());
                if state_index == index {
                    output_mat_index = mat_index
                }
                state_index
            })
            .collect::<Vec<_>>();
        (output_mat_index, res)
    }

    /// Compute the purity estimator of the state and return as a floating point value.
    pub fn get_purity_iterator(&self) -> impl IndexedParallelIterator<Item = f64> + '_ {
        let hilbert_d = self.states.as_ref().unwrap().shape()[0];
        let (probs, amps, _) = self.amplitudes.as_ref().unwrap();
        let probs = probs.as_slice().unwrap();
        get_purity_iterator(hilbert_d, probs, amps)
    }

    // TODO A pain to debug so skipping for now.
    // fn move_defects<F>(
    //     &self,
    //     n_to_move: usize,
    //     ia: usize,
    //     ib: usize,
    //     mut index: usize,
    //     state_lookup: F,
    //     n_to_ia: usize,
    // ) -> usize
    // where
    //     F: Fn(usize) -> (usize, usize),
    // {
    //     if ia < ib {
    //         let (ia_m, ia_s) = state_lookup(ia);
    //         let (ib_m, ib_s) = state_lookup(ib);
    //         let n_to_and_ia = n_to_ia + ia_m;
    //
    //         debug_assert_eq!(n_to_and_ia, {
    //             (0..ia + 1).map(|x| state_lookup(x).0).sum()
    //         });
    //         debug_assert!(state_lookup(ia).0 >= n_to_move);
    //         debug_assert!(self.ds.len() > state_lookup(ib).0 + n_to_move);
    //         debug_assert!(self.ds[state_lookup(ib).0 + n_to_move] > 0);
    //
    //         // Move a defect from ia to ib
    //         // First decrement ia
    //         let mut subl = self.l - ia - 1;
    //         let mut base_subn = self.n - n_to_and_ia;
    //         let mut subn = base_subn;
    //         index -= self.cmat[(subl, subn)] * ia_s; // reduce to (ia_m, 0)
    //         for mm in 0..n_to_move {
    //             subn += 1;
    //             index -= self.cmat[(subl, subn)] * self.ds[ia_m - mm - 1];
    //         } // reduce (ia_m - n_to_move, 0)
    //         index += self.cmat[(subl, subn)] * ia_s; // increase to (ia_m - n_to_move, ia_s)
    //
    //         for x in ia + 1..ib - 1 {
    //             // Decrement subl as we move forward.
    //             subl -= 1;
    //             // For each of these the m doesn't change, only the subn
    //             let (x_m, x_s) = state_lookup(x);
    //             let de = self.ds[x_m];
    //             base_subn -= x_m;
    //
    //             let mut subn = base_subn;
    //             index -= self.cmat[(subl, subn)] * x_s; // reduce to (x_m, 0)
    //             for _ in 0..n_to_move {
    //                 subn += 1;
    //                 index -= self.cmat[(subl, subn)] * de;
    //             }
    //             index += self.cmat[(subl, subn)] * x_s; // increase to (x_m, x_s)
    //         }
    //
    //         // Now increment ib
    //         base_subn -= ib_m;
    //         base_subn += n_to_move;
    //         let mut subn = base_subn;
    //         index -= self.cmat[(subl, subn)] * ib_s; // reduce to (ib_m, 0)
    //         for x in 0..n_to_move {
    //             index += self.cmat[(subl, subn)] * self.ds[ib_m + x];
    //             subn -= 1;
    //         } // increase to (ib_m, 0)
    //         index += self.cmat[(subl, subn)] * ib_s; // increase to (ib_m + n_to_move, ib_s)
    //
    //         index
    //     } else {
    //         unimplemented!()
    //     }
    // }

    fn get_amplitudes(&self) -> (&Array1<f64>, &Array3<Complex<f64>>) {
        self.amplitudes.as_ref().map(|(p, a, _)| (p, a)).unwrap()
    }

    fn get_amplitudes_mut(&mut self) -> (&mut Array1<f64>, &mut Array3<Complex<f64>>) {
        self.amplitudes.as_mut().map(|(p, a, _)| (p, a)).unwrap()
    }

    fn get_raw_states(&self) -> &Array3<usize> {
        self.states.as_ref().unwrap()
    }

    fn get_index_for_state<It>(&self, state: It) -> usize
    where
        It: IntoIterator<Item = (usize, usize)>,
    {
        let mut index = 0;
        let mut net_n = 0;

        for (i, (m, s)) in state.into_iter().enumerate() {
            let subl = self.l - (i + 1);
            for mm in 0..m {
                let subn = self.n - net_n;
                let sub_states = self.ds[mm] * self.cmat[(subl, subn)];
                index += sub_states;
                net_n += 1;
            }
            let subn = self.n - net_n;
            let sub_states = s * self.cmat[(subl, subn)];
            index += sub_states;
        }
        index
    }
}

fn random_unitary<R: Rng>(n: usize, rng: &mut R) -> Array2<Complex<f64>> {
    let gaussian_matrix = Array2::from_shape_fn((n, n), |_| {
        Complex::new(rng.sample(StandardNormal), rng.sample(StandardNormal))
            * std::f64::consts::FRAC_1_SQRT_2
    });
    let (mut q, r) = gaussian_matrix.qr().unwrap();
    let lambda = r.diag().map(|r| r / r.norm());
    q.axis_iter_mut(Axis(0))
        .zip(lambda.into_iter())
        .for_each(|(mut row, lambda)| row.iter_mut().for_each(|r| *r *= lambda));
    q
}

fn make_states(
    total_n: usize,
    l: usize,
    n: usize,
    ds: &[usize],
    head: &mut Vec<(usize, usize)>,
    states: &mut Vec<Vec<(usize, usize)>>,
) {
    if l == 0 {
        let net = head.iter().map(|(x, _)| *x).sum::<usize>();
        if net == total_n {
            states.push(head.clone());
        }
    } else {
        for m in 0..n + 1 {
            let de = if m < ds.len() { ds[m] } else { 0 };
            for s in 0..de {
                head.push((m, s));
                make_states(total_n, l - 1, n - m, ds, head, states);
                head.pop();
            }
        }
    }
}

fn make_cmat(l: usize, n: usize, ds: &[usize]) -> Array2<usize> {
    let mut cmat = Array2::zeros((l + 1, n + 1));
    cmat[(0, 0)] = 1;
    for l in 1..l + 1 {
        for m in 0..n + 1 {
            for mm in 0..m + 1 {
                cmat[(l, m)] += ds[mm] * cmat[(l - 1, m - mm)]
            }
        }
    }
    cmat
}

#[cfg(test)]
mod generic_tests {
    use super::*;

    #[test]
    fn cmat_test() {
        let cmat = make_cmat(4, 2, &[2, 2, 1, 0, 0]);
        println!("{:?}", cmat);
        assert_eq!(cmat[(4, 2)], 128);
    }

    #[test]
    fn states_test() {
        let mut head = vec![];
        let mut states = vec![];
        make_states(2, 4, 2, &[2, 2, 1, 0, 0], &mut head, &mut states);
        assert_eq!(states.len(), 128);
    }

    #[test]
    fn get_index() {
        // Test getting each index for a L=4 N=2 system with nontrivial de
        let multistate =
            GenericMultiDefectState::new(4, 2, vec![2, 2, 1, 0, 0], None, None, None, None);
        let states = multistate.get_raw_states().clone();
        states.axis_iter(Axis(0)).for_each(|state| {
            let state_index = multistate
                .get_index_for_state(state.clone().axis_iter(Axis(0)).map(|a| (a[0], a[1])));
            let slice = states.slice(s![state_index, .., ..]);

            for (astate, bstate) in state
                .clone()
                .axis_iter(Axis(0))
                .zip(slice.axis_iter(Axis(0)))
            {
                assert_eq!(astate[0], bstate[0]);
                assert_eq!(astate[1], bstate[1]);
            }
        });
    }

    #[test]
    fn get_index_packed() {
        // Test getting each index for a L=4 N=2 system with nontrivial de
        let multistate =
            GenericMultiDefectState::new(4, 2, vec![2, 2, 0, 0, 0], None, None, None, None);
        let states = multistate.get_raw_states().clone();
        states.axis_iter(Axis(0)).for_each(|state| {
            let state_index = multistate
                .get_index_for_state(state.clone().axis_iter(Axis(0)).map(|a| (a[0], a[1])));
            let slice = states.slice(s![state_index, .., ..]);

            for (astate, bstate) in state
                .clone()
                .axis_iter(Axis(0))
                .zip(slice.axis_iter(Axis(0)))
            {
                assert_eq!(astate[0], bstate[0]);
                assert_eq!(astate[1], bstate[1]);
            }
        });
    }

    #[test]
    fn test_get_relevant_entries() {
        let ds = vec![2, 2, 1, 0, 0];
        let multistate = GenericMultiDefectState::new(4, 2, ds.clone(), None, None, None, None);
        let example_state = [(1, 0), (1, 0), (0, 0), (0, 0)];
        let a = 0;
        let b = 1;
        let example_index = multistate.get_index_for_state(example_state.clone());
        let (out, rels) =
            multistate.get_relevant_states_for_n_sector(example_index, example_state.clone(), a, b);
        assert_eq!(out, 2);
        let mut deduped_rels = rels.clone();
        deduped_rels.dedup();
        assert_eq!(
            deduped_rels.len(),
            ds[0] * ds[2] + ds[1] * ds[1] + ds[2] * ds[0]
        );

        let states = multistate.get_raw_states();
        rels.into_iter()
            .map(|index| states.slice(s![index, .., ..]))
            .enumerate()
            .for_each(|(index, val)| {
                let val = val
                    .axis_iter(Axis(0))
                    .map(|a| (a[0], a[1]))
                    .collect::<Vec<_>>();
                assert_eq!(val[a].0 + val[b].0, example_state[a].0 + example_state[b].0);
            })
    }

    #[test]
    fn test_get_relevant_entries_mix() {
        let ds = vec![2, 2, 1, 0, 0];
        let multistate = GenericMultiDefectState::new(4, 2, ds.clone(), None, None, None, None);
        let example_state = [(1, 0), (1, 0), (0, 0), (0, 0)];
        let a = 0;
        let b = 2;
        let example_index = multistate.get_index_for_state(example_state.clone());
        let (out, rels) =
            multistate.get_relevant_states_for_n_sector(example_index, example_state.clone(), a, b);

        assert_eq!(out, 4);
        let mut deduped_rels = rels.clone();
        deduped_rels.dedup();
        assert_eq!(deduped_rels.len(), ds[0] * ds[1] + ds[1] * ds[0]);

        let states = multistate.get_raw_states();
        rels.into_iter()
            .map(|index| states.slice(s![index, .., ..]))
            .enumerate()
            .for_each(|(index, val)| {
                let val = val
                    .axis_iter(Axis(0))
                    .map(|a| (a[0], a[1]))
                    .collect::<Vec<_>>();
                assert_eq!(val[a].0 + val[b].0, example_state[a].0 + example_state[b].0);
            })
    }

    #[test]
    fn test_apply_matrix_two_sector() {
        let ds = vec![1, 1, 1, 0, 0];
        let mut multistate = GenericMultiDefectState::new(4, 2, ds.clone(), None, None, None, None);

        let (a, b) = (0, 1);
        let n_sector = 2;

        let example_state = [(2, 0), (0, 0), (0, 0), (0, 0)];
        let flipped_state = [(1, 0), (1, 0), (0, 0), (0, 0)];
        // there's also (0,0), (2,0) around.
        assert_eq!(multistate.states_in_sector(2), 3);
        let example_index = multistate.get_index_for_state(example_state.clone());

        let relevant_states =
            multistate.get_relevant_states_for_n_sector(example_index, example_state.clone(), a, b);

        let (_, amps) = multistate.get_amplitudes_mut();
        amps.iter_mut().for_each(|a| *a = Complex::zero());
        amps[(0, 0, example_index)] = Complex::one();

        multistate.apply_matrix(a, b, |_, n, pos| {
            if n == n_sector {
                Complex::one()
            } else if pos.0 == pos.1 {
                Complex::one()
            } else {
                Complex::zero()
            }
        });

        let (_, amps) = multistate.get_amplitudes();
        let indices = amps
            .iter()
            .enumerate()
            .filter(|(_, c)| c.norm_sqr() > f64::EPSILON)
            .map(|x| x.0)
            .collect::<Vec<_>>();
        assert_eq!(relevant_states.1.as_slice(), indices.as_slice())
    }

    #[test]
    fn test_apply_matrix_one_sector() {
        let ds = vec![1, 1, 1, 0, 0];
        let mut multistate = GenericMultiDefectState::new(4, 2, ds.clone(), None, None, None, None);

        let (a, b) = (1, 2);
        let n_sector = 1;

        let example_state = [(1, 0), (1, 0), (0, 0), (0, 0)];
        let flipped_state = [(1, 0), (0, 0), (1, 0), (0, 0)];
        // there's also (0,0), (2,0) around.
        assert_eq!(multistate.states_in_sector(1), 2);
        let example_index = multistate.get_index_for_state(example_state.clone());

        let relevant_states =
            multistate.get_relevant_states_for_n_sector(example_index, example_state.clone(), a, b);

        let (_, amps) = multistate.get_amplitudes_mut();
        amps.iter_mut().for_each(|a| *a = Complex::zero());
        amps[(0, 0, example_index)] = Complex::one();

        multistate.apply_matrix(a, b, |_, n, pos| {
            if n == n_sector {
                Complex::one()
            } else if pos.0 == pos.1 {
                Complex::one()
            } else {
                Complex::zero()
            }
        });

        let (_, amps) = multistate.get_amplitudes();
        let indices = amps
            .iter()
            .enumerate()
            .filter(|(_, c)| c.norm_sqr() > f64::EPSILON)
            .map(|x| x.0)
            .collect::<Vec<_>>();
        assert_eq!(relevant_states.1.as_slice(), indices.as_slice());
    }

    #[test]
    fn test_layer() {
        let mut g = GenericMultiDefectState::new(20, 5, vec![1, 1], None, None, None, None);
        g.apply_layer()
    }

    #[test]
    fn test_precompute_apply_matrix_two_sector() {
        let ds = vec![1, 1, 1, 0, 0];
        let mut multistate = GenericMultiDefectState::new(4, 2, ds.clone(), None, None, None, None);
        multistate.precompute_connections();

        let (a, b) = (0, 1);
        let n_sector = 2;

        let example_state = [(2, 0), (0, 0), (0, 0), (0, 0)];
        let flipped_state = [(1, 0), (1, 0), (0, 0), (0, 0)];
        // there's also (0,0), (2,0) around.
        assert_eq!(multistate.states_in_sector(2), 3);
        let example_index = multistate.get_index_for_state(example_state.clone());

        let relevant_states =
            multistate.get_relevant_states_for_n_sector(example_index, example_state.clone(), a, b);

        let (_, amps) = multistate.get_amplitudes_mut();
        amps.iter_mut().for_each(|a| *a = Complex::zero());
        amps[(0, 0, example_index)] = Complex::one();

        multistate.apply_matrix(a, b, |_, n, pos| {
            if n == n_sector {
                Complex::one()
            } else if pos.0 == pos.1 {
                Complex::one()
            } else {
                Complex::zero()
            }
        });

        let (_, amps) = multistate.get_amplitudes();
        let indices = amps
            .iter()
            .enumerate()
            .filter(|(_, c)| c.norm_sqr() > f64::EPSILON)
            .map(|x| x.0)
            .collect::<Vec<_>>();
        assert_eq!(relevant_states.1.as_slice(), indices.as_slice())
    }

    #[test]
    fn test_precompute_apply_matrix_one_sector() {
        let ds = vec![1, 1, 1, 0, 0];
        let mut multistate = GenericMultiDefectState::new(4, 2, ds.clone(), None, None, None, None);
        multistate.precompute_connections();

        let (a, b) = (1, 2);
        let n_sector = 1;

        let example_state = [(1, 0), (1, 0), (0, 0), (0, 0)];
        let flipped_state = [(1, 0), (0, 0), (1, 0), (0, 0)];
        // there's also (0,0), (2,0) around.
        assert_eq!(multistate.states_in_sector(1), 2);
        let example_index = multistate.get_index_for_state(example_state.clone());

        let relevant_states =
            multistate.get_relevant_states_for_n_sector(example_index, example_state.clone(), a, b);

        let (_, amps) = multistate.get_amplitudes_mut();
        amps.iter_mut().for_each(|a| *a = Complex::zero());
        amps[(0, 0, example_index)] = Complex::one();

        multistate.apply_matrix(a, b, |_, n, pos| {
            if n == n_sector {
                Complex::one()
            } else if pos.0 == pos.1 {
                Complex::one()
            } else {
                Complex::zero()
            }
        });

        let (_, amps) = multistate.get_amplitudes();
        let indices = amps
            .iter()
            .enumerate()
            .filter(|(_, c)| c.norm_sqr() > f64::EPSILON)
            .map(|x| x.0)
            .collect::<Vec<_>>();
        assert_eq!(relevant_states.1.as_slice(), indices.as_slice());
    }

    // TODO
    // #[test]
    // fn test_change_index() {
    //     let multistate = GenericMultiDefectState::new(4, 2, vec![2, 2, 1, 0, 0], None);
    //     let example_state = [(2, 0), (0, 0), (0, 0), (0, 0)];
    //     let expected_state = [(1, 0), (1, 0), (0, 0), (0, 0)];
    //     let example_index = multistate.get_index_for_state(example_state.clone());
    //     let expected_index = multistate.get_index_for_state(expected_state.clone());
    //
    //     let moved_index = multistate.move_defects(1, 0, 1, example_index, |x| example_state[x], 0);
    //     let newstate = multistate.get_raw_states().slice(s![moved_index, .., ..]);
    //     assert_eq!(
    //         moved_index, expected_index,
    //         "Incorrect state: {:?}",
    //         newstate
    //     );
    // }
}
