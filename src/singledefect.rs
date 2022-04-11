use crate::utils::*;
use num_complex::Complex;
use numpy::ndarray::{s, Array1, Array3, Axis};
use numpy::{c64, IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct SingleDefectState {
    pub d: usize,
    // Mixed states probs
    pub probs: Array1<f64>,
    // Experiments x Mixed states x Hilbert Space
    pub states: Array3<Complex<f64>>,
}

#[pymethods]
impl SingleDefectState {
    /// Construct a new instance.
    #[new]
    fn new(state: PyReadonlyArray1<c64>, num_experiments: Option<usize>) -> PyResult<Self> {
        Self::new_mixed(vec![(1.0, state)], num_experiments)
    }

    #[staticmethod]
    fn new_mixed(
        state: Vec<(f64, PyReadonlyArray1<c64>)>,
        num_experiments: Option<usize>,
    ) -> PyResult<Self> {
        let num_experiments = num_experiments.unwrap_or(1);
        if state.is_empty() {
            return Err(PyValueError::new_err(
                "Mixed state must contain at least one state",
            ));
        }
        let probs = state.iter().map(|(w, _)| *w).collect();
        let states = state
            .iter()
            .try_fold(vec![], |mut acc, (_, s)| -> Result<_, _> {
                s.as_slice().map(|s| {
                    acc.push(s);
                    acc
                })
            })?;
        let d = state[0].1.len();
        let mut res_states = Array3::zeros((num_experiments, state.len(), d));
        res_states
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut x| {
                // x is a Mixed x Hilbert matrix
                x.axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(states.par_iter())
                    .for_each(|(mut x, s)| {
                        x.iter_mut().zip(s.iter()).for_each(|(x, s)| {
                            *x = *s;
                        });
                    })
            });
        let states = res_states;
        Ok(Self { d, probs, states })
    }

    pub fn apply_layer(&mut self, offset: bool, periodic_boundaries: Option<bool>) {
        let periodic_boundaries = periodic_boundaries.unwrap_or_default();
        self.states
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut state| {
                let num_states_mixed = state.shape()[0];
                let mut state_subsystem = if offset {
                    state.slice_mut(s![.., 1..])
                } else {
                    state.slice_mut(s![.., ..])
                };
                state_subsystem
                    .axis_chunks_iter_mut(Axis(1), 2)
                    .into_par_iter()
                    .filter(|c| c.shape()[1] == 2)
                    .for_each(|mut substate| {
                        debug_assert_eq!(substate.shape(), [num_states_mixed, 2]);
                        let mut rng = rand::thread_rng();
                        let mat = make_unitary(&mut rng);
                        substate
                            .axis_iter_mut(Axis(0))
                            .into_par_iter()
                            .for_each(|mut ab| {
                                debug_assert_eq!(ab.shape(), [2]);
                                let mut oa = ab[0];
                                let mut ob = ab[1];
                                apply_matrix(&mut oa, &mut ob, &mat);
                                ab[0] = oa;
                                ab[1] = ob;
                            });
                    });

                if periodic_boundaries && offset {
                    let mut rng = rand::thread_rng();
                    let mat = make_unitary(&mut rng);
                    let n = state.shape()[1];
                    state
                        .axis_iter_mut(Axis(0))
                        .into_par_iter()
                        .for_each(|mut state| {
                            let mut oa = state[n - 1];
                            let mut ob = state[0];
                            apply_matrix(&mut oa, &mut ob, &mat);
                            state[n - 1] = oa;
                            state[0] = ob;
                        })
                }
            });
    }

    pub fn make_unitaries(&self, py: Python, n: usize) -> PyResult<Py<PyArray3<c64>>> {
        let mut rng = rand::thread_rng();
        let mut res = Array3::zeros((n, 2, 2));
        res.axis_iter_mut(numpy::ndarray::Axis(0))
            .zip((0..n).map(|_| make_unitary(&mut rng)))
            .for_each(|(mut r, mat)| {
                r.iter_mut().zip(mat.into_iter()).for_each(|(r, m)| *r = m);
            });
        Ok(res.into_pyarray(py).to_owned())
    }

    /// Apply alternating layers of random unitaries.
    pub fn apply_alternative_layers(&mut self, n_layers: usize, periodic_boundaries: Option<bool>) {
        for i in 0..n_layers {
            // Layers alternate between offset and not-offset.
            self.apply_layer(i % 2 == 1, periodic_boundaries);
        }
    }

    /// Get the state of the system.
    pub fn get_state(&self, py: Python) -> (Py<PyArray1<f64>>, Py<PyArray3<c64>>) {
        let probs = self.probs.to_pyarray(py).to_owned();
        let states = self.states.to_pyarray(py).to_owned();
        (probs, states)
    }

    pub fn get_mean_purity(&self) -> f64 {
        let purities = self.get_purity_iterator();
        purities.sum::<f64>() / (self.states.shape()[0] as f64)
    }

    pub fn get_all_purities(&self, py: Python) -> Py<PyArray1<f64>> {
        let purity_iterator = self.get_purity_iterator();
        let mut purities = Array1::<f64>::zeros((self.states.shape()[0],));
        purity_iterator
            .zip(purities.axis_iter_mut(Axis(0)).into_par_iter())
            .for_each(|(pa, mut pb)| {
                // Should just be 1.
                pb.iter_mut().for_each(|pb| *pb = pa);
            });
        purities.into_pyarray(py).to_owned()
    }

    /// Compute the purity at each layer of the process and save to a numpy array.
    pub fn apply_alternative_layers_and_save_mean_purity(
        &mut self,
        py: Python,
        n_layers: usize,
        periodic_boundaries: Option<bool>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let mut res = Array1::zeros((n_layers,));
        self.apply_alternative_layers_and_store_mean_purity(res.iter_mut(), periodic_boundaries);
        Ok(res.into_pyarray(py).to_owned())
    }
}

impl SingleDefectState {
    /// Compute the purity estimator of the state and return as a floating point value.
    pub fn get_purity_iterator(&self) -> impl IndexedParallelIterator<Item = f64> + '_ {
        self.states
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|state| -> f64 {
                if state.shape()[0] == 1 {
                    // Fast way
                    let state = state.slice(s![0, ..]);
                    // F = D ( sum_s p(s)^2 - D sum_{s!=s'} p(s)p(s') )
                    // p(s) = |<s|u|i>|^2
                    // internal state is u|i>

                    // First term is sum over |psi(i)|^4
                    let first_term = state.iter().map(|c| c.norm_sqr().powi(2)).sum::<f64>();

                    // Second term is 2 sum_s p(s) ( sum_{s'>s} p(s') )
                    // Can build the sum_{s'>s} backwards from the end to turn O(n^2) into O(n)
                    let fs = state.iter().rev().map(Complex::norm_sqr);
                    let half_second_term = sum_s_sprime_iterator(fs, 0.0, 0.0);
                    let second_term = 2.0 * half_second_term / (self.d as f64);

                    first_term - second_term
                } else {
                    // Slower way
                    // first_term = sum_s ( sum_alpha p_alpha |<s|U|alpha>|^2)^2
                    let first_term = state
                        .axis_iter(Axis(1))
                        .into_par_iter()
                        .map(|s_for_each_mix| {
                            s_for_each_mix
                                .iter()
                                .zip(self.probs.iter())
                                .map(|(c, p_alpha)| p_alpha * c.norm_sqr())
                                .sum::<f64>()
                                .powi(2)
                        })
                        .sum::<f64>();

                    let probs = &self.probs;
                    let f = |s: usize| -> f64 {
                        state
                            .slice(s![.., s])
                            .iter()
                            .zip(probs.iter().cloned())
                            .map(|(c, p_alpha)| p_alpha * c.norm_sqr())
                            .sum::<f64>()
                    };
                    let half_second_term =
                        sum_s_sprime_iterator((0..self.d).rev().map(f), 0.0, 0.0);

                    let second_term = 2.0 * half_second_term / (self.d as f64);

                    first_term - second_term
                }
            })
    }

    /// Compute the purity at each layer of the process and save to a numpy array.
    pub fn apply_alternative_layers_and_store_mean_purity<'a, It>(
        &mut self,
        purity_iterator: It,
        periodic_boundaries: Option<bool>,
    ) where
        It: IntoIterator<Item = &'a mut f64>,
    {
        purity_iterator.into_iter().enumerate().for_each(|(i, f)| {
            self.apply_layer(i % 2 == 1, periodic_boundaries);
            *f = self.get_mean_purity();
        });
    }

    /// Get the state of the system as a vector with real and imaginary values.
    pub fn get_state_raw(&self) -> Array3<Complex<f64>> {
        self.states.clone()
    }
}
