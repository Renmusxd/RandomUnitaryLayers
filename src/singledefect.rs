use crate::utils::*;
use num_complex::Complex;
use numpy::ndarray::{Array1, Array2, Array3, Axis};
use numpy::{c64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct SingleDefectState {
    pub d: usize,
    pub state: Vec<(f64, Vec<Complex<f64>>)>,
}

#[pymethods]
impl SingleDefectState {
    /// Construct a new instance.
    #[new]
    fn new(state: PyReadonlyArray1<c64>) -> PyResult<Self> {
        let state = vec![(1.0, state.as_slice()?.to_vec())];
        let d = state[0].1.len();
        Ok(Self { d, state })
    }

    #[staticmethod]
    fn new_mixed(state: Vec<(f64, PyReadonlyArray1<c64>)>) -> PyResult<Self> {
        let (state, d) =
            state
                .into_iter()
                .try_fold((vec![], None), |(mut acc, mut size), (w, s)| {
                    let s = s.as_slice()?.to_vec();
                    if let Some(d) = size {
                        if s.len() != d {
                            return Err(PyValueError::new_err("All states must be the same size"));
                        }
                    } else {
                        size = Some(s.len());
                    }
                    acc.push((w, s));
                    Ok((acc, size))
                })?;
        if let Some(d) = d {
            Ok(Self { d, state })
        } else {
            Err(PyValueError::new_err(
                "Mixed state must contain at least one state",
            ))
        }
    }

    pub fn apply_layer(&mut self, offset: bool, periodic_boundaries: Option<bool>) {
        let periodic_boundaries = periodic_boundaries.unwrap_or_default();
        let mut rng = rand::thread_rng();

        for (_, state) in &mut self.state {
            let state_subsystem = if offset {
                &mut state[1..]
            } else {
                state.as_mut()
            };
            // Seed randoms
            state_subsystem.par_chunks_exact_mut(2).for_each(|c| {
                let mut rng = rand::thread_rng();
                if let [a, b] = c {
                    let mat = make_unitary(&mut rng);
                    apply_matrix(a, b, &mat);
                } else {
                    unreachable!()
                }
            });
            // If periodic boundary conditions are on, and the offset is on, then do 0 and n-1
            if periodic_boundaries && offset {
                let mat = make_unitary(&mut rng);
                let n = state.len();
                let mut oa = state[n - 1];
                let mut ob = state[0];
                apply_matrix(&mut oa, &mut ob, &mat);
                state[n - 1] = oa;
                state[0] = ob;
            }
        }
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

    /// Compute the purity at each layer of the process and save to a numpy array.
    pub fn apply_alternative_layers_and_save_purity(
        &mut self,
        py: Python,
        n_layers: usize,
        periodic_boundaries: Option<bool>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let mut res = Array1::zeros((n_layers,));
        self.apply_alternative_layers_and_store_purity(res.iter_mut(), periodic_boundaries);
        Ok(res.into_pyarray(py).to_owned())
    }

    /// Get the state of the system.
    pub fn get_state(&self, py: Python) -> PyResult<Vec<(f64, Py<PyArray1<c64>>)>> {
        self.state.iter().try_fold(vec![], |mut acc, (w, v)| {
            let s = PyArray1::from_iter(py, v.iter().cloned()).to_owned();
            acc.push((*w, s));
            Ok(acc)
        })
    }

    /// Compute the purity of the state and return as a floating point value.
    pub fn get_purity(&self) -> f64 {
        if let [(_, state)] = self.state.as_slice() {
            // F = D ( sum_s p(s)^2 - D sum_{s!=s'} p(s)p(s') )
            // p(s) = |<s|u|i>|^2
            // internal state is u|i>

            // First term is sum over |psi(i)|^4
            let first_term = state.iter().map(|c| c.norm_sqr().powi(2)).sum::<f64>();

            // Second term is 2 sum_s p(s) ( sum_{s'>s} p(s') )
            // Can build the sum_{s'>s} backwards from the end to turn O(n^2) into O(n)
            let f = |x: Complex<f64>| -> f64 { x.norm_sqr() };
            let half_second_term = sum_s_sprime(state.iter().rev().cloned(), 0.0, 0.0, f, f);
            let second_term = 2.0 * half_second_term / (self.d as f64);

            first_term - second_term
        } else {
            // first_term = sum_s ( sum_alpha p_alpha |<s|U|alpha>|^2)^2
            let first_term = (0..self.d)
                .map(|s| {
                    self.state
                        .iter()
                        .map(|(w, state)| *w * state[s].norm_sqr())
                        .sum::<f64>()
                        .powi(2)
                })
                .sum::<f64>();

            let f = |s: usize| -> f64 {
                self.state
                    .iter()
                    .map(|(w, state)| *w * state[s].norm_sqr())
                    .sum::<f64>()
            };
            let half_second_term = sum_s_sprime((0..self.d).rev(), 0.0, 0.0, f, f);
            let second_term = 2.0 * half_second_term / (self.d as f64);

            first_term - second_term
        }
    }
}

impl SingleDefectState {
    /// Compute the purity at each layer of the process and save to a numpy array.
    pub fn apply_alternative_layers_and_store_purity<'a, It>(
        &mut self,
        purity_iterator: It,
        periodic_boundaries: Option<bool>,
    ) where
        It: IntoIterator<Item = &'a mut f64>,
    {
        purity_iterator.into_iter().enumerate().for_each(|(i, f)| {
            self.apply_layer(i % 2 == 1, periodic_boundaries);
            *f = self.get_purity();
        });
    }

    /// Get the state of the system as a vector with real and imaginary values.
    pub fn get_state_raw(&self) -> Vec<(f64, Vec<Complex<f64>>)> {
        self.state.clone()
    }
}

#[pyclass]
pub struct ThreadedSingleDefectStates {
    n: usize,
    mixture_probs: Vec<f64>,
    states: Vec<SingleDefectState>,
}

#[pymethods]
impl ThreadedSingleDefectStates {
    #[new]
    pub fn new(num_samples: usize, state: PyReadonlyArray1<c64>) -> PyResult<Self> {
        let state = state.as_slice()?.to_vec();
        Ok(Self {
            n: state.len(),
            mixture_probs: vec![1.0],
            states: (0..num_samples)
                .map(|_| SingleDefectState {
                    d: state.len(),
                    state: vec![(1.0, state.clone())],
                })
                .collect(),
        })
    }

    #[staticmethod]
    fn new_mixed(num_samples: usize, state: Vec<(f64, PyReadonlyArray1<c64>)>) -> PyResult<Self> {
        let mixture_probs = state.iter().map(|(w, _)| *w).collect();
        let d = state[0].1.len();
        Ok(Self {
            n: d,
            mixture_probs,
            states: (0..num_samples).try_fold(vec![], |mut acc, _| -> Result<_, PyErr> {
                let state = state
                    .iter()
                    .try_fold(
                        vec![],
                        |mut acc, (w, s)| -> Result<_, numpy::NotContiguousError> {
                            let s = s.as_slice()?;
                            acc.push((*w, s.to_vec()));
                            Ok(acc)
                        },
                    )
                    .map_err(|e| PyValueError::new_err(format!("Error making state: {:?}", e)))?;
                let sds = SingleDefectState { d, state };
                acc.push(sds);
                Ok(acc)
            })?,
        })
    }

    /// Apply alternating layers of random unitaries.
    pub fn apply_alternative_layers(&mut self, n_layers: usize, periodic_boundaries: Option<bool>) {
        self.states
            .par_iter_mut()
            .for_each(|s| s.apply_alternative_layers(n_layers, periodic_boundaries))
    }

    /// Compute the purity at each layer of the process and save to a numpy array.
    pub fn apply_alternative_layers_and_save_purity(
        &mut self,
        py: Python,
        n_layers: usize,
        periodic_boundaries: Option<bool>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let mut res = Array2::zeros((self.states.len(), n_layers));
        res.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(self.states.par_iter_mut())
            .for_each(|(mut row, state)| {
                state.apply_alternative_layers_and_store_purity(row.iter_mut(), periodic_boundaries)
            });
        Ok(res.into_pyarray(py).to_owned())
    }

    /// Compute the purity at each layer of the process and save to a numpy array.
    pub fn apply_alternative_layers_and_save_mean_purity(
        &mut self,
        py: Python,
        n_layers: usize,
        periodic_boundaries: Option<bool>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let mut res = Array1::zeros((n_layers,));
        res.iter_mut().enumerate().for_each(|(i, row)| {
            let offset = i % 2 == 1;
            *row = self
                .states
                .par_iter_mut()
                .map(|state| {
                    state.apply_layer(offset, periodic_boundaries);
                    state.get_purity()
                })
                .sum::<f64>()
                / (self.states.len() as f64);
        });
        Ok(res.into_pyarray(py).to_owned())
    }

    /// Get the state of the system.
    pub fn get_state(&self, py: Python) -> PyResult<Vec<(f64, Py<PyArray2<c64>>)>> {
        self.mixture_probs
            .iter()
            .cloned()
            .enumerate()
            .try_fold(vec![], |mut acc, (i, w)| {
                let mut res = Array2::zeros((self.states.len(), self.n));
                res.axis_iter_mut(Axis(0))
                    .zip(self.states.iter())
                    .for_each(|(mut row, state)| {
                        row.iter_mut()
                            .zip(state.state[i].1.iter().cloned())
                            .for_each(|(r, c)| *r = c)
                    });
                let res = res.into_pyarray(py).to_owned();
                acc.push((w, res));
                Ok(acc)
            })
    }

    /// Compute the purity of the state and return as a floating point value.
    pub fn get_purity(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        let mut res = Array1::zeros((self.states.len(),));
        res.iter_mut()
            .zip(self.states.iter())
            .for_each(|(f, state)| *f = state.get_purity());
        Ok(res.into_pyarray(py).to_owned())
    }
}
