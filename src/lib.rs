use num_complex::Complex;
use numpy::ndarray::{Array1, Array2, Array3, Axis};
use numpy::{c64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct SingleDefectState {
    pub state: Vec<Complex<f64>>,
}

#[pymethods]
impl SingleDefectState {
    /// Construct a new instance.
    #[new]
    fn new(state: PyReadonlyArray1<c64>) -> PyResult<Self> {
        let state = state.as_slice()?.to_vec();
        Ok(Self { state })
    }

    pub fn apply_layer(&mut self, offset: bool, periodic_boundaries: Option<bool>) {
        let periodic_boundaries = periodic_boundaries.unwrap_or_default();
        let mut rng = rand::thread_rng();

        let state_subsystem = if offset {
            &mut self.state[1..]
        } else {
            &mut self.state
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
            let n = self.state.len();
            let oa = self.state[n - 1];
            let ob = self.state[0];
            self.state[n - 1] = oa * mat[0] + ob * mat[1];
            self.state[0] = oa * mat[2] + ob * mat[3];
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
    pub fn get_state(&self, py: Python) -> PyResult<Py<PyArray1<c64>>> {
        Ok(PyArray1::from_iter(py, self.state.iter().cloned()).to_owned())
    }

    /// Compute the purity of the state and return as a floating point value.
    pub fn get_purity(&self) -> f64 {
        // F = D ( sum_s p(s)^2 - D sum_{s!=s'} p(s)p(s') )
        // p(s) = |<s|u|i>|^2
        // internal state is u|i>
        let d = self.state.len();

        // First term is sum over |psi(i)|^4
        let first_term = self.state.iter().map(|c| c.norm_sqr().powi(2)).sum::<f64>();

        // Second term is 2 sum_s p(s) ( sum_{s'>s} p(s') )
        // Can build the sum_{s'>s} backwards from the end to turn O(n^2) into O(n)
        let (half_second_term, _) =
            self.state
                .iter()
                .rev()
                .cloned()
                .fold((0.0, 0.0), |(acc, sum_ss), c| {
                    // acc should be sum_{s>i} v[s] sum_{s'>s} v[s']
                    // sum_ss should be sum_{s'>i} v[s']
                    let c_ns = c.norm_sqr();
                    (acc + c_ns * sum_ss, sum_ss + c_ns)
                });
        let second_term = 2.0 * half_second_term / (d as f64);

        first_term - second_term
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
    pub fn get_state_raw(&self) -> Vec<Complex<f64>> {
        self.state.clone()
    }
}

// From secion 2.3 of http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
fn make_unitary<R: Rng>(mut rng: R) -> [Complex<f64>; 4] {
    let two_pi = std::f64::consts::PI * 2.0;
    let alpha: f64 = rng.gen::<f64>() * two_pi;
    let psi: f64 = rng.gen::<f64>() * two_pi;
    let chi: f64 = rng.gen::<f64>() * two_pi;
    let xi: f64 = rng.gen::<f64>();
    let phi = xi.sqrt().asin();

    let ei_alpha = Complex::from_polar(1.0, alpha);
    let ei_psi = Complex::from_polar(1.0, psi);
    let ei_chi = Complex::from_polar(1.0, chi);
    let (phi_s, phi_c) = phi.sin_cos();
    [
        ei_alpha * ei_psi * phi_c,
        ei_alpha * ei_chi * phi_s,
        -ei_alpha * ei_chi.conj() * phi_s,
        ei_alpha * ei_psi.conj() * phi_c,
    ]
}

fn apply_matrix(a: &mut Complex<f64>, b: &mut Complex<f64>, mat: &[Complex<f64>; 4]) {
    let oa = *a;
    let ob = *b;
    *a = oa * mat[0] + ob * mat[1];
    *b = oa * mat[2] + ob * mat[3];
}

#[pyclass]
pub struct ThreadedSingleDefectStates {
    n: usize,
    states: Vec<SingleDefectState>,
}

#[pymethods]
impl ThreadedSingleDefectStates {
    #[new]
    pub fn new(num_samples: usize, state: PyReadonlyArray1<c64>) -> PyResult<Self> {
        let state = state.as_slice()?.to_vec();
        Ok(Self {
            n: state.len(),
            states: (0..num_samples)
                .map(|_| SingleDefectState {
                    state: state.clone(),
                })
                .collect(),
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
    pub fn get_state(&self, py: Python) -> PyResult<Py<PyArray2<c64>>> {
        let mut res = Array2::zeros((self.states.len(), self.n));
        res.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(self.states.par_iter())
            .for_each(|(mut row, state)| {
                row.iter_mut()
                    .zip(state.get_state_raw().into_iter())
                    .for_each(|(r, c)| *r = c)
            });
        Ok(res.into_pyarray(py).to_owned())
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

#[pymodule]
fn py_entropy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SingleDefectState>()?;
    m.add_class::<ThreadedSingleDefectStates>()?;
    Ok(())
}
