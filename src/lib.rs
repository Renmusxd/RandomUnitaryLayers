use num_complex::Complex;
use num_traits::Zero;
use numpy::{c64, IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1};
use numpy::ndarray::Array3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;

/// Unlike the Lattice class this maintains a set of graphs with internal state.
#[pyclass]
pub struct State {
    state: Vec<Complex<f64>>,
}

#[pymethods]
impl State {
    /// Construct a new instance.
    #[new]
    fn new(state: PyReadonlyArray1<c64>) -> PyResult<Self> {
        let state = state.as_slice()?.to_vec();
        Ok(Self { state })
    }

    fn apply_layer(&mut self, offset: bool, periodic_boundaries: Option<bool>) {
        let periodic_boundaries = periodic_boundaries.unwrap_or_default();
        // Thetas
        let d = if periodic_boundaries { 0 } else { 1 };
        let mut rng = rand::thread_rng();
        let thetas = (0..self.state.len() / 2 - d)
            .map(|_| 2.0*rng.gen::<f64>()*std::f64::consts::PI)
            .collect::<Vec<f64>>();
        let last_theta = thetas[thetas.len() - 1];
        let sum_thetas: f64 = thetas.iter().sum();

        let state_subsystem = if offset {
            &mut self.state[1..]
        } else {
            &mut self.state
        };
        // Seed randoms
        state_subsystem
            .par_chunks_exact_mut(2)
            .zip(thetas.into_par_iter())
            .for_each(|(c, theta)| {
                let mut rng = rand::thread_rng();
                if let [a, b] = c {
                    let mat = make_phased_unitary(sum_thetas, theta, &mut rng);
                    apply_matrix(a, b, &mat);
                } else {
                    unreachable!()
                }
            });
        // If periodic boundary conditions are on, and the offset is on, then do 0 and n-1
        if periodic_boundaries && offset {
            let mat = make_phased_unitary(sum_thetas, last_theta, &mut rng);
            let n = self.state.len();
            let oa = self.state[n - 1];
            let ob = self.state[0];
            self.state[n - 1] = oa * mat[0] + ob * mat[1];
            self.state[0] = oa * mat[2] + ob * mat[3];
        }
    }

    fn make_unitaries(&self, py: Python, thetas: Vec<f64>) -> PyResult<Py<PyArray3<c64>>> {
        let mut rng = rand::thread_rng();
        let sum_thetas = thetas.iter().sum();
        let n_thetas = thetas.len();
        let mats = thetas.into_iter().map(|theta| {
            make_phased_unitary(sum_thetas, theta, &mut rng)
        });
        let mut res = Array3::zeros((n_thetas, 2, 2));
        res.axis_iter_mut(numpy::ndarray::Axis(0)).zip(mats).for_each(|(mut r, mat)| {
            r.iter_mut().zip(mat.into_iter()).for_each(|(r, m)| *r = m);
        });
        Ok(res.into_pyarray(py).to_owned())
    }

    fn apply_alternative_layers(&mut self, n_layers: usize, periodic_boundaries: Option<bool>) {
        for i in 0..n_layers {
            self.apply_layer(i % 2 == 1, periodic_boundaries);
        }
    }

    fn get_state(&self, py: Python) -> PyResult<Py<PyArray1<c64>>> {
        Ok(PyArray1::from_iter(py, self.state.iter().cloned()).to_owned())
    }
}

fn make_phased_unitary<R: Rng>(sum_thetas: f64, theta: f64, mut rng: R) -> [Complex<f64>; 4] {
    let mut mat = make_unitary(&mut rng);
    let phase = Complex::from_polar(1.0, sum_thetas - theta);
    mat[0] = mat[0] * phase;
    mat[1] = mat[1] * phase;
    mat[2] = mat[2] * phase;
    mat[3] = mat[3] * phase;
    mat
}

// From secion 2.3 of http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
fn make_unitary<R: Rng>(mut rng: R) -> [Complex<f64>; 4] {
    let two_pi = std::f64::consts::PI*2.0;
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
        ei_alpha * ei_psi * phi_c, ei_alpha * ei_chi * phi_s,
        - ei_alpha * ei_chi.conj() * phi_s, ei_alpha * ei_psi.conj() * phi_c
    ]
}

fn apply_matrix(a: &mut Complex<f64>, b: &mut Complex<f64>, mat: &[Complex<f64>; 4]) {
    let oa = *a;
    let ob = *b;
    *a = oa * mat[0] + ob * mat[1];
    *b = oa * mat[2] + ob * mat[3];
}

#[pymodule]
fn py_entropy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<State>()?;
    Ok(())
}
