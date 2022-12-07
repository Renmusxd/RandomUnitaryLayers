use crate::utils::{
    apply_matrix, enumerate_rec, get_purity_iterator, get_trace_rho, index_of_state,
    make_aa_bb_matrix, make_index_deltas, make_unitary, split_num_by_indices,
};
use ndarray::{s, ArrayViewMut1};
use num_complex::Complex;
use num_traits::Zero;
use numpy::ndarray::{Array1, Array2, Array3, Axis};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, ToPyArray,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};
use std::collections::HashSet;
use std::ops::IndexMut;

#[pyclass]
#[derive(Clone)]
pub struct MultidefectPureState {
    indices: Array2<usize>,
    amplitudes: Array1<Complex<f64>>,
}

#[pymethods]
impl MultidefectPureState {
    #[new]
    pub fn new(
        indices: PyReadonlyArray2<usize>,
        amplitudes: PyReadonlyArray1<Complex64>,
    ) -> PyResult<Self> {
        let total_prob = amplitudes
            .as_slice()
            .map_err(PyValueError::new_err)?
            .iter()
            .map(|a| a.norm_sqr())
            .sum::<f64>();
        if (total_prob - 1.0).abs() > 100. * f64::EPSILON {
            return Err(PyValueError::new_err(
                "Total amplitude squared must sum to 1.0",
            ));
        }
        Ok(Self {
            indices: indices.to_owned_array(),
            amplitudes: amplitudes.to_owned_array(),
        })
    }

    pub fn num_defects(&self) -> usize {
        self.indices.shape()[1]
    }

    pub fn non_zero_indices(&self) -> Vec<Vec<usize>> {
        self.indices
            .axis_iter(Axis(0))
            .zip(self.amplitudes.iter().copied())
            .filter_map(|(ss, amp)| {
                if amp.norm_sqr() > f64::EPSILON {
                    Some(ss.iter().copied().collect::<Vec<_>>())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn trace_out_site(
        &self,
        site: usize,
    ) -> ((f64, MultidefectPureState), (f64, MultidefectPureState)) {
        let prob_with_defect = self
            .indices
            .axis_iter(Axis(0))
            .zip(self.amplitudes.iter())
            .filter_map(|(state, amp)| {
                if state.iter().copied().any(|i| i.eq(&site)) {
                    Some(amp.norm_sqr())
                } else {
                    None
                }
            })
            .sum::<f64>();
        let prob_without_defect = 1.0 - prob_with_defect;

        let mut with_defect_amps = self.amplitudes.clone();
        let mut without_defect_amps = self.amplitudes.clone();
        self.indices
            .axis_iter(Axis(0))
            .zip(
                with_defect_amps
                    .iter_mut()
                    .zip(without_defect_amps.iter_mut()),
            )
            .for_each(|(state, (amp_with, amp_without))| {
                if state.iter().copied().any(|i| i.eq(&site)) {
                    *amp_with /= prob_with_defect.sqrt();
                    *amp_without = Complex::zero();
                } else {
                    *amp_without /= prob_without_defect.sqrt();
                    *amp_with = Complex::zero();
                }
            });
        if prob_with_defect.abs() < f64::EPSILON {
            with_defect_amps
                .iter_mut()
                .for_each(|amp| *amp = Complex::zero());
        }
        if prob_without_defect.abs() < f64::EPSILON {
            without_defect_amps
                .iter_mut()
                .for_each(|amp| *amp = Complex::zero());
        }

        (
            (
                prob_with_defect,
                Self {
                    amplitudes: with_defect_amps,
                    indices: self.indices.clone(),
                },
            ),
            (
                prob_without_defect,
                Self {
                    amplitudes: without_defect_amps,
                    indices: self.indices.clone(),
                },
            ),
        )
    }
}

#[pyclass]
pub struct MultiDefectState {
    mds: MultiDefectStateRaw<8>,
}

impl MultiDefectState {
    /// Compute the purity at each layer of the process and save to a numpy array.
    pub fn apply_alternative_layers_and_store_mean_purity<'a, It>(&mut self, purity_iterator: It)
    where
        It: IntoIterator<Item = &'a mut f64>,
    {
        purity_iterator.into_iter().enumerate().for_each(|(i, f)| {
            self.apply_layer(i % 2 == 1);
            *f = self.get_mean_purity();
        });
    }

    /// Compute the purity at each layer of the process and save to a numpy array.
    pub fn apply_alternative_layers_and_store_mean_purity_and_density<'a, 'b, It>(
        &mut self,
        iterator: It,
    ) where
        It: IntoIterator<Item = (&'b mut f64, ArrayViewMut1<'a, f64>)>,
    {
        self.mds.construct_density_lookup();
        let density_lookup = self.mds.details.density_hints.take().unwrap();

        iterator.into_iter().enumerate().for_each(|(i, (f, rho))| {
            self.apply_layer(i % 2 == 1);
            *f = self.get_mean_purity();

            self.mds.get_mean_density(rho);
        });
        self.mds.details.density_hints = Some(density_lookup);
    }

    /// Compute the purity at each layer of the process and save to a numpy array.
    /// Also compute the purity from tracing out the chosen sites.
    pub fn apply_alternative_layers_and_store_mean_purity_and_trace_purity<'a, 'b, It>(
        &mut self,
        iterator: It,
        trace_sites: &[usize],
    ) where
        It: IntoIterator<Item = (&'a mut f64, &'b mut f64)>,
    {
        // Tracing out k sites of a total L with M defects.
        // Cheat and use a hashmap
        // sum Px Py cxi cxj* cyk cyl* [Ai=Aj][Bj=Bk][Ak=Al][Bl=Bi]
        let aa_bb = self.mds.get_aa_bb_matrix(trace_sites);

        iterator.into_iter().enumerate().for_each(|(i, (f, tf))| {
            self.apply_layer(i % 2 == 1);
            *f = self.get_mean_purity();
            *tf = self.mds.get_substate_purity(&aa_bb);
        });
    }

    /// Compute the purity at each layer of the process and save to a numpy array.
    pub fn apply_alternative_layers_and_store_mean_purity_and_density_and_variance<'a, 'b, It>(
        &mut self,
        iterator: It,
        sectors: &[(usize, usize)],
    ) where
        It: IntoIterator<Item = (&'b mut f64, ArrayViewMut1<'a, f64>, ArrayViewMut1<'a, f64>)>,
    {
        self.mds.construct_density_lookup();
        let density_lookup = self.mds.details.density_hints.take().unwrap();

        iterator
            .into_iter()
            .enumerate()
            .for_each(|(i, (f, rho, var_rho))| {
                self.apply_layer(i % 2 == 1);
                *f = self.get_mean_purity();
                self.mds
                    .get_sector_density_and_variance(sectors, rho, var_rho);
            });
        self.mds.details.density_hints = Some(density_lookup);
    }
}

#[pymethods]
impl MultiDefectState {
    /// Construct a state with multiple defects
    /// `state`: PureState
    /// `n_sites`: number of sites on lattice.
    /// `num_experiments`: number of independent experiments to run.
    #[new]
    fn new(
        indices: Vec<Vec<usize>>,
        amplitudes: PyReadonlyArray1<Complex64>,
        n_sites: usize,
        num_experiments: Option<usize>,
        skip_float_checks: Option<bool>,
    ) -> PyResult<Self> {
        if indices.len() != amplitudes.shape()[0] {
            return Err(PyValueError::new_err(format!(
                "Expected {} amplitudes, found {}",
                indices.len(),
                amplitudes.shape()[0]
            )));
        }

        let mut sizes = indices.iter().map(|is| is.len()).collect::<Vec<_>>();
        sizes.dedup();
        if sizes.len() != 1 {
            return Err(PyValueError::new_err(
                "All states must be in same number sector.",
            ));
        }
        let n_defects = sizes[0];

        let state = indices
            .into_iter()
            .zip(amplitudes.to_owned_array().into_iter())
            .map(|(indx, c)| (indx, Complex::new(c.re, c.im)))
            .collect();
        let mds = MultiDefectStateRaw::<8>::new_pure(
            state,
            n_sites,
            n_defects,
            num_experiments,
            skip_float_checks,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { mds })
    }

    /// Construct a state with multiple defects
    /// `state`: PureState
    /// `n_sites`: number of sites on lattice.
    /// `num_experiments`: number of independent experiments to run.
    #[staticmethod]
    fn new_pure(
        state: MultidefectPureState,
        n_sites: usize,
        num_experiments: Option<usize>,
        skip_float_checks: Option<bool>,
    ) -> PyResult<Self> {
        let (indices, amplitudes) = (state.indices, state.amplitudes);
        let n_defects =
            if let ([indices_num_states, indices_num_defects], [amplitudes_num_states]) =
                (indices.shape(), amplitudes.shape())
            {
                if amplitudes_num_states != indices_num_states {
                    return Err(PyValueError::new_err(format!(
                        "Expected {} state amplitudes, found {}",
                        indices_num_states, amplitudes_num_states
                    )));
                }
                *indices_num_defects
            } else {
                return Err(PyValueError::new_err(format!(
                    "Invalid array shapes ({},{}) expecting (2,1)",
                    indices.shape().len(),
                    amplitudes.shape().len()
                )));
            };

        let state = indices
            .axis_iter(Axis(0))
            .zip(amplitudes.into_iter())
            .map(|(indx, c)| {
                (
                    indx.into_iter().cloned().collect(),
                    Complex::new(c.re, c.im),
                )
            })
            .collect();
        let mds = MultiDefectStateRaw::<8>::new_pure(
            state,
            n_sites,
            n_defects,
            num_experiments,
            skip_float_checks,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { mds })
    }

    /// Construct a state with multiple defects
    /// `details`: List of (probability, PureState)
    /// `n_sites`: number of sites on lattice.
    /// `n_defects`: number of defects in system.
    /// `num_experiments`: number of independent experiments to run.
    #[staticmethod]
    fn new_mixed_from_states(
        states: Vec<(f64, MultidefectPureState)>,
        n_sites: usize,
        num_experiments: Option<usize>,
        skip_float_checks: Option<bool>,
    ) -> PyResult<Self> {
        let mut defect_nums = states
            .iter()
            .map(|(_, s)| s.num_defects())
            .collect::<Vec<_>>();
        defect_nums.dedup();
        if defect_nums.len() != 1 {
            return Err(PyValueError::new_err(
                "All pure states must be in same number sector",
            ));
        }
        let n_defects = defect_nums[0];

        let state = states
            .into_iter()
            .map(|(p, s)| {
                (
                    p,
                    s.indices
                        .axis_iter(Axis(0))
                        .zip(s.amplitudes.into_iter())
                        .map(|(indices, amplitude)| (indices.to_vec(), amplitude))
                        .collect(),
                )
            })
            .collect();

        let mds = MultiDefectStateRaw::<8>::new_mixed(
            state,
            n_sites,
            n_defects,
            num_experiments,
            skip_float_checks,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { mds })
    }

    /// Construct a state with multiple defects
    /// `indices`: array of size [num mixed states, occupation strings, n_defects].
    /// `probs`: numpy array of probabilities for mixed states of size [num mixed states].
    /// `amplitudes`: numpy array of amplitudes of size [num mixed states, occupation strings].
    /// `n_sites`: number of sites on lattice.
    /// `n_defects`: number of defects in system.
    /// `num_experiments`: number of independent experiments to run.
    #[staticmethod]
    fn new_mixed(
        indices: PyReadonlyArray3<usize>,
        probs: PyReadonlyArray1<f64>,
        amplitudes: PyReadonlyArray2<Complex64>,
        n_sites: usize,
        n_defects: usize,
        num_experiments: Option<usize>,
        skip_float_checks: Option<bool>,
    ) -> PyResult<Self> {
        if let (
            [indices_num_mixed, indices_num_states, indices_num_defects],
            [probs_num_mixed],
            [amplitudes_num_mixed, amplitudes_num_states],
        ) = (indices.shape(), probs.shape(), amplitudes.shape())
        {
            if *indices_num_defects != n_defects {
                return Err(PyValueError::new_err(format!(
                    "Expected {} defects, found indices for {}",
                    n_defects, indices_num_defects
                )));
            }
            if probs_num_mixed != indices_num_mixed {
                return Err(PyValueError::new_err(format!(
                    "Expected {} mixed states, found indices for {}",
                    probs_num_mixed, indices_num_mixed
                )));
            }
            if probs_num_mixed != amplitudes_num_mixed {
                return Err(PyValueError::new_err(format!(
                    "Expected {} mixed states, found amplitudes for {}",
                    probs_num_mixed, indices_num_mixed
                )));
            }
            if amplitudes_num_states != indices_num_states {
                return Err(PyValueError::new_err(format!(
                    "Expected {} state amplitudes, found {}",
                    indices_num_states, amplitudes_num_states
                )));
            }
        } else {
            return Err(PyValueError::new_err(format!(
                "Invalid array shapes ({},{},{}) expecting (3,1,2)",
                indices.shape().len(),
                probs.shape().len(),
                amplitudes.shape().len()
            )));
        }

        let state = probs
            .to_owned_array()
            .iter()
            .cloned()
            .zip(amplitudes.to_owned_array().axis_iter(Axis(0)))
            .zip(indices.to_owned_array().axis_iter(Axis(0)))
            .map(|((prob, amplitudes), indices)| {
                let index_amplitudes = indices
                    .axis_iter(Axis(0))
                    .zip(amplitudes.iter().cloned())
                    .map(|(indx, c)| (indx.iter().cloned().collect(), Complex::new(c.re, c.im)))
                    .collect();
                (prob, index_amplitudes)
            })
            .collect();
        let mds = MultiDefectStateRaw::<8>::new_mixed(
            state,
            n_sites,
            n_defects,
            num_experiments,
            skip_float_checks,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { mds })
    }

    /// Get trace of rho
    fn get_trace_rho(&self) -> f64 {
        self.mds.details.trace_rho
    }

    /// Apply a single brick layer
    /// `offset`: if true bricks start at [1,2], else [0,1]
    pub fn apply_layer(&mut self, offset: bool) {
        self.mds.apply_brick_layer(offset, false)
    }

    /// Apply `n_layers` of alternating brick layers.
    pub fn apply_alternative_layers(&mut self, n_layers: usize) {
        for i in 0..n_layers {
            // Layers alternate between offset and not-offset.
            self.apply_layer(i % 2 == 1);
        }
    }

    /// Get density of defects by site.
    pub fn get_density(&mut self, py: Python) -> Py<PyArray1<f64>> {
        let mut rho = Array1::zeros((self.mds.details.num_sites,));
        self.mds.get_mean_density(rho.view_mut());
        rho.into_pyarray(py).to_owned()
    }

    /// Get the state of the system.
    pub fn get_state(&self, py: Python) -> (Py<PyArray1<f64>>, Py<PyArray3<Complex64>>) {
        let probs = self.mds.details.probs.to_pyarray(py).to_owned();
        let states = self.mds.experiment_states.to_pyarray(py).to_owned();
        (probs, states)
    }

    /// Get the list of N defect states in occupation representation for the current object.
    pub fn get_enumerated_states(&self, py: Python) -> Py<PyArray2<usize>> {
        let mut res = Array2::zeros((
            self.mds.details.enumerated_states.len(),
            self.mds.details.num_defects,
        ));
        res.axis_iter_mut(Axis(0))
            .zip(self.mds.details.enumerated_states.iter())
            .for_each(|(mut r, s)| {
                r.iter_mut().zip(s.into_iter()).for_each(|(r, s)| *r = *s);
            });
        res.into_pyarray(py).to_owned()
    }

    /// Get the list of N defect states in occupation representation.
    #[staticmethod]
    pub fn gen_enumerated_states(
        py: Python,
        num_sites: usize,
        num_defects: usize,
    ) -> Py<PyArray2<usize>> {
        let states = MultiDefectStateRaw::<8>::enumerate_states(num_sites, num_defects);
        let mut res = Array2::zeros((states.len(), num_defects));
        res.axis_iter_mut(Axis(0))
            .zip(states.into_iter())
            .for_each(|(mut r, s)| {
                r.iter_mut().zip(s.into_iter()).for_each(|(r, s)| *r = s);
            });
        res.into_pyarray(py).to_owned()
    }

    /// Get the mean purity across all experiments.
    pub fn get_mean_purity(&self) -> f64 {
        let purities = self.mds.get_purity_iterator();
        purities.sum::<f64>() / (self.mds.experiment_states.shape()[0] as f64)
    }

    /// Get the each experiments purity.
    pub fn get_all_purities(&self, py: Python) -> Py<PyArray1<f64>> {
        let purity_iterator = self.mds.get_purity_iterator();
        let mut purities = Array1::<f64>::zeros((self.mds.experiment_states.shape()[0],));
        purity_iterator
            .zip(purities.axis_iter_mut(Axis(0)).into_par_iter())
            .for_each(|(pa, mut pb)| {
                // Should just be 1.
                pb.iter_mut().for_each(|pb| *pb = pa);
            });
        purities.into_pyarray(py).to_owned()
    }

    /// Compute the purity at each of `n_layers` and save to a numpy array.
    pub fn apply_alternative_layers_and_save_mean_purity(
        &mut self,
        py: Python,
        n_layers: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let mut res = Array1::zeros((n_layers,));
        self.apply_alternative_layers_and_store_mean_purity(res.iter_mut());
        Ok(res.into_pyarray(py).to_owned())
    }

    /// Compute the purity and density at each of `n_layers` and save to a numpy array.
    pub fn apply_alternative_layers_and_save_mean_purity_and_density(
        &mut self,
        py: Python,
        n_layers: usize,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
        let mut res = Array1::zeros((n_layers,));
        let mut dens = Array2::zeros((n_layers, self.mds.details.num_sites));
        let iter = res.iter_mut().zip(dens.axis_iter_mut(Axis(0)));
        self.apply_alternative_layers_and_store_mean_purity_and_density(iter);
        Ok((
            res.into_pyarray(py).to_owned(),
            dens.into_pyarray(py).to_owned(),
        ))
    }

    /// Compute the purity and density at each of `n_layers` and save to a numpy array.
    /// Requires a list of sectors in which to calculate density and variances
    /// `sectors`: list of (start, end) - inclusive start and exclusive end.
    pub fn apply_alternative_layers_and_save_mean_purity_and_density_and_variances(
        &mut self,
        py: Python,
        n_layers: usize,
        sectors: Vec<(usize, usize)>,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
        let mut res = Array1::zeros((n_layers,));
        let mut dens = Array2::zeros((n_layers, sectors.len()));
        let mut vars = Array2::zeros((n_layers, sectors.len()));
        let iter = res
            .iter_mut()
            .zip(dens.axis_iter_mut(Axis(0)))
            .zip(vars.axis_iter_mut(Axis(0)))
            .map(|((i, a), b)| (i, a, b));
        self.apply_alternative_layers_and_store_mean_purity_and_density_and_variance(
            iter, &sectors,
        );
        Ok((
            res.into_pyarray(py).to_owned(),
            dens.into_pyarray(py).to_owned(),
            vars.into_pyarray(py).to_owned(),
        ))
    }

    /// Compute the purity and density at each of `n_layers` and save to a numpy array.
    pub fn apply_alternative_layers_and_save_mean_purity_and_trace_purity(
        &mut self,
        py: Python,
        n_layers: usize,
        sites: Vec<usize>,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let mut res = Array1::zeros((n_layers,));
        let mut half_res = Array1::zeros((n_layers,));
        let iter = res.iter_mut().zip(half_res.iter_mut());
        self.apply_alternative_layers_and_store_mean_purity_and_trace_purity(iter, &sites);
        Ok((
            res.into_pyarray(py).to_owned(),
            half_res.into_pyarray(py).to_owned(),
        ))
    }

    /// Get the density and variance for the sectors.
    pub fn get_sector_densities_and_variance(
        &mut self,
        py: Python,
        sectors: Vec<(usize, usize)>,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let mut dens = Array1::zeros((sectors.len(),));
        let mut vars = Array1::zeros((sectors.len(),));
        self.mds
            .get_sector_density_and_variance(&sectors, dens.view_mut(), vars.view_mut());
        Ok((
            dens.into_pyarray(py).to_owned(),
            vars.into_pyarray(py).to_owned(),
        ))
    }

    pub fn get_trace_purity(&mut self, trace_sites: Vec<usize>) -> f64 {
        let aa_bb = self.mds.get_aa_bb_matrix(&trace_sites);
        self.mds.get_substate_purity(&aa_bb)
    }
}

struct StateDetails<const N: usize> {
    num_defects: usize,
    num_sites: usize,
    probs: Array1<f64>,
    enumerated_states: Vec<SmallVec<[usize; N]>>,
    /// vector of index deltas for moving the nth defect from `p` to `p+1`.
    index_deltas: Vec<usize>,
    /// `v[i]` is a list of (index with occupation at i, index of defect occupying i).
    occupied_indices: Vec<Vec<(usize, usize)>>,
    trace_rho: f64,
    density_hints: Option<Vec<Vec<usize>>>,
}

/// const generic tunes memory usage to optimize for num_defects <= N.
pub struct MultiDefectStateRaw<const N: usize> {
    experiment_states: Array3<Complex<f64>>,
    details: StateDetails<N>,
}

impl<const N: usize> MultiDefectStateRaw<N> {
    fn new_pure(
        state: Vec<(Vec<usize>, Complex<f64>)>,
        n_sites: usize,
        n_defects: usize,
        num_experiments: Option<usize>,
        skip_float_checks: Option<bool>,
    ) -> Result<Self, String> {
        Self::new_mixed(
            vec![(1.0, state)],
            n_sites,
            n_defects,
            num_experiments,
            skip_float_checks,
        )
    }

    pub fn get_mean_density(&mut self, mut rho: ArrayViewMut1<f64>) {
        self.construct_density_lookup();
        let density_lookup = self.details.density_hints.take().unwrap();

        let probs = &self.details.probs;
        let ne = self.experiment_states.shape()[0];

        rho.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(site, mut rho)| {
                let rho = rho.get_mut([]).unwrap();
                *rho = density_lookup[site]
                    .iter()
                    .copied()
                    .map(|state_index| {
                        let experiments_and_mixtures =
                            self.experiment_states.slice(s![.., .., state_index]);
                        experiments_and_mixtures
                            .axis_iter(Axis(1))
                            .into_par_iter()
                            .zip(probs.axis_iter(Axis(0)).into_par_iter())
                            .map(|(states, prob)| {
                                prob.get([]).unwrap()
                                    * states.iter().map(|amp| amp.norm_sqr()).sum::<f64>()
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
                    / (ne as f64)
            });
        self.details.density_hints = Some(density_lookup);
    }

    fn get_substate_purity(&mut self, aa_bb: &HashSet<(usize, usize, usize, usize)>) -> f64 {
        self.experiment_states
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|amplitudes| {
                let mut sum = 0.0;
                for x in 0..self.details.probs.len() {
                    let probx = self.details.probs[x];
                    let coeffx = amplitudes.index_axis(Axis(0), x);
                    for y in 0..self.details.probs.len() {
                        let proby = self.details.probs[y];
                        let coeffy = amplitudes.index_axis(Axis(0), y);

                        let mut subsum = 0.0;
                        for (i, j, k, l) in aa_bb.iter().copied() {
                            let complex =
                                coeffx[i] * coeffx[j].conj() * coeffy[k] * coeffy[l].conj();
                            subsum += complex.re;
                        }
                        sum += probx * proby * subsum;
                    }
                }
                sum
            })
            .sum::<f64>()
            / (self.experiment_states.shape()[0] as f64)
    }

    pub fn get_aa_bb_matrix(
        &mut self,
        trace_sites: &[usize],
    ) -> HashSet<(usize, usize, usize, usize)> {
        let mut sorted_trace = trace_sites.to_vec();
        sorted_trace.sort_unstable();

        let ingroup_outgroup = self
            .details
            .enumerated_states
            .par_iter()
            .map(|state| {
                let mut num = 0usize;
                for s in state {
                    num |= 1 << s;
                }
                num
            })
            .map(|num| split_num_by_indices(num, &sorted_trace, self.details.num_sites))
            .collect::<Vec<_>>();

        make_aa_bb_matrix(&ingroup_outgroup)
    }

    pub fn get_sector_density_and_variance(
        &mut self,
        sectors: &[(usize, usize)],
        mut rho: ArrayViewMut1<f64>,
        mut var_rho: ArrayViewMut1<f64>,
    ) {
        self.construct_density_lookup();
        let density_lookup = self.details.density_hints.take().unwrap();
        let states = &self.experiment_states;
        let probs = &self.details.probs;
        let enumerated_states = &self.details.enumerated_states;
        let ne = states.shape()[0];

        rho.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(var_rho.axis_iter_mut(Axis(0)).into_par_iter())
            .zip(sectors.into_par_iter().copied())
            .for_each(|((mut rho, mut var_rho), (start, end))| {
                let (res_rho, res_var) = states
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|exp_states| {
                        let exp_rho = (start..end)
                            .map(|site| {
                                probs
                                    .iter()
                                    .zip(exp_states.axis_iter(Axis(0)))
                                    .map(|(p, ss)| {
                                        *p * density_lookup[site]
                                            .iter()
                                            .copied()
                                            .map(|state| ss[[state]].norm_sqr())
                                            .sum::<f64>()
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>();
                        let exp_var = exp_states
                            .axis_iter(Axis(0))
                            .zip(probs.iter().copied())
                            .map(|(states_for_mixture, prob)| {
                                prob * states_for_mixture
                                    .axis_iter(Axis(0))
                                    .zip(enumerated_states.iter())
                                    .map(|(amp, state)| {
                                        let in_region = state
                                            .iter()
                                            .cloned()
                                            .filter(|index| (*index >= start) && (*index < end))
                                            .count();
                                        amp.get([]).unwrap().norm_sqr()
                                            * ((in_region as f64) - exp_rho).powi(2)
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>();
                        (exp_rho, exp_var)
                    })
                    .reduce(
                        || (0., 0.),
                        |(arho, avar), (brho, bvar)| (arho + brho, avar + bvar),
                    );
                *rho.get_mut([]).unwrap() = res_rho / (ne as f64);
                *var_rho.get_mut([]).unwrap() = res_var / (ne as f64);
            });

        self.details.density_hints = Some(density_lookup);
    }

    /// Make a lookup table from site -> list of states.
    /// Speeds up density calculations.
    fn construct_density_lookup(&mut self) {
        if self.details.density_hints.is_none() {
            let mut v: Vec<Vec<usize>> = vec![vec![]; self.details.num_sites];
            v.iter_mut().enumerate().for_each(|(site, v)| {
                v.extend(
                    self.details
                        .enumerated_states
                        .iter()
                        .enumerate()
                        .filter_map(|(s_index, s_state)| {
                            if s_state.contains(&site) {
                                Some(s_index)
                            } else {
                                None
                            }
                        }),
                );
            });
            self.details.density_hints = Some(v)
        }
    }

    fn check_input(
        state: &Vec<(f64, Vec<(Vec<usize>, Complex<f64>)>)>,
        n_sites: usize,
        n_defects: usize,
        skip_float_checks: Option<bool>,
    ) -> Option<String> {
        let skip_float_checks = skip_float_checks.unwrap_or_default();
        let mut sum_p = 0.0;
        for (p, s) in state {
            sum_p += *p;
            let mut sum_amp = 0.0;
            for (indx, amp) in s {
                sum_amp += amp.norm_sqr();

                let correct_length = indx.len() == n_defects;
                if !correct_length {
                    return Some(format!(
                        "Expected state descriptors of length {}, found {:?}",
                        n_defects, indx
                    ));
                }

                let in_order = indx
                    .iter()
                    .cloned()
                    .map(|x| x as i64)
                    .try_fold(
                        -1,
                        |last, curr| {
                            if last < curr {
                                Ok(curr)
                            } else {
                                Err(())
                            }
                        },
                    )
                    .is_ok();
                if !in_order {
                    return Some(format!(
                        "Expected state descriptors to be ordered, found {:?}",
                        indx
                    ));
                }

                for x in indx {
                    if *x >= n_sites {
                        return Some(format!(
                            "Expected site occupations to be less than {}, found {:?}",
                            n_sites, indx
                        ));
                    }
                }
            }
            if !skip_float_checks && (sum_amp - 1.0).abs() > f64::EPSILON {
                return Some(format!(
                    "Expected amplitudes squared to sum to 1.0 found {}",
                    sum_amp
                ));
            }
        }
        if !skip_float_checks && (sum_p - 1.0).abs() > f64::EPSILON {
            return Some(format!(
                "Expected probabilities to sum to 1.0 found {}",
                sum_p
            ));
        }
        None
    }

    fn new_mixed(
        state: Vec<(f64, Vec<(Vec<usize>, Complex<f64>)>)>,
        n_sites: usize,
        n_defects: usize,
        num_experiments: Option<usize>,
        skip_float_checks: Option<bool>,
    ) -> Result<Self, String> {
        let num_experiments = num_experiments.unwrap_or(1);

        // Check the states are valid.
        let err = Self::check_input(&state, n_sites, n_defects, skip_float_checks);
        if let Some(err) = err {
            return Err(err);
        }

        // Convert occupation representation into index representation.
        let enumerated_states = Self::enumerate_states(n_sites, n_defects);

        let mut probs = Array1::zeros((state.len(),));
        probs
            .iter_mut()
            .zip(state.iter())
            .for_each(|(arr_w, (w, _))| {
                *arr_w = *w;
            });

        let mut full_state = Array3::zeros((num_experiments, state.len(), enumerated_states.len()));
        full_state
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut mixed_state| {
                mixed_state
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(state.par_iter())
                    .for_each(|(mut arr_state, (_, s))| {
                        s.iter().for_each(|(s, w)| {
                            let state_index = index_of_state(s, &enumerated_states).unwrap();
                            arr_state[state_index] = *w;
                        });
                    })
            });

        let index_deltas = make_index_deltas(n_sites, n_defects);

        let mut occupied_indices = vec![vec![]; n_sites];
        enumerated_states
            .iter()
            .enumerate()
            .for_each(|(index, state)| {
                state.iter().cloned().enumerate().for_each(|(m, p)| {
                    occupied_indices[p].push((index, m));
                })
            });

        let trace_rho = get_trace_rho(probs.as_slice().unwrap(), full_state.slice(s![0, .., ..]));

        Ok(Self {
            experiment_states: full_state,
            details: StateDetails {
                num_defects: n_defects,
                num_sites: n_sites,
                probs,
                enumerated_states,
                index_deltas,
                occupied_indices,
                trace_rho,
                density_hints: None,
            },
        })
    }

    fn apply_brick_layer(&mut self, offset: bool, periodic_boundaries: bool) {
        if periodic_boundaries {
            unimplemented!()
        }
        let offset = if offset { 1 } else { 0 };
        self.experiment_states
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut mixed_state| {
                let mut rng = rand::thread_rng();
                (0..(self.details.num_sites - offset) / 2).for_each(|i| {
                    let p = 2 * i + offset;
                    let mat = make_unitary(&mut rng);
                    let phase = 2f64 * std::f64::consts::PI * rng.gen::<f64>();
                    mixed_state
                        .axis_iter_mut(Axis(0))
                        .into_par_iter()
                        .for_each(|mut s| {
                            Self::apply_matrix(&mut s, &self.details, p, &mat, phase);
                        });
                });
            })
    }

    /// Apply a matrix linking p and p+1, if both are occupied apply phase instead.
    fn apply_matrix<S>(
        s: &mut S,
        details: &StateDetails<N>,
        p: usize,
        exchange_mat: &[Complex<f64>; 4],
        adj_phase: f64,
    ) where
        S: IndexMut<usize, Output = Complex<f64>> + ?Sized,
    {
        // Does not handle periodic boundary conditions.
        debug_assert!(p < details.num_sites - 1);

        // Go through all states with occupation on p or p+1
        // States with p and p+1 will appear twice.
        details.occupied_indices[p]
            .iter()
            .cloned()
            .for_each(|(index, defect)| {
                // index has an occupation on p or p+1 or both.
                let state = &details.enumerated_states[index];
                // Check for adjacent occupations first
                let adjacent_occ = {
                    debug_assert_eq!(state[defect], p);
                    if defect < details.num_defects - 1 {
                        // If there can be another defect, check if its at p+1
                        state[defect + 1] == p + 1
                    } else {
                        // If there can't be another, then no adjacency.
                        false
                    }
                };
                if adjacent_occ {
                    // If adjacency then add a phase.
                    // Only apply phase once
                    s[index] *= Complex::from_polar(1.0, adj_phase);
                } else {
                    // Otherwise mix between sites.
                    let delta = details.index_deltas[p * details.num_defects + defect];
                    let other_index = index + delta;
                    let mut a = s[index];
                    let mut b = s[other_index];
                    apply_matrix(&mut a, &mut b, exchange_mat);
                    s[index] = a;
                    s[other_index] = b;
                }
            });
    }

    pub fn enumerate_states(sites: usize, defects: usize) -> Vec<SmallVec<[usize; N]>> {
        let mut states = vec![];
        enumerate_rec(&mut states, smallvec![], defects - 1, 0, sites);
        states
    }

    /// Compute the purity estimator of the state and return as a floating point value.
    pub fn get_purity_iterator(&self) -> impl IndexedParallelIterator<Item = f64> + '_ {
        let hilbert_d = self.details.enumerated_states.len();
        let probs = self.details.probs.as_slice().unwrap();
        let amps = &self.experiment_states;
        get_purity_iterator(hilbert_d, probs, amps, self.details.trace_rho)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};
    use numpy::ndarray::s;

    #[test]
    fn test_apply_ident() -> Result<(), String> {
        let states = MultiDefectStateRaw::<1>::new_pure(
            vec![(vec![0], Complex::one())],
            2,
            1,
            Some(1),
            Some(true),
        )?;
        let old_state = states
            .experiment_states
            .slice(s![0, 0, ..])
            .iter()
            .cloned()
            .collect::<Vec<_>>();

        let mat = [
            Complex::one(),
            Complex::zero(),
            Complex::zero(),
            Complex::one(),
        ];
        let mut new_state = old_state.clone();
        MultiDefectStateRaw::apply_matrix(new_state.as_mut_slice(), &states.details, 0, &mat, 0.0);
        assert_eq!(old_state, new_state);
        Ok(())
    }

    #[test]
    fn test_apply_flip() -> Result<(), String> {
        let states = MultiDefectStateRaw::<1>::new_pure(
            vec![(vec![0], Complex::one())],
            2,
            1,
            Some(1),
            Some(true),
        )?;
        let old_state = states
            .experiment_states
            .slice(s![0, 0, ..])
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mat = [
            Complex::zero(),
            Complex::one(),
            Complex::one(),
            Complex::zero(),
        ];
        let mut new_state = old_state.clone();
        MultiDefectStateRaw::apply_matrix(new_state.as_mut_slice(), &states.details, 0, &mat, 0.0);
        new_state.reverse();
        assert_eq!(old_state, new_state);
        Ok(())
    }

    #[test]
    fn test_apply_flip_three() -> Result<(), String> {
        let states = MultiDefectStateRaw::<1>::new_pure(
            vec![(vec![0], Complex::one())],
            3,
            1,
            Some(1),
            Some(true),
        )?;
        let old_state = states
            .experiment_states
            .slice(s![0, 0, ..])
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mat = [
            Complex::zero(),
            Complex::one(),
            Complex::one(),
            Complex::zero(),
        ];

        let mut new_state = old_state.clone();
        MultiDefectStateRaw::apply_matrix(new_state.as_mut_slice(), &states.details, 0, &mat, 0.0);
        MultiDefectStateRaw::apply_matrix(new_state.as_mut_slice(), &states.details, 1, &mat, 0.0);

        new_state.reverse();
        assert_eq!(old_state, new_state);
        Ok(())
    }

    #[test]
    fn test_apply_flip_multi() -> Result<(), String> {
        let states = MultiDefectStateRaw::<1>::new_pure(
            vec![(vec![0, 2], Complex::one())],
            3,
            2,
            Some(1),
            Some(true),
        )?;
        let old_state = states
            .experiment_states
            .slice(s![0, 0, ..])
            .iter()
            .cloned()
            .collect::<Vec<_>>();

        let mat = [
            Complex::zero(),
            Complex::one(),
            Complex::one(),
            Complex::zero(),
        ];
        let mut new_state = old_state.clone();
        // Takes |02> to |12>
        // Takes i=1 to i=2
        MultiDefectStateRaw::apply_matrix(new_state.as_mut_slice(), &states.details, 0, &mat, 0.0);
        // i=1
        new_state.rotate_left(1);

        assert_eq!(old_state, new_state);
        Ok(())
    }
}
