use crate::utils::{apply_matrix, enumerate_rec, index_of_state, make_index_deltas};
use num_complex::Complex;
use num_traits::Zero;
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};

struct MixedState {
    state: Vec<(f64, Vec<Complex<f64>>)>,
}

struct StateDetails<const N: usize> {
    num_defects: usize,
    num_sites: usize,
    enumerated_states: Vec<SmallVec<[usize; N]>>,
    /// vector of index deltas for moving the nth defect from `p` to `p+1`.
    index_deltas: Vec<usize>,
    /// `v[i]` is a list of (index with occupation at i, index of defect occupying i).
    occupied_indices: Vec<Vec<(usize, usize)>>,
}

/// const generic tunes memory usage to optimize for num_defects <= N.
pub struct MultiDefectStateRaw<const N: usize> {
    experiment_states: Vec<MixedState>,
    details: StateDetails<N>,
}

impl<const N: usize> MultiDefectStateRaw<N> {
    fn new_pure(
        num_experiments: usize,
        state: Vec<(Vec<usize>, Complex<f64>)>,
        n_defects: usize,
        n_sites: usize,
    ) -> Self {
        Self::new_mixed(num_experiments, vec![(1.0, state)], n_defects, n_sites)
    }

    fn new_mixed(
        num_experiments: usize,
        state: Vec<(f64, Vec<(Vec<usize>, Complex<f64>)>)>,
        n_defects: usize,
        n_sites: usize,
    ) -> Self {
        // Check all states are valid.
        debug_assert!({
            state
                .iter()
                .flat_map(|(_, s)| {
                    s.iter().flat_map(|(s, _)| {
                        let correct_length = s.len() == n_defects;
                        let in_order = s
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
                        s.iter().cloned().map(move |x| {
                            let bounded_by_sites = x < n_sites;
                            correct_length && in_order && bounded_by_sites
                        })
                    })
                })
                .all(|x| x)
        });

        // Convert occupation representation into index representation.
        let enumerated_states = Self::enumerate_states(n_sites, n_defects);
        let full_state = state
            .into_iter()
            .map(|(w, s)| {
                let mut new_state = vec![Complex::zero(); enumerated_states.len()];
                s.into_iter().for_each(|(s, w)| {
                    let state_index = index_of_state(&s, &enumerated_states).unwrap();
                    new_state[state_index] = w;
                });
                (w, new_state)
            })
            .collect::<Vec<_>>();

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

        Self {
            experiment_states: (0..num_experiments)
                .map(|_| MixedState {
                    state: full_state.clone(),
                })
                .collect(),
            details: StateDetails {
                num_defects: n_defects,
                num_sites: n_sites,
                enumerated_states,
                index_deltas,
                occupied_indices,
            },
        }
    }

    /// Apply a matrix linking p and p+1, if both are occupied apply phase instead.
    fn apply_matrix(
        mixed_state: &mut MixedState,
        details: &StateDetails<N>,
        p: usize,
        exchange_mat: &[Complex<f64>; 4],
        adj_phase: f64,
    ) {
        mixed_state.state.par_iter_mut().for_each(|(_, s)| {
            // Go through all states with occupation on p or p+1
            // States with p and p+1 will appear twice.
            details.occupied_indices[p]
                .iter()
                .cloned()
                .map(|(index, defect)| (index, defect, true))
                .chain(
                    details.occupied_indices[p + 1]
                        .iter()
                        .cloned()
                        .map(|(index, defect)| (index, defect, false)),
                )
                .for_each(|(index, defect, on_p)| {
                    // index has an occupation on p or p+1 or both.
                    let state = &details.enumerated_states[index];
                    // Check for adjacent occupations first
                    let adjacent_occ = if on_p {
                        debug_assert_eq!(state[defect], p);
                        if defect < details.num_defects - 1 {
                            // If there can be another defect, check if its at p+1
                            state[defect + 1] == p + 1
                        } else {
                            // If there can't be another, then no adjacency.
                            false
                        }
                    } else {
                        debug_assert_eq!(state[defect], p + 1);
                        if defect > 0 {
                            // If this is not the first defect, check if a previous is on p
                            state[defect - 1] == p
                        } else {
                            // If it's the first, then no adjacency
                            false
                        }
                    };
                    if adjacent_occ {
                        if on_p {
                            // If adjacency then add a phase.
                            // Only apply phase once
                            s[index] *= Complex::from_polar(1.0, adj_phase);
                        }
                    } else {
                        // Otherwise mix between sites.
                        let delta = details.index_deltas[p * details.num_defects + defect];
                        let other_index = if on_p { index + delta } else { index - delta };
                        let (lower, upper) = match (index, other_index) {
                            (a, b) if a < b => (a, b),
                            (a, b) if a > b => (b, a),
                            _ => panic!(),
                        };
                        let mut a = s[lower];
                        let mut b = s[upper];
                        apply_matrix(&mut a, &mut b, exchange_mat);
                        s[lower] = a;
                        s[upper] = b;
                    }
                });
        });
    }

    fn enumerate_states(sites: usize, defects: usize) -> Vec<SmallVec<[usize; N]>> {
        let mut states = vec![];
        enumerate_rec(&mut states, smallvec![], defects - 1, 0, sites);
        states
    }
}
