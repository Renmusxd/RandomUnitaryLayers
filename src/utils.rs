use num_complex::Complex;
use num_integer::binomial;
use numpy::ndarray::{s, Array3, Axis};
use rand::Rng;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::ops::{Add, Mul};

/// Calculates SUM in O(N) from an iterator of [t_n,...,t_0]
/// Where SUM = sum_base + sum_i f(t_i)*acc_base + sum_{i<j} f(t_i) g(t_j)
#[inline]
pub fn sum_s_sprime<F, G, T, V, IT>(ts: IT, mut sum_base: V, mut acc_base: V, f: F, g: G) -> V
where
    IT: IntoIterator<Item = T>,
    F: Fn(T) -> V,
    G: Fn(T) -> V,
    V: Add<Output = V> + Mul<Output = V> + Copy,
    T: Copy,
{
    for t in ts.into_iter() {
        sum_base = sum_base + f(t) * acc_base;
        acc_base = acc_base + g(t);
    }
    sum_base
}

/// Calculates SUM in O(N) from a iterators of [f_n,...,f_0] and [g_n,...g_0]
/// Where SUM = sum_base + sum_i f_i*acc_base + sum_{i<j} f_i g_j
#[inline]
pub fn sum_s_sprime_iterators<FS, GS, V>(f: FS, g: GS, sum_base: V, acc_base: V) -> V
where
    FS: IntoIterator<Item = V>,
    GS: IntoIterator<Item = V>,
    V: Add<Output = V> + Mul<Output = V> + Copy,
{
    let (sum, _) = f.into_iter().zip(g.into_iter()).fold(
        (sum_base, acc_base),
        |(mut sum, mut acc), (f_i, g_i)| {
            sum = sum + f_i * acc;
            acc = acc + g_i;
            (sum, acc)
        },
    );
    sum
}

/// Calculates SUM in O(N) from a iterator of [f_n,...,f_0]
/// Where SUM = sum_base + sum_i f_i*acc_base + sum_{i<j} f_i f_j
#[inline]
pub fn sum_s_sprime_iterator<FS, V>(f: FS, sum_base: V, acc_base: V) -> V
where
    FS: IntoIterator<Item = V>,
    V: Add<Output = V> + Mul<Output = V> + Copy,
{
    let (sum, _) = f
        .into_iter()
        .fold((sum_base, acc_base), |(mut sum, mut acc), f_i| {
            sum = sum + f_i * acc;
            acc = acc + f_i;
            (sum, acc)
        });
    sum
}

// From secion 2.3 of http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
/// Make a random 2x2 unitary matrix.
pub fn make_unitary<R: Rng>(mut rng: R) -> [Complex<f64>; 4] {
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

/// Apply a matrix `mat` to a vector `[a,b]`.
pub fn apply_matrix(a: &mut Complex<f64>, b: &mut Complex<f64>, mat: &[Complex<f64>; 4]) {
    let oa = *a;
    let ob = *b;
    *a = oa * mat[0] + ob * mat[1];
    *b = oa * mat[2] + ob * mat[3];
}

pub fn enumerate_rec<const N: usize>(
    acc: &mut Vec<SmallVec<[usize; N]>>,
    prefix: SmallVec<[usize; N]>,
    loop_num: usize,
    min_val: usize,
    max_val: usize,
) {
    if loop_num == 0 {
        for i in min_val..max_val {
            let mut state = prefix.clone();
            state.push(i);
            acc.push(state);
        }
    } else {
        for i in min_val..max_val - loop_num {
            let mut state = prefix.clone();
            state.push(i);
            enumerate_rec(acc, state, loop_num - 1, i + 1, max_val);
        }
    }
}

// https://codereview.stackexchange.com/questions/233872/writing-slice-compare-in-a-more-compact-way
pub fn compare_slices<T: Ord>(a: &[T], b: &[T]) -> Ordering {
    for (ai, bi) in a.iter().zip(b.iter()) {
        match ai.cmp(bi) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }

    /* if every single element was equal, compare length */
    a.len().cmp(&b.len())
}

pub fn index_of_state<T: AsRef<[usize]>>(
    state: &[usize],
    enumerated_states: &[T],
) -> Result<usize, usize> {
    enumerated_states.binary_search_by(|a| compare_slices(a.as_ref(), state))
}

/// Get the index delta matrix for states.
pub fn make_index_deltas(n_sites: usize, n_defects: usize) -> Vec<usize> {
    let mut index_deltas = vec![0; n_defects * n_sites];
    // Moving the mth defect from p to p+1, with M defects and N sites
    // causes an index change of (N-p-2) choose (M-m-1)
    // since there are N-(p-2) positions to place M-m-1 defects to the right of the moved one.

    for m in 0..n_defects {
        for p in m..n_sites - (n_defects - m) {
            let delta = binomial(n_sites - p - 2, n_defects - m - 1);
            index_deltas[p * n_defects + m] = delta;
        }
    }

    index_deltas
}

/// Compute the purity estimator of the state and return as a floating point value.
/// Assumes Tr(rho) = 1.0
/// Calculates D(sum_{s,s'} (-D)^{-delta_{s,s'}} P(s) P(s')
/// Can be rearranged into (D+1)sum_s P(s)^2 - (sum_s P(s))^2
/// That is was it calculated here, assuming sum_s P(s) = 1.0
pub fn get_purity_iterator<'a>(
    hilbert_d: usize,
    probs: &'a [f64],
    amps: &'a Array3<Complex<f64>>,
) -> impl IndexedParallelIterator<Item = f64> + 'a {
    // Iterate across experiments
    amps.axis_iter(Axis(0))
        .into_par_iter()
        .map(move |state| -> f64 {
            (hilbert_d + 1) as f64
                * state
                    .axis_iter(Axis(1)) // Iterate across states
                    .into_par_iter()
                    .map(|mixes| {
                        // Iterate across mixtures and their probabilities
                        mixes
                            .iter()
                            .zip(probs)
                            .map(|(amp, prob)| *prob * amp.norm_sqr())
                            .sum::<f64>()
                            .powi(2)
                    })
                    .sum::<f64>()
                - 1.0
        })
}

#[cfg(test)]
mod util_tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn enumerate_small_test() {
        let n = 6;
        let m = 3;
        let mut res = vec![];
        let prefix: SmallVec<[usize; 3]> = smallvec![];
        enumerate_rec(&mut res, prefix, m - 1, 0, n);

        let mut index = 0;
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    assert_eq!([i, j, k].as_slice(), res[index].as_slice());
                    index += 1;
                }
            }
        }
    }

    #[test]
    fn test_state_lookup() {
        let n = 6;
        let m = 3;
        let mut states = vec![];
        let prefix: SmallVec<[usize; 3]> = smallvec![];
        enumerate_rec(&mut states, prefix, m - 1, 0, n);

        let mut index = 0;
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    assert_eq!(index_of_state(&[i, j, k], &states), Ok(index));
                    index += 1;
                }
            }
        }
    }

    #[test]
    fn test_index_deltas() {
        let n_sites = 10;
        let n_defects = 4;
        let deltas = make_index_deltas(n_sites, n_defects);

        for m in 0..n_defects {
            for p in m..n_sites - (n_defects - m) {
                print!("{}\t", deltas[p * n_defects + m]);
            }
            println!();
        }
        println!("=====================");
        for m in 0..n_defects {
            for p in 0..n_sites {
                print!("{}\t", deltas[p * n_defects + m]);
            }
            println!();
        }
        println!();

        let mut states = vec![];
        let prefix: SmallVec<[usize; 4]> = smallvec![];
        enumerate_rec(&mut states, prefix, n_defects - 1, 0, n_sites);

        for i in 0..n_defects {
            // i is the one to change.
            for index in 0..states.len() {
                let state = &states[index];
                let mut new_state = state.clone();
                new_state[i] += 1;
                // if new_state is valid check its index
                let valid = new_state.windows(2).all(|slice| {
                    if let [a, b] = slice {
                        a < b && *a < n_sites && *b < n_sites
                    } else {
                        panic!();
                    }
                });
                if !valid {
                    continue;
                };
                // state is valid, check the index matches
                let delta = deltas[state[i] * n_defects + i];
                assert_eq!(new_state.as_slice(), states[index + delta].as_slice());
            }
        }
    }
}
