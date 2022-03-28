use num_complex::Complex;
use num_traits::{One, Zero};
use py_entropy::SingleDefectState;

fn main() {
    let mut state = vec![Complex::zero(); 10];
    state[0] = Complex::one();
    let mut s = SingleDefectState { state };

    print_prob(&s.get_state_raw());
    for i in 0..100 {
        s.apply_layer(i % 2 == 1, Some(true));
        print_prob(&s.get_state_raw());
    }
}

fn print_prob(s: &[(f64, f64)]) {
    s.into_iter()
        .for_each(|(a, b)| print!("{:.3}\t", a.powi(2) + b.powi(2)));
    println!()
}
