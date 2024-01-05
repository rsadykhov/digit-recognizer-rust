use rand::distributions::{Distribution, Uniform};



pub fn init_matrix(length: usize, width: usize) -> Vec<Vec<f64>> {
    let dist_range = Uniform::from(-0.5..0.5);
    let mut rng = rand::thread_rng();
    let mut matrix = Vec::<Vec<f64>>::new();
    for _ in 0..length {
        let mut layer = Vec::<f64>::new();
        for _ in 0..width {
            layer.push(dist_range.sample(&mut rng));
        }
        matrix.push(layer);
    }
    matrix
}



pub fn init_vector(length: usize) -> Vec<f64> {
    let dist_range = Uniform::from(-0.5..0.5);
    let mut rng = rand::thread_rng();
    let mut vector = Vec::<f64>::new();
    for _ in 0..length {
        vector.push(dist_range.sample(&mut rng));
    }
    vector
}



pub fn transpose<T>(v: &Vec<Vec<T>>) -> Vec<Vec<T>> where T: Clone, {
    assert!(!v.is_empty());
    (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect()
}



pub fn dot_product(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    if v1.len()!=v2.len() {panic!("Vectors must have the same length");}
    let mut sum: f64 = 0.0;
    for i in 0..v1.len() {
        sum = sum + v1[i]*v2[i];
    }
    sum
}



pub fn matrix_product(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if a[0].len()!=b.len() {panic!("Mismatching dimensions of the matrices");}
    let b_transpose = transpose(b);
    let mut result = Vec::<Vec<f64>>::new();
    for i in 0..a.len() {
        let mut layer = Vec::<f64>::new();
        for j in 0..b[0].len() {
            layer.push(dot_product(&a[i], &b_transpose[j]));
        }
        result.push(layer);
    }
    result
}



pub fn matrix_pointwise_mult(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if a.len()!=b.len() || a[0].len()!=b[0].len() {panic!("Mismatching dimensions of the matrices")}
    let mut m = Vec::<Vec<f64>>::new();
    for i in 0..a.len() {
        let mut layer = Vec::<f64>::new();
        for j in 0..a[0].len() {
            layer.push(a[i][j] * b[i][j]);
        }
        m.push(layer);
    }
    m
}



pub fn matrix_subtraction(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if a.len()!=b.len() || a[0].len()!=b[0].len() {panic!("Mismatching dimensions of the matrices");}
    let mut m = Vec::<Vec<f64>>::new();
    for i in 0..a.len() {
        let mut layer = Vec::<f64>::new();
        for j in 0..a[0].len() {
            layer.push(a[i][j] - b[i][j]);
        }
        m.push(layer)
    }
    m
}



pub fn matrix_scalar_mult(matrix: &Vec<Vec<f64>>, scalar: &f64) -> Vec<Vec<f64>> {
    let mut m = Vec::<Vec<f64>>::new();
    for i in 0..matrix.len() {
        let mut layer = Vec::<f64>::new();
        for j in 0..matrix[0].len() {
            layer.push(matrix[i][j] * scalar);
        }
        m.push(layer);
    }
    m
}



pub fn increment_matrtix_by_vector(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<Vec<f64>> {
    if matrix.len()!=vector.len() {panic!("Matrix and vector have mismatching dimensions")}
    let mut m = Vec::<Vec<f64>>::new();
    for i in 0..matrix.len() {
        let mut layer = Vec::<f64>::new();
        for j in 0..matrix[0].len() {
            layer.push(matrix[i][j]+vector[i]);
        }
        m.push(layer);
    }
    m
}