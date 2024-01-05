use crate::{utils::{transpose, increment_matrtix_by_vector, matrix_product, matrix_subtraction, matrix_scalar_mult, matrix_pointwise_mult}, data_config, neural_network::metrics::{get_accuracy, get_predictions}};



fn relu(z: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if z.is_empty() {panic!("Matrix is not defined")}
    let mut relu_z = Vec::<Vec<f64>>::new();
    for i in 0..z.len() {
        let mut layer = Vec::<f64>::new();
        for j in 0..z[0].len() {
            if z[i][j]<0.0 {
                layer.push(0.0);
            } else {
                layer.push(z[i][j]);
            }
        }
        relu_z.push(layer);
    };
    relu_z
}



fn derivative_relu(z: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut d_relu = Vec::<Vec<f64>>::new();
    for i in 0..z.len() {
        let mut layer = Vec::<f64>::new();
        for j in 0..z[0].len() {
            if z[i][j]<0.0 {
                layer.push(0.0);
            } else {
                layer.push(1.0);
            }
        }
        d_relu.push(layer);
    }
    d_relu
}



fn softmax(z: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut a = Vec::<Vec<f64>>::new();
    for row in z.iter() {
        let mut layer = Vec::<f64>::new();
        for entry in row.iter() {
            layer.push(entry.exp());
        }
        a.push(layer);
    }
    let mut renormalization = Vec::<f64>::new();
    for row in transpose(&a).iter() {
        renormalization.push(row.iter().sum());
    }
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            a[i][j] = a[i][j] / renormalization[j];
        }
    }
    a
}



pub fn forward_propagation(w1: &Vec<Vec<f64>>, b1: &Vec<f64>, w2: &Vec<Vec<f64>>, b2: &Vec<f64>,
    x: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let z1 = increment_matrtix_by_vector(&matrix_product(w1, x), b1);
    let a1 = relu(&z1);
    let z2 = increment_matrtix_by_vector(&matrix_product(w2, &a1), b2);
    let a2 = softmax(&z2);
    (z1, a1, z2, a2)
}



fn one_hot(y: &Vec<u16>) -> Vec<Vec<f64>> {
    let mut one_hot_y = Vec::<Vec<f64>>::new();
    for i in 0..y.len() {
        let mut layer = Vec::<f64>::new();
        for j in 0..10 {
            if j as u16==y[i] {
                layer.push(1.0);
            } else {
                layer.push(0.0);
            }
        }
        one_hot_y.push(layer);
    }
    transpose(&one_hot_y)
}



pub fn backward_propagation(z1: &Vec<Vec<f64>>, a1: &Vec<Vec<f64>>, a2: &Vec<Vec<f64>>, w2: &Vec<Vec<f64>>,
    x: &Vec<Vec<f64>>, y: &Vec<u16>) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let m = y.len() as f64;
    // Second layer
    let one_hot_y = one_hot(&y);
    let dz2 = matrix_subtraction(a2, &one_hot_y);
    let mut dw2 = matrix_product(&dz2, &transpose(a1));
    dw2 = matrix_scalar_mult(&dw2, &(1.0/m));
    let mut db2 = Vec::<f64>::new();
    for element in transpose(&dz2) {
        db2.push(element.iter().sum());
    }
    // First layer
    let dz1 = matrix_pointwise_mult(&matrix_product(&transpose(w2), &dz2), &derivative_relu(z1));
    let mut dw1 = matrix_product(&dz1, &transpose(x));
    dw1 = matrix_scalar_mult(&dw1, &(1.0/m));
    let mut db1 = Vec::<f64>::new();
    for element in transpose(&dz1) {
        db1.push(element.iter().sum());
    }
    (dw1, db1, dw2, db2)
}



pub fn update_params(mut w1: Vec<Vec<f64>>, mut b1: Vec<f64>, mut w2: Vec<Vec<f64>>, mut b2: Vec<f64>, dw1: Vec<Vec<f64>>, db1: Vec<f64>,
    dw2: Vec<Vec<f64>>, db2: Vec<f64>, alpha: &f64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    w1 = matrix_subtraction(&w1, &matrix_scalar_mult(&dw1, &alpha));
    for i in 0..b1.len() {
        b1[i] = b1[i] - alpha*db1[i]
    }
    w2 = matrix_subtraction(&w2, &matrix_scalar_mult(&dw2, &alpha));
    for i in 0..b2.len() {
        b2[i] = b2[i] - alpha*db2[i]
    }
    (w1, b1, w2, b2)
}


pub fn gradient_descent(x: Vec<Vec<f64>>, y: Vec<u16>, iterations: u16, alpha: f64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let nn_params = data_config::init_params(x.len(), 10);
    let mut w1 = nn_params.w1;
    let mut b1 = nn_params.b1;
    let mut w2 = nn_params.w2;
    let mut b2 = nn_params.b2;
    for i in 0..iterations {
        let (z1, a1, _z2, a2) = forward_propagation(&w1, &b1, &w2, &b2, &x);
        let (dw1, db1, dw2, db2) = backward_propagation(&z1, &a1, &a2, &w2, &x, &y);
        (w1, b1, w2, b2) = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, &alpha);
        if i%1==0 {
            println!("Iteration: {}", i);
            println!("Accuracy: {}%", get_accuracy(&get_predictions(&a2), &y)*100.0);
        }
    }
    (w1, b1, w2, b2)
}