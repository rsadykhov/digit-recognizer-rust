use crate::utils::transpose;
use super::components::forward_propagation;



pub fn get_predictions(a2: &Vec<Vec<f64>>) -> Vec<u16> {
    let mut argmax = Vec::<u16>::new();
    let a2_t = transpose(a2);
    for i in 0..a2_t.len() {
        let mut max_value_index: u16 = 0;
        let mut max_value: f64 = 0.0;
        for j in 0..a2_t[0].len() {
            if a2_t[i][j]>max_value {
                max_value = a2_t[i][j];
                max_value_index = j as u16;
            }
        }
        argmax.push(max_value_index);
    }
    argmax
}



pub fn get_accuracy(predictions: &Vec<u16>, y: &Vec<u16>) -> f64 {
    if predictions.len()!=y.len() {panic!("Mismatching vector lengths")}
    let mut n_correct: f64 = 0.0;
    for i in 0..predictions.len() {
        if predictions[i]==y[i] {
            n_correct = n_correct + 1.0;
        }
    }
    n_correct/(y.len() as f64)
}



pub fn make_predictions(x: &Vec<Vec<f64>>, w1: &Vec<Vec<f64>>, b1: &Vec<f64>, w2: &Vec<Vec<f64>>, b2: &Vec<f64>) -> Vec<u16> {
    let (_, _, _, a2) = forward_propagation(w1, b1, w2, b2, x);
    get_predictions(&a2)
}



pub fn test_prediction(x_sample: &Vec<Vec<f64>>, y_sample: &Vec<u16>, w1: &Vec<Vec<f64>>, b1: &Vec<f64>, w2: &Vec<Vec<f64>>, b2: &Vec<f64>) -> () {
    let prediction = make_predictions(x_sample, w1, b1, w2, b2);
    for i in 0..prediction.len() {
        println!("Prediction: {}", prediction[i]);
        println!("Actual Value: {}", y_sample[i]);
    }
}