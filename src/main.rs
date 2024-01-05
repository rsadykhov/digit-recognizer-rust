mod utils;
mod data_config;
mod neural_network;
use neural_network::components::gradient_descent;



fn main() {
    let execution = data_config::user_prompt();
    let train = execution.train;
    let data = data_config::get_data(execution);
    let nn_data: data_config::NNData;
    match data {
        Ok(nn_data_) => { nn_data = nn_data_; },
        Err(_) => panic!("Error during data extraction"),
    }
    if train {
        let iterations = data_config::user_n_iterations();
        // Train model
        let (w1, b1, w2, b2) = gradient_descent(nn_data.x, nn_data.y, iterations, 0.1);
        // Save model parameters
        data_config::save_to_json(w1, b1, w2, b2);
    } else {
        // Get model parameters from JSON file
        let nn_params = data_config::retrieve_from_json();
        // Prompt user to choose the index of sample x
        let index = data_config::user_test_prompt(nn_data.x[0].len());
        // Sample x
        let x_sample = utils::transpose(&utils::transpose(&nn_data.x)[index..index+1].to_vec());
        let y_sample = nn_data.y[index..index+1].to_vec();
        // Run make_predictions and print the result
        neural_network::metrics::test_prediction(&x_sample, &y_sample, &nn_params.w1, &nn_params.b1, &nn_params.w2, &nn_params.b2);
    }
}
