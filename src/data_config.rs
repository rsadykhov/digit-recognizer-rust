use std::io;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use serde::{Serialize, Deserialize};
use serde_json;
use csv::Reader;
use crate::utils::{init_matrix, init_vector, transpose};



pub struct Execution {
    pub train: bool,
    pub path: String,
}



pub struct NNData {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<u16>
}



#[derive(Serialize, Deserialize)]
pub struct NNParams {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
}



pub fn user_prompt() -> Execution {
    println!("Use 'train' command to train the model.\nUse 'test' to run the trained model.", );
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input).expect("User input is undefined");
    let user_input = user_input.trim().to_lowercase().to_string();
    let execution: Execution;
    if user_input == String::from("test") {
        // train.csv can be replaced with test.csv, but test.csv does not have labeled data
        execution = Execution{ train: false, path: String::from("./static/train.csv") };
    } else if user_input == String::from("train") {
        execution = Execution{ train: true, path: String::from("./static/train.csv") };
    } else {
        panic!("Undefined command");
    }
    execution
}



pub fn get_data(execution: Execution) -> Result<NNData, Box<dyn Error>> {
    let mut reader = Reader::from_path(execution.path).expect("File path is incorrect");
    // NNData struct agruments
    let mut x = Vec::<Vec<f64>>::new();
    let mut y = Vec::<u16>::new();
    for line in reader.records() {
        match line {
            Ok(string_record) => {
                let mut item_counter = 0;
                let mut x_layer = Vec::<f64>::new();
                for item in string_record.iter() {
                    // Extract dataset
                    let value: f64 = match item.parse() {
                        Ok(num) => num,
                        Err(_) => panic!("Value from CSV cannot be converted to f64"),
                    };
                    // Check if value is the y data
                    // If using test.csv, add "execution.train &&" to the if statement
                    if item_counter<1 {
                        y.push(value as u16);
                    } else {
                        x_layer.push(value/255.0);
                    }
                    item_counter = item_counter + 1;
                };
                x.push(x_layer);
            },
            Err(_) => ()
        }
    }
    let result = NNData {
        x: transpose(&x),
        y: y,
    };
    Ok(result)
}



pub fn init_params(w1_width: usize, w2_width: usize) -> NNParams {
    let w1 = init_matrix(10, w1_width);
    let b1 = init_vector(10);
    // let b1 = init_matrix(10, 1);
    let w2 = init_matrix(10, w2_width);
    let b2 = init_vector(10);
    // let b2 = init_matrix(10, 1);
    NNParams { w1: w1, b1: b1, w2: w2, b2: b2 }
}



pub fn save_to_json(w1: Vec<Vec<f64>>, b1: Vec<f64>, w2: Vec<Vec<f64>>, b2: Vec<f64>) -> () {
    let nn_params = NNParams {w1, b1, w2, b2};
    let file = File::create("./static/params.json").expect("Failed to create new JSON file");
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &nn_params).expect("Failed to save to JSON file");
    let _ = writer.flush();
}



pub fn retrieve_from_json() -> NNParams {
    let file = File::open("./static/params.json").expect("Model was never trained. Train the model to obtain parameters");
    let nn_params: NNParams = serde_json::from_reader(file).expect("Failed to parse JSON file");
    nn_params
}



pub fn user_n_iterations() -> u16 {
    let iterations: u16;
    loop {
        println!("Provide number of iterations for model to train for:");
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).expect("User input is undefined");
        match user_input.trim().parse() {
            Ok(num) => {
                iterations = num;
                break
            },
            Err(_) => continue,
        };
    }
    iterations
}



pub fn user_test_prompt(x_width: usize) -> usize {
    let index: usize;
    loop {
        println!("Choose index of x in the range [0, {}]", x_width);
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).expect("User input is undefined");
        match user_input.trim().parse() {
            Ok(num) => {
                if x_width<num {
                    continue
                } else {
                    index = num;
                    break
                }
            },
            Err(_) => continue,
        };
    }
    index
}