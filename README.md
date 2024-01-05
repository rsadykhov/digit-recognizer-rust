# digit-recognizer-rust
Rust implementation of an AI digit recognizer.

This is a minimalistic AI aimed at recognizing hand written digits and converting them to u16 digits. The mathematical functions (e.g., dot product, pointwise multiplication) in the code are not optimized so the training of the model may take some time, in particular with high number of iterations. However, once the model is trained it will save the parameters locally in a JSON file so they can be reused when running the model in the future.

# Credits
This project is inspired by Samsong Zhang and his video on building neural networks from scratch in Python. This video has great explanations and a brilliant implementation, please do check it out if you are interested in learning how AI works.
- Video: [Original Source](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- Code: [Python Implementation](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook)
- Data: [Dataset on Kaggle](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/input)
