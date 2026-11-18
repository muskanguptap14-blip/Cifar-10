# CIFAR-10 Image Classification with a CNN

A convolutional neural network built with TensorFlow/Keras that classifies 32x32 color images from the CIFAR-10 dataset into 10 categories. Everything lives in a single notebook, `cifar.ipynb`.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) has 60,000 color images (32x32 pixels) in 10 classes: 50,000 for training and 10,000 for testing.

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The dataset is downloaded automatically through `tf.keras.datasets.cifar10`.

## Model Architecture

A VGG-style CNN with three convolutional blocks, followed by a dense classifier. Total: **1,341,226 trainable parameters** (~5.12 MB).

| Block | Layers |
|-------|--------|
| 1 | 2x Conv2D (32 filters, 3x3, ReLU, same padding), MaxPooling 2x2, Dropout 0.25 |
| 2 | 2x Conv2D (64 filters, 3x3, ReLU, same padding), MaxPooling 2x2, Dropout 0.25 |
| 3 | 2x Conv2D (128 filters, 3x3, ReLU, same padding), MaxPooling 2x2, Dropout 0.25 |
| Classifier | Flatten, Dense 512 (ReLU), Dropout 0.5, Dense 10 (Softmax) |

## Training Setup

- **Preprocessing:** pixel values scaled to [0, 1]; labels one-hot encoded
- **Optimizer:** Adam
- **Loss:** categorical cross-entropy
- **Epochs:** 50
- **Batch size:** 64
- **Validation split:** 20% of the training data

## Results

Final epoch (50/50):

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 87.27% | 81.77% |
| Loss | 0.3699 | 0.6063 |

Validation accuracy climbed from about 73% in the early epochs to around 80-82% by the end. The gap between training and validation accuracy suggests some overfitting, even with dropout.

The notebook also plots accuracy/loss curves over the epochs and shows predictions on the first 5 test images, printing the true and predicted label for each.

> Note: the notebook doesn't run `model.evaluate()` on the held-out test set, so no final test accuracy is reported here.

## Getting Started

### Requirements

- Python 3.9+
- TensorFlow (includes Keras)
- NumPy
- Matplotlib
- Jupyter

```bash
pip install tensorflow numpy matplotlib jupyter
```

### Run

```bash
jupyter notebook cifar.ipynb
```

Run the cells in order. The first run downloads the dataset (~170 MB). Training for 50 epochs takes roughly 1.5 minutes per epoch on a CPU, so a GPU is recommended if you have one.

## Project Structure

```
.
├── cifar.ipynb   # data loading, model, training, plots, predictions
└── README.md
```

## Ideas for Improvement

- Add data augmentation (random flips, shifts, rotations) to reduce overfitting
- Add Batch Normalization layers
- Use early stopping and learning-rate scheduling
- Evaluate on the test set and add a confusion matrix
- Save the trained model (`model.save(...)`) so it can be reused without retraining
