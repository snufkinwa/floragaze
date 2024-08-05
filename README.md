<p align="center">
<img src="https://imgur.com/5wSdLCq.png" width="500">
</p>

# FloraGaze: AI Flower Recognition

This project demonstrates an image classifier built using PyTorch and converted into a command line application. It is part of the AWS AI/ML Scholarship at Udacity. The instructions and markdowns provided in the accompanying Jupyter Notebook are credited to @udacity/active-public-content.

## Table of Contents

- [Project Overview](#project-overview)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Resources](#resources)

## Project Overview

This project involves building an image classifier using a deep learning model with PyTorch. The steps include:

1. **Data Loading and Preprocessing**: Load and preprocess the dataset for training and validation.
2. **Model Building and Training**: Develop a neural network to classify images into various categories.
3. **Model Evaluation**: Assess the model's performance on the test dataset.
4. **Saving and Loading Checkpoints**: Save the trained model as a checkpoint and implement functionality to load it.
5. **Command Line Application**: Convert the model into a command line application to classify new images.

## Usage

### Jupyter Notebook

1. Training the Model:

Open the `AI_Programming_with_Python_Project.ipynb` notebook in Google Colab. Ensure you have the GPU runtime enabled for faster training. Follow the instructions in the notebook to load data, build, train, and save the model.

2. Predicting Image Class:

After training the model in the notebook, use the provided cells to test the model on new images.

### Command Line Application

There are two notebooks in the `command-line-app/test` directory showing how to utilize remote GPUs in platforms like Google Colab.

⚠️ **Warning**: Training the model on a CPU can be very time-consuming. It is highly recommended to use a GPU for training to significantly reduce the training time.

1. Zip the command-line-app:

   Before running the scripts in Google Colab, zip the `command-line-app` directory. This makes it easier to upload and use the application in the Colab enviroment.

   ```bash
    zip -r command-line-app.zip command-line-app/
   ```

2. Upload and Use with remote GPUs:
   - Upload the zipped `command-line-app.zip` to your remote enviroment.
   - Upload one of the test notebooks from the test directory.
3. Run the application:
   - Use the test notebook to unzip the files and set up the environment.
   - Follow the notebook instructions to run the command-line application on the remote GPU.

## Screenshots

#### Screenshot 1: Notebook Results

<p align="center">
<img src="https://imgur.com/l7OSDIR.png" width="500">
</p>

#### Screenshot 2: CLI Results

<p align="center">
<img src="https://imgur.com/mIc50Hl.png" width="500">
</p>

## Resources

The course material Udacity was a great start, I challenged myself to make a model over 80%.

[Pytorch, No Tears](https://learn-pytorch.oneoffcoder.com/index.html) was helpful in understanding the ins and outs of training a new network.
