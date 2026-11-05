# UrbanSound8K Audio Classification

This repository contains a completed notebook for classifying environmental sounds from the **UrbanSound8K** dataset using a Convolutional Neural Network (CNN).

## Project objective

The goal is to build a model that can recognise which of ten common urban sound categories is present in a short audio recording. Rather than feeding raw audio directly into the network, the notebook converts each recording into a Mel spectrogram. A Mel spectrogram represents how the energy of different sound frequencies changes over time, giving the CNN a two-dimensional representation it can learn from.

## Dataset

The notebook downloads the UrbanSound8K dataset using `kagglehub`:

- **Dataset:** [UrbanSound8K on Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
- **Size:** 8,732 labelled audio clips
- **Classes:** air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren and street music
- **Metadata:** `UrbanSound8K.csv`, which contains each clip's filename, fold, class ID and class name.

The notebook expects the audio files to be stored in the dataset's `audio/fold1` through `audio/fold10` folders. The path resolver also supports dataset copies where the fold folders are directly under the dataset directory.

## Notebook walkthrough

Open **[Copy_of_[Breakout_Room]_ML_Pipeline_Week_5_Practice.ipynb](./Copy_of_%5BBreakout_Room%5D_ML_Pipeline_Week_5_Practice.ipynb)** to follow the full implementation.

### 1. Import libraries and download the dataset

The notebook imports the tools needed for audio processing, data handling, visualisation and deep learning:

- `librosa` loads audio, trims silence and creates Mel spectrograms.
- `numpy` and `pandas` handle arrays and metadata.
- `matplotlib` visualises waveforms, spectrograms and training history.
- `TensorFlow/Keras` builds, trains and evaluates the CNN.
- `scikit-learn` splits the data and produces evaluation metrics.
- `kagglehub` downloads UrbanSound8K.

The dataset is downloaded and its metadata CSV is loaded into a DataFrame.

### 2. Explore the metadata

The notebook checks the dataset dimensions, previews the metadata, counts the total number of recordings and displays how many recordings belong to each sound class. A bar chart makes any class imbalance easier to see.

### 3. Locate and load audio recordings

A helper function constructs the path to each audio file using its fold number and filename. It checks the expected path and raises a clear error if a file is missing.

The audio-loading function converts recordings to mono, resamples them to **22,050 Hz**, and makes each clip four seconds long. Short clips are padded with zeros and longer clips are trimmed to the target duration. This gives the later processing steps a consistent input length.

### 4. Visualise and listen to a recording

The first recording is loaded and displayed as a waveform. The notebook also plays the recording inside the notebook. The waveform shows how the signal's amplitude changes over time, while listening gives a direct example of the sound being classified.

### 5. Trim silence and demonstrate noise

The `trim_audio()` function uses `librosa.effects.trim()` to remove leading and trailing sections that fall below the selected energy threshold. The original and trimmed waveforms are plotted for comparison.

The `add_random_noise()` function adds a small amount of random noise to an audio signal. This demonstrates a basic audio-augmentation technique and shows how background noise changes a waveform. The demonstration is kept separate from the main feature-extraction pipeline so the held-out test examples are not deliberately corrupted.

### 6. Convert audio into Mel spectrograms

The `audio_to_mel_spectrogram()` function performs three main operations:

1. Computes a Mel-scaled spectrogram from the audio waveform.
2. Uses 128 Mel frequency bands and a hop length of 512 samples.
3. Converts the power values to decibels so the energy differences are easier to represent.

The notebook plots a spectrogram for the first recording and compares examples from several sound classes. These plots help show that different sounds can produce different time-frequency patterns.

### 7. Prepare the CNN dataset

The `prepare_cnn_data()` function processes the metadata rows and converts each corresponding audio file into a Mel spectrogram. Each spectrogram is padded or truncated to **128 frequency bands by 173 time frames**. The resulting arrays are then given a final channel dimension, producing the four-dimensional input format expected by a 2D CNN:

`(number of recordings, 128, 173, 1)`

The class IDs are stored separately as the target labels.

By default, `MAX_SAMPLES = None`, so the notebook processes the full dataset. For a quicker smoke test, set `MAX_SAMPLES` to a smaller number; the function then takes a roughly balanced number of examples from each class. The smaller run is useful for checking that the pipeline works, but final results should be produced using the full dataset.

### 8. Split the data

The labels are converted to one-hot encoded vectors because the model predicts one of ten classes. The data is split into:

- **80% training data**, used to learn the model parameters.
- **20% test data**, held back for the final evaluation.

The split uses a fixed random seed for reproducibility and stratification to preserve the class proportions in both subsets.

### 9. Build the CNN

The Sequential CNN contains three convolutional blocks. Each block uses convolution layers to learn local patterns in the spectrogram, batch normalisation to help stabilise training, and max pooling to reduce the spatial dimensions. Dropout is used to reduce overfitting.

After the convolutional blocks, global average pooling summarises the learned feature maps. Dense layers then combine those features, and the final softmax layer outputs a probability for each of the ten sound classes.

### 10. Train the model

The model is compiled with:

- **Adam optimiser** with an initial learning rate of 0.001.
- **Categorical cross-entropy** loss, suitable for a multi-class classification task with one-hot labels.
- **Accuracy** as a training metric.

Training runs for up to 30 epochs with a batch size of 32 and uses part of the training data for validation. Early stopping restores the best weights when validation loss stops improving. Reduce-on-plateau lowers the learning rate when validation loss stalls. Accuracy and loss curves are plotted to compare training and validation behaviour.

### 11. Evaluate and save the model

The notebook evaluates the trained model on the test set, which was not used for fitting. It reports test loss and accuracy, then generates a classification report containing precision, recall and F1-score for each class. A confusion matrix shows which sound classes the model predicts correctly and which classes it confuses.

Finally, the trained model is saved as `urbansound8k_cnn.keras`.

## How to run the notebook

The notebook is designed for **Google Colab** and can use a GPU runtime.

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type** and choose a GPU if one is available.
3. Run the cells from top to bottom.
4. If Kaggle prompts for authentication, configure Kaggle access in the Colab session so `kagglehub` can download the dataset.
5. Allow time for feature extraction. Processing all 8,732 recordings is the most time-consuming stage.
6. Check the training curves, test metrics, classification report and confusion matrix at the end.

Required Python packages include `librosa`, `kagglehub`, `numpy`, `pandas`, `matplotlib`, `scikit-learn` and `tensorflow`. Colab normally includes most scientific Python packages; install any missing package in a separate cell with `%pip install package-name`.

## Important notes and limitations

- **Runtime:** full-dataset audio loading and spectrogram generation can take a while. A GPU speeds up CNN training, but audio preprocessing may still be CPU-bound.
- **Evaluation split:** the notebook uses a stratified random clip-level split. A stronger generalisation experiment would hold out complete UrbanSound8K folds, rather than randomly distributing clips, to reduce the chance of related recordings appearing in both training and test data.
- **Noise demonstration:** random noise is shown as a separate example. The main training pipeline uses trimmed audio without adding random noise.
- **Reproducibility:** random seeds are set, but exact results can still vary with TensorFlow, hardware and library versions.
- **Interpretation:** accuracy alone does not describe performance on every class. Use the per-class precision, recall, F1-score and confusion matrix when discussing the model.

## Repository contents

| File | Purpose |
|---|---|
| `Copy_of_[Breakout_Room]_ML_Pipeline_Week_5_Practice.ipynb` | Complete notebook containing data exploration, audio preprocessing, CNN training and evaluation |
| `README.md` | Project overview, implementation walkthrough and run instructions |

## Outcome

The notebook implements an end-to-end multi-class audio classification workflow. It takes labelled audio recordings, converts them into fixed-size Mel spectrograms, trains a CNN to recognise the sound categories, and evaluates its predictions on a held-out test set. The actual performance values are generated when the notebook is run; they are intentionally not hard-coded here.
