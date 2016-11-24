#!/usr/bin/env python

"""
This module uses the UrbanSound8K/metadata/UrbanSound8K.csv file to provide
a path to each sound sample for the samples provided in the UrbanSound dataset.
Features are extracted from each sound sample using Librosa and Y values are
pulled from the metadata file.

The values are cached to a file in the root directory of the project so that
they can be easily pulled to train neural nets with different configurations
without having to extract the features each time.
"""

import os.path
import pandas as pd
import numpy as np
import librosa


def __extract_feature(filename):
    """
    Extracts various features about the sound sample using librosa. Feature extraction taken
    from http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
    """
    sound_clip_waveform, sampling_rate = librosa.load(filename)
    stft = np.abs(librosa.stft(sound_clip_waveform))
    mfccs = np.mean(librosa.feature.mfcc(y=sound_clip_waveform, sr=sampling_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampling_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(sound_clip_waveform, sr=sampling_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampling_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound_clip_waveform), sr=sampling_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

 
def __write_csv_from_folds(fold_numbers, dataframe, output_filename):
    """
    Writes a CSV file with all features and labels to the relative file
    given by "filename." Only writes writes data from fold numbers provided
    within the fold_numbers argument (folds can be 1-10). Appends label
    to the last column (column 194) in the CSV file
    """
    features = np.empty((0, 194))
    for idx, row in dataframe.iterrows():
        if row.fold in fold_numbers:
            #Gets the filename and extracts the features from the file
            filename = "./UrbanSound8K/audio/fold{}/{}".format(row.fold, row.slice_file_name)
            mfccs, chroma, mel, contrast, tonnetz = __extract_feature(filename)

            #puts features into a 193-dim vector and appends label at the end
            feature_row = np.hstack([mfccs, chroma, mel, contrast, tonnetz, row.classID])

            #Adds sample to running list of samples
            features = np.vstack([features, feature_row])

    #Caches data to file so it doesn't have to be extracted each time
    feature_dataframe = pd.DataFrame(features)
    feature_dataframe.to_csv(path_or_buf=output_filename, header=False, index=False)



def process_csv(num_train_folds):
    """
    This function loads a the metadata file from a relative file path,
    processes all of the files, then writes them to a file in the root
    project directory. It does this for the training data as well as the 
    the test data, outputting each to two seperate files. It takes the
    number of folds to use as a training set as a parameter.
    """
    df = pd.read_csv('./UrbanSound8K/metadata/UrbanSound8K.csv')
    __write_csv_from_folds(range(1, num_train_folds + 1), df, "train_data.csv")
    __write_csv_from_folds(range(num_train_folds + 1, 11), df, "test_data.csv")
    
if __name__ == "__main__":
    process_csv()