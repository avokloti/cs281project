import os
import numpy as np
import pandas as pd
import shutil

""" Script that copies the files from the Berlin nature archive which contains birdsong
    into a new directory. Can optionally filter by quality, duration and date. """

# Note: berlin_dir must contain only the berlin audio files with the original names
# (with any extension) for example AccGen00001.mp3 or ChlHyb00013.wav
# files satisfying criteria are copied to filter_dir


## change if needed ##
berlin_dir = './berlin_all/'
filter_dir = './berlin_filtered/'
metadata_filename = './RefSysMetadata.csv'

## import the berlin metadata
metadata = pd.read_csv(metadata_filename, delimiter=',', index_col='id')

# orders corresponding to birds
bird_check_dict = {
    'Accipitriformes': True,
    'Anseriformes': True,
    'Anura': False,
    'Apodiformes': True,
    'Artiodactyla': False,
    'Bucerotiformes': True,
    'Caprimulgiformes': True,
    'Carnivora': False,
    'Charadriiformes': True,
    'Columbiformes': True,
    'Coraciiformes': True,
    'Cuculiformes': True,
    'Falconiformes': True,
    'Flamingos': True,
    'Galliformes': True,
    'Gaviformes': True,
    'Gruiformes': True,
    'Orthoptera': False,
    'Passeriformes': True,
    'Pelecaniformes': True,
    'Piciformes': True,
    'Podicipediformes': True,
    'Strigiformes': True,
    'Suliformes': True
}

def filter(filename, min_quality=None, min_duration=-1.0, min_year=None):
    # min_quality is one of 'a', 'b', 'c', 'u'
    # min_duration is a float
    # min_year is an integer
    
    # A few (5) files are not listed in the metadata file. We discard those
    file_id = os.path.splitext(filename)[0]
    if not file_id in metadata.index:
        print(file_id + " not in metadata")
        return False

    data = metadata.loc[os.path.splitext(filename)[0]]
    
    # filter out non-birds
    is_bird = bird_check_dict[data.order]
    
    # filter by quality
    quality_check = True
    if min_quality:
        quality_check = data.quality <= min_quality
    
    # filter by duration
    duration_check = data.duration >= min_duration
    
    # filter by year
    year_check = True
    if min_year:
        if pd.isna(data.date):
            # date not available
            year_check = False
        else:
            try:
                year_check = int(data.date.split('-')[0]) >= min_year
            except ValueError:
                year_check = False
    
    return is_bird and quality_check and duration_check and year_check


def main():

    for filename in os.listdir(berlin_dir):
        if filter(filename):
            shutil.copy(berlin_dir+filename, filter_dir)
            continue


    return


if __name__ == "__main__":

    main()

