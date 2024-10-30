from os import path
import glob
from functools import lru_cache, reduce
from operator import add

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA_DIR
from nmr import NMRDataset

FOLDER_PHOTO =    path.join(DATA_DIR, "20180418-1809-photochemical_space")  # contains 1018 folders
FOLDER_SIMPLE2R = path.join(DATA_DIR, "20171031-1642-Simple_space-DMSO")    # contains 102 folders
FOLDER_SIMPLE6R = path.join(DATA_DIR, "20180404-1917-6R-Simple_space-DMSO") # contains 72 folders
paths_photo =    { path.basename(d): d   for d in glob.glob(path.join(FOLDER_PHOTO,    "**", "[0-9]*-NMR-*[0-9]")) }
paths_simple2r = { path.basename(d): d   for d in glob.glob(path.join(FOLDER_SIMPLE2R, "**", "[0-9]*-NMR-*[0-9]")) }
paths_simple6r = { path.basename(d): d   for d in glob.glob(path.join(FOLDER_SIMPLE6R, "**", "[0-9]*-NMR-*[0-9]")) }

df_photo =    pd.read_csv(path.join(DATA_DIR, "photo.csv"))       # labeled test data: 1018 reactions; num_classes=4 (0,1,2,3)
df_simple2r = pd.read_csv(path.join(DATA_DIR, "simple_2r.csv"))   # labeled training data: 390 reactions; 
df_simple6r = pd.read_csv(path.join(DATA_DIR, "simple_6r.csv"))   # labeled training data: 54 reactions; 

# Preprocessing performed on NMR spectra before using them as input
xform = ( lambda s: s.crop(0, 12, inplace=True).erase(1.8, 4, inplace=True).normalize(inplace=True))  # erase(1.8, 4) ->removing the intensity values between 1.8 & 4 ppm (solvent cut)

# NMR spectra of the reaction mixture to construct data-point AFTER reaction execution
photo_reaction_dataset =    NMRDataset([paths_photo[n]    for n in df_photo.name   ],                                                  transform=xform)
simple2r_reaction_dataset = NMRDataset([paths_simple2r[n] for n in df_simple2r.name], target_length=photo_reaction_dataset.min_length, transform=xform,)
simple6r_reaction_dataset = NMRDataset([paths_simple6r[n] for n in df_simple6r.name], target_length=photo_reaction_dataset.min_length, transform=xform,)

assert (photo_reaction_dataset.min_length == simple2r_reaction_dataset.min_length == simple6r_reaction_dataset.min_length)
min_length = photo_reaction_dataset.min_length

# NMR spectra of the pure reagents to construct data-point BEFORE reaction execution - test data(?)
FOLDER_PHOTO_REAGENTS = path.join(DATA_DIR, "002_photo_space/reagents")   # 18 reagent + 1 base folders
photo_dirs = glob.glob(path.join(FOLDER_PHOTO_REAGENTS, "*-NMR-0"))
photo_dataset = NMRDataset(photo_dirs, target_length=min_length, transform=xform)  
photo_reagents = [path.basename(d).split("-")[0] for d in photo_dirs]  # list of reagent_name strings
photo_numbers = {d: i for i, d in enumerate(photo_reagents)}           # dictionary {reagent_name : number}  # number is like reagent_id 0,1,2,3. ...

# NMR spectra of the pure reagents to construct data-point BEFORE reaction execution - training data(?)
FOLDER_SIMPLE_REAGENTS = path.join(DATA_DIR, "001_simple_space/reagents")   # 9 reagent folders
simple_dirs = glob.glob(path.join(FOLDER_SIMPLE_REAGENTS, "*-NMR-0"))
simple_dataset = NMRDataset(simple_dirs, target_length=min_length, transform=xform)
simple_reagents = [path.basename(d).split("-")[0] for d in simple_dirs]
simple_numbers = {d: i for i, d in enumerate(simple_reagents)}

@lru_cache(maxsize=None)
def get_reagents(folder, file): # returns a dictionary {reagent_name : reagent_volume} and a float total_volume
    ''' sample folder: FOLDER_PHOTO = path.join(DATA_DIR, "20180418-1809-photochemical_space") = data/20180418-1809-photochemical_space
        sample file: "0000-post-reactor1-NMR-20180418-2128" - which is a name of the folder containing the corresponding NMR data.'''        
    reactor = file.split("-")[2] # reactor:  "reactor1"
    counter = file.split("-")[0] # counter:  "0000"
    csv_file = path.join(folder, counter, reactor + ".csv")   # csv_file = data/20180418-1809-photochemical_space/0000/reactor1.csv
    if not path.exists(csv_file):  return ({}, None)
    space = pd.read_csv(csv_file)

    if "photo" in folder:
        volume_first = 2
        volume_second = 2
        volume_third = 2
        volume_base = 2
    elif "Simple" in folder:
        volume_first = space["first_vol"][0]
        volume_second = space["second_vol"][0]
        volume_third = space["third_vol"][0]
        volume_base = 1.5

    volume_acid = 0.5
    volume_cat = 0.5
    volume_tot = volume_first + volume_second + volume_third
    reagents = {}

    reagents[space["first"][0]] = volume_first
    reagents[space["second"][0]] = volume_second

    if "post" in file: reagents[space["third"][0]] = volume_third

    if space["base"][0] == True:
        if "Simple" in folder:  reagents["dbn"] = volume_base
        else: reagents["base"] = volume_base
        volume_tot += volume_base
    try:
        if space["acid"][0] == True:
            reagents["acid"] = volume_acid
            volume_tot += volume_acid

        if space["cat"][0] != "none":
            cat = space["cat"][0]
            reagents[cat] = volume_cat
            volume_tot += volume_cat
    except KeyError: pass

    return reagents, volume_tot

def generate_training_dataset( augmentation_factor=250, max_shift=15, max_total_shift=450, coeff_wobble=0.1, circular_shift=False, quiet=True):

    num_training = augmentation_factor * (len(df_simple2r) + len(df_simple6r))
    num_validation = int(0.1*num_training)
    num_test = len(df_photo)

    # total training data -> to be splitted into training + validation
    inputs = np.zeros((num_training, 2, min_length), dtype="float32")     # num_samples x num_spectra x spectrum_length (num_spectra = 2: before and after reaction)
    outcomes = np.zeros((num_training, 4), dtype="float32")   # label representing values for num_classes=4

    # validation data
    # val_inputs = np.zeros((num_validation, 2, min_length), dtype="float32")
    # val_outcomes = np.zeros((num_validation, 4), dtype="float32")
    
    # test data
    test_inputs = np.zeros((num_test, 2, min_length), dtype="float32")
    test_outcomes = np.zeros((num_test, 4), dtype="float32")

    # generate and augment training & validation data points together (initially)
    cntr = 0   # data point (inputs entry) counter variable
    for i in tqdm(range(augmentation_factor)):
        '''in every loop count, NMR spectra are shifted randomly, thereby augmenting the dataset '''
        if not quiet: print(f"Augmentation @ {i+1}x")
        for folder, df, numbers, dataset, rxn_dataset in [  (FOLDER_SIMPLE2R, df_simple2r, simple_numbers, simple_dataset, simple2r_reaction_dataset),
                                                            (FOLDER_SIMPLE6R, df_simple6r, simple_numbers, simple_dataset, simple6r_reaction_dataset) ]:
            for j, (rxn, label) in df.reset_index(drop=True).iterrows():
                # rxn refers to the name of the folder which contains the corresponding NMR data. # rxn sample: "0000-post-reactor1-NMR-20180418-2128" - which is a folder name
                # label refers to the class value, i.e., one digit among [0,1,2,3]
                shift = np.random.randint(max_total_shift * 2) - max_total_shift
                outcomes[cntr, label] = 1.0

                inputs[cntr, 0, :] = (reduce(add,   # NMR data (nd.array real part of the NMRSpectrum.spectrum attribute) before reaction execution
                                                [dataset[numbers[r]].shift(np.random.randint(max_shift * 2) - max_shift)* (np.random.randn() * coeff_wobble + 1.0)* coeff 
                                                for r, coeff in get_reagents(folder, rxn)[0].items() if r in numbers],  # r is reagent_name string, coeff is reagent_volume float
                                            )
                                            .normalize().shift(shift, circular=circular_shift).spectrum.real
                                        )                
                # 'dataset[numbers[r]]' retrieves the NMR spectrum for the reagent 'r'.
                # The expression '[dataset[numbers[r]].shift(SHIFTING OPERATIONS) for r, coeff in get_reagents(folder, rxn)[0].items() if r in numbers]'
                # generates a collection of shifted NMR spectra for the reagents used in the specified reaction.
                # 'reduce(add, [shifted reagent NMR Spectra])' combines these shifted NMR spectra into a single, summed spectrum for all reagents
                # which results in a final spectrum before reaction execution.

                inputs[cntr, 1, :] = (rxn_dataset[j].shift(shift, circular=circular_shift).spectrum.real) # NMR data after reaction execution
                # The variable 'j' corresponds to the same entry as 'rxn', meaning both refer to the same reaction.
                # 'rxn_dataset[j]' retrieves the NMR spectrum of the reaction mixture for entry 'j',
                # which results in a spectrum after reaction execution
                # Additionally, 'rxn' is passed to the 'get_reagents()' function, thus linking the reagent spectra 
                # to the same reaction, but representing the state before the reaction is executed.

                cntr += 1
    
    # # split the generated augmented dataset intro training & validation
    # indices = np.arange(num_training)
    # np.random.shuffle(indices) # shuffle the training data (both inputs and outcomes)
    
    # inputs, outcomes = inputs[indices], outcomes[indices]

    # # Split validation and training data
    # val_inputs = inputs[:num_validation]    # First 10% for validation
    # val_outcomes = outcomes[:num_validation]
    
    # train_inputs = inputs[num_validation:]  # Remaining 90% for training
    # train_outcomes = outcomes[num_validation:]
    
    # generate testing data points (no need for augmentation in inference)
    cntr = 0
    for j, (rxn, label) in df_photo.reset_index(drop=True).iterrows():
        test_outcomes[cntr, label] = 1.0
        test_inputs[cntr, 0, :] = (reduce(add,
                                            [photo_dataset[photo_numbers[r]] for r, coeff in get_reagents(FOLDER_PHOTO, rxn)[0].items() if r in photo_numbers],
                                        )
                                        .normalize().spectrum.real
                                    )
        test_inputs[cntr, 1, :] = photo_reaction_dataset[j].spectrum.real
        cntr += 1

    return  inputs, outcomes, test_inputs, test_outcomes
        # train_inputs, train_outcomes, val_inputs, val_outcomes, test_inputs, test_outcomes