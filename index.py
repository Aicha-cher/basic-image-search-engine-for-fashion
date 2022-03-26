
from tools import ColorDescriptor, orb_descriptors, NumpyArrayEncoder
#Used for parsing command line arguments
import argparse
#Used for getting paths of our images
import glob
import cv2
import random
from tqdm import tqdm
import csv
import json


if __name__ == "__main__":

    #Create the argument parser to parse the arguments
    arg = argparse.ArgumentParser()

    #Switch for the path to our photos directory
    arg.add_argument("-d", "--dataset", required=True, help="Path to directory that contains images")
    arg.add_argument("-ci", "--histgram_index", required=False, help="Path to where the histogram color index will be stored")
    arg.add_argument("-orbi", "--orb_index", required=False, help="Path to where the ORB index will be stored")
    arg.add_argument("-moment", "--moment_index", required=False, help="Path to where the Moment color index will be stored")
    args = arg.parse_args()

    dataset_path = args.dataset
    orb_index = args.orb_index
    hist_index = args.histgram_index
    moment_index = args.moment_index

    # reduce the dataset to speedup the search
    random.seed(1025)
    dataset_sample = random.sample(glob.glob(dataset_path+"/*.jpg"), 900)
    queries = random.sample(dataset_sample, 100)
    with open('queries.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(queries)
    dataset_sample = [q for q in dataset_sample if q not in queries]
    #Initializing our color descriptor
    cd = ColorDescriptor((16, 4, 4))

    #Using glob to get path of images and go through all of them
    features_dict_hist, features_dict_orb, features_dict_moment = {}, {}, {}
    for imagePath in tqdm(dataset_sample):
        #Get the UID of the image path and load the image
        imageUID = imagePath[imagePath.rfind("/")+1:]
        image = cv2.imread(imagePath)
        #Using the describe function
        if hist_index:
            features_hist = cd.describe(image, x_segments=2, y_segments=2, color_descriptor='histogram')
            features_hist = [str(f) for f in features_hist]
            features_dict_hist.update({imageUID: features_hist})
        if orb_index:
            features_orb = orb_descriptors(image)
            features_dict_orb.update({imageUID: features_orb})
        if moment_index:
            features_moment = cd.describe(image, x_segments=2, y_segments=2, color_descriptor='moment')
            features_dict_moment.update({imageUID: features_moment})


    #open the output index file for writing
    if hist_index:
        with open(hist_index, "w") as file:
            json.dump(features_dict_hist, file, indent=4)
    if orb_index:
        with open(orb_index, "w") as write_file:
            json.dump(features_dict_orb, write_file, cls=NumpyArrayEncoder, indent=4)
    if moment_index:
        with open(moment_index, "w") as file:
            json.dump(features_dict_moment, file, indent=4)
