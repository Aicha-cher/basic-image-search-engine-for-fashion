
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
import numpy

if __name__ == "__main__":

    #Create the argument parser to parse the arguments
    ap = argparse.ArgumentParser()

    #Switch for the path to our photos directory
    ap.add_argument("-d","--dataset", required = True , help = "Path to directory that contains images")
    ap.add_argument("-i","--index", required = True , help = "Path to where the index will be stored")
    #args = vars(ap.parse_args())

    dataset_path = 'CBIR/CBIR/image_fashion'
    # reduce the dataset to speedup the search
    random.seed(1025)
    dataset_sample = random.sample(glob.glob(dataset_path+"/*.jpg"),900 )
    queries = random.sample(dataset_sample, 100)
    with open ('CBIR/CBIR/queries.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(queries)
    dataset_sample = [q for q in dataset_sample if q not in queries]
    #Initializing our color descriptor
    cd = ColorDescriptor((12,17,15))


    #Using glob to get path of images and go through all of them
    features_dict_hist, features_dict_orb = {}, {}
    for imagePath in tqdm(dataset_sample):
        #Get the UID of the image path and load the image
        imageUID =  imagePath[imagePath.rfind("/")+1:]
        image = cv2.imread(imagePath)
        #Using the describe function
        features_hist = cd.describe(image)
        features_orb = orb_descriptors(image)
        # store the features
        features_hist = [str(f) for f in features_hist]
        features_dict_hist.update({imageUID:features_hist})
        features_dict_orb.update({imageUID:features_orb})

        #open the output index file for writing
    with open('CBIR/CBIR/index.json',"w") as file:
        json.dump(features_dict_hist, file, indent=4)

    with open("CBIR/CBIR/index_orb.json", "w") as write_file:
        json.dump(features_dict_orb, write_file, cls=NumpyArrayEncoder, indent=4)

