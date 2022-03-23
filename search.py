
import argparse
import numpy as np
import cv2
import csv
import json
from Descriptor import  Bounding_box, Searcher, ColorDescriptor, orb_descriptors
from Descriptor import NumpyArrayEncoder

# creating the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True, help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True, help = "Path to the result path")
#args = vars(ap.parse_args())

#intializing the color descriptor
cd = ColorDescriptor((12,17,15))

#loading the query image and describe it
with open('CBIR/CBIR/queries.csv', 'r') as file:
    reader = csv.reader(file)
    queries = [row for row in reader][0]
def result(query, results):
    result = {}
    result['query'] = cv2.resize(query, (300,300))
    for (score, resultID) in results:
        #load the result image and display it
        result[resultID] = cv2.resize(cv2.imread('CBIR/CBIR/image_fashion' + "/" + resultID), (300,300))
    
    h_stack1 = np.hstack([img for k,img in result.items() if k in list(result.keys())[:3]])
    h_stack2 = np.hstack([img for k,img in result.items() if k in list(result.keys())[3:]])
    v_stack = np.vstack([h_stack1,h_stack2])
    return v_stack

for q in queries:
    query = cv2.imread(q)

    queryFeatures = cd.describe(query)
    features_hist = [str(f) for f in queryFeatures]
    imgId = q[q.rfind("/")+1:]
    

    queryFeatures_orb = orb_descriptors(query)
    
    #performing the search
    s1 = Searcher('CBIR/CBIR/index.json')

    results = s1.search(queryFeatures,limit=5)
    orb_results = s1.orb_searcher(queryFeatures_orb,path='CBIR/CBIR/index_orb.json', limit=5)

    #Results
    v_stack= result(query, results)
    v_stack_orb = result(query, orb_results)
    cv2.imshow('results', v_stack)
    cv2.imshow('results_orb', v_stack_orb)
    cv2.waitKey(0)
    #save the query index
    with open("CBIR/CBIR/index_orb.json", "r") as f:
        index_orb = json.load(f)
    index_orb.update({imgId:queryFeatures_orb})
    with open("CBIR/CBIR/index_orb.json", "w") as f:
        json.dump(index_orb, f, cls=NumpyArrayEncoder, indent=4)
    with open('CBIR/CBIR/index.json',"r") as file:
        index = json.load(file)
    index.update({imgId:features_hist})
    with open('CBIR/CBIR/index.json',"w") as file:
        json.dump(index, file, indent=4)

print('END')
