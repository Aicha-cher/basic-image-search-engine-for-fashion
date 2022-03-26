
import argparse
import numpy as np
import cv2
import csv
import json
import os
from tools import  Bounding_box, Searcher, ColorDescriptor, orb_descriptors, NumpyArrayEncoder

# creating the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True, help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True, help = "Path to the result path")
#args = vars(ap.parse_args())


#loading the query image
with open('queries.csv', 'r') as file:
    reader = csv.reader(file)
    queries = [row for row in reader][0]

def result(query, results):
    result = {}

    image = cv2.resize(query, (300, 300))
    cv2.rectangle(image,(0,0),(300,300), (0,0,255), 5)
    result['query'] = image
    for (score, resultID) in results:
        #load the result image and display it
        result[resultID] = cv2.resize(cv2.imread('image_fashion' + "/" + resultID), (300,300))

    h_stack1 = np.hstack([img for k,img in result.items() if k in list(result.keys())[:3]])
    h_stack2 = np.hstack([img for k,img in result.items() if k in list(result.keys())[3:]])
    v_stack = np.vstack([h_stack1,h_stack2])
    return v_stack

def plotting(query, results, window_name):
    v_stack= result(query,results)
    cv2.imshow(window_name, v_stack)
    cv2.waitKey(0)
    cv2.imwrite(f'resulted_image/{window_name}.jpg', v_stack)

def save(query, results, id, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    v_stack= result(query,results)
    cv2.imwrite(path + f'results_{id}.jpg', v_stack)

#intializing the color descriptor
cd = ColorDescriptor((16,4,4))
searcher = ['combined']
moment_q_index, hist_q_index, orb_q_index = [], [], []
for q in queries:
    query = cv2.imread(q)
    imgId = q[q.rfind("/")+1:]
    #performing the search
    s1 = Searcher()
    if 'histogram' in searcher:
        queryFeatures = cd.describe(query,x_segments=2, y_segments=2, color_descriptor='histogram')
        #queryFeatures = [str(f) for f in queryFeatures]
        hist_results, hist_index = s1.search(queryFeatures,limit=5, combine=False, path='index.json')
        hist_q_index.append({imgId:queryFeatures})
        #plotting(query,hist_results, window_name ='hist_results')
        save(query,hist_results, id=imgId, path='resulted_image/histogram/')

    if 'orb' in  searcher:
        queryFeatures_orb = orb_descriptors(query)
        orb_results, index_orb = s1.orb_searcher(queryFeatures_orb,path='index_orb.json', limit=5, combine=False)
        orb_q_index.append({imgId:queryFeatures_orb})
        #plotting(query,orb_results, window_name ='orb_results')
        save(query,orb_results, id=imgId, path='resulted_image/orb/')

    if 'moment' in  searcher:
        queryFeatures_moment = cd.describe(query,x_segments=2, y_segments=2, color_descriptor='moment')
        moment_results, moment_index = s1.search(queryFeatures_moment,path='index_moment.json', limit=5, combine=False)
        moment_q_index.append({imgId:queryFeatures_moment})
        #plotting(query,moment_results, window_name ='moment_results')
        save(query,moment_results, id=imgId, path='resulted_image/moment/')

    if 'combined' in searcher:
        queryFeatures_moment = cd.describe(query,x_segments=2, y_segments=2, color_descriptor='moment')
        moment_results, moment_index = s1.search(queryFeatures_moment,path='index_moment.json', limit=5, combine=True)
        moment_q_index.append({imgId:queryFeatures_moment})
        queryFeatures_orb = orb_descriptors(query)
        orb_results, index_orb = s1.orb_searcher(queryFeatures_orb,path='index_orb.json', limit=5, combine=True)
        orb_q_index.append({imgId:queryFeatures_orb})
        combined_results = s1.combined_search(moment_results, orb_results)
        #plotting(query,combined_results, window_name ='combined_results')
        save(query,combined_results, id=imgId, path='resulted_image/combined_2/')
    cv2.destroyAllWindows()
#save the query index
if 'histogram' in searcher:
    for q in hist_q_index:
        hist_index .update(q)

    with open('index.json',"w") as file:
        json.dump(hist_index, file, indent=4)

if 'orb' in searcher or 'combined' in searcher:
    for q in orb_q_index:
        index_orb .update(q)

    with open('index_orb.json',"w") as file:
        json.dump(index_orb, file, cls=NumpyArrayEncoder, indent=4)

if 'moment' in searcher or 'combined' in searcher:
    for q in moment_q_index:
        moment_index.update(q)

    with open('index_moment.json',"w") as file:
        json.dump(moment_index, file, indent=4)
print('END')
