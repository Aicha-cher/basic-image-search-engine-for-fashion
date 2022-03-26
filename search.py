
import argparse
import json
import os

import cv2
import numpy as np

from tools import ColorDescriptor, NumpyArrayEncoder, Searcher, orb_descriptors

# creating the argument parser and parsing the arguments
arg = argparse.ArgumentParser()
arg.add_argument("-Q", "--query", nargs='+', required=True, help="Path to the query image")
arg.add_argument("-M", "--method", nargs='+', required=True, help="search method list")
arg.add_argument("-P", "--plot", action='count', required=False, help="Plot or not the resulted image")
arg.add_argument("-S", "--save", action='count', required=False, help="save or not the resulted image")
args = arg.parse_args()

queries = args.query
searcher = args.method
plot = args.plot
_save = args.save


def result(query, results):
    result = {}

    image = cv2.resize(query, (300, 300))
    cv2.rectangle(image, (0, 0), (300, 300), (0, 0, 255), 5)
    result['query'] = image
    for (score, resultID) in results:
        #load the result image and display it
        result[resultID] = cv2.resize(cv2.imread('image_fashion' + "/" + resultID), (300, 300))

    h_stack1 = np.hstack([img for k, img in result.items() if k in list(result.keys())[:3]])
    h_stack2 = np.hstack([img for k, img in result.items() if k in list(result.keys())[3:]])
    v_stack = np.vstack([h_stack1, h_stack2])
    return v_stack


def plotting(query, results, window_name):
    v_stack = result(query, results)
    cv2.imshow(window_name, v_stack)
    cv2.waitKey(0)
    cv2.imwrite(f'resulted_image/{window_name}.jpg', v_stack)


def save(query, results, id, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    v_stack = result(query, results)
    cv2.imwrite(path + f'results_{id}.jpg', v_stack)

#intializing the color descriptor
cd = ColorDescriptor((16, 4, 4))


moment_q_index, hist_q_index, orb_q_index = [], [], []
for q in queries:
    query = cv2.imread(q)
    imgId = q[q.rfind("/")+1:]
    #performing the search
    s1 = Searcher()
    if 'histogram' in searcher:
        queryFeatures = cd.describe(query, x_segments=2, y_segments=2, color_descriptor='histogram')
        #queryFeatures = [str(f) for f in queryFeatures]
        hist_results, hist_index = s1.search(queryFeatures, limit=5, combine=False, path='index.json')
        hist_q_index.append({imgId: queryFeatures})
        if plot:
            plotting(query, hist_results, window_name='hist_results')
        if _save:
            save(query, hist_results, id=imgId, path='resulted_image/histogram/')

    if 'orb' in searcher:
        queryFeatures_orb = orb_descriptors(query)
        orb_results, index_orb = s1.orb_searcher(queryFeatures_orb, path='index_orb.json', limit=5, combine=False)
        orb_q_index.append({imgId: queryFeatures_orb})
        if plot:
            plotting(query, orb_results, window_name='orb_results')
        if _save:
            save(query, orb_results, id=imgId, path='resulted_image/orb/')

    if 'moment' in searcher:
        queryFeatures_moment = cd.describe(query, x_segments=2, y_segments=2, color_descriptor='moment')
        moment_results, moment_index = s1.search(queryFeatures_moment, path='index_moment.json', limit=5, combine=False)
        moment_q_index.append({imgId: queryFeatures_moment})
        if plot:
            plotting(query, moment_results, window_name='moment_results')
        if _save:
            save(query, moment_results, id=imgId, path='resulted_image/moment/')

    if 'combined' in searcher:
        queryFeatures_moment = cd.describe(query, x_segments=2, y_segments=2, color_descriptor='moment')
        moment_results, moment_index = s1.search(queryFeatures_moment, path='index_moment.json', limit=5, combine=True)
        moment_q_index.append({imgId: queryFeatures_moment})
        queryFeatures_orb = orb_descriptors(query)
        orb_results, index_orb = s1.orb_searcher(queryFeatures_orb, path='index_orb.json', limit=5, combine=True)
        orb_q_index.append({imgId: queryFeatures_orb})
        combined_results = s1.combined_search(moment_results, orb_results)
        if plot:
            plotting(query, combined_results, window_name='combined_results')
        if _save:
            save(query, combined_results, id=imgId, path='resulted_image/combined/')
    cv2.destroyAllWindows()
#save the query index
if 'histogram' in searcher:
    for q in hist_q_index:
        hist_index.update(q)

    with open('index.json', "w") as file:
        json.dump(hist_index, file, indent=4)

if 'orb' in searcher or 'combined' in searcher:
    for q in orb_q_index:
        index_orb.update(q)

    with open('index_orb.json', "w") as file:
        json.dump(index_orb, file, cls=NumpyArrayEncoder, indent=4)

if 'moment' in searcher or 'combined' in searcher:
    for q in moment_q_index:
        moment_index.update(q)

    with open('index_moment.json', "w") as file:
        json.dump(moment_index, file, indent=4)
print('END')
