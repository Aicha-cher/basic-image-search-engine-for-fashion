import numpy as np
import cv2
import json
from tqdm import tqdm
import numpy
from json import JSONEncoder
from matplotlib import pyplot as plt
from scipy import stats

class Bounding_box:
    def __init__(self) -> None:
        self.thresh1 = 52
        self.thresh2 = 26

    def scale(self, image):
        width = int(image.shape[1] / 3)
        height = int(image.shape[0] / 3)
        scr_rescaled = cv2.resize(image, (width, height))
        return scr_rescaled

    def pre_processing(self, image):
        scr_rescaled = self.scale(image)
        # Convert image to gray and blur it
        image_gray = cv2.cvtColor(scr_rescaled, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
        return image_gray

    def edge_detection(self, image):
        image_gray = self.pre_processing(image)
        img_canny = cv2.Canny(image_gray, self.thresh1, self.thresh2)
        kernel = np.ones((5, 5))
        img_dill = cv2.dilate(img_canny, kernel, iterations=1)
        return img_dill

    def get_countours(self, img, ouputimage=None, draw=False):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:
                #cv2.drawContours(ouputimage, contours, 0, (255,0,255),5)
                pr = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.005 * pr, True)
                x, y, w, h = cv2.boundingRect(approx)
                if draw:
                    cv2.rectangle(ouputimage, (x, y), (x + w, y + h), (0, 255, 0), 5)
        try: return x, y, x + w, y + h
        except: return 0, 0, img.shape[1], img.shape[0]


class ColorDescriptor():
    def __init__(self, bins):
        #Storing number of bins for histogram
        self.bins = bins

    def describe(self, image, x_segments, y_segments, color_descriptor, draw=False):
        """
        parm : color_descriptor specify which parameter to use possible options are 'histogram' or 'moment'
        """
        bbox = Bounding_box()
        #Convert the image into hsv and initialize the features to quantify the image
        image = image.astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Obtaining the dimensions and center of the image
        img_dill = bbox.edge_detection(image)
        x, y, w, h = bbox.get_countours(img_dill, draw=False, ouputimage=image)
        segements = []
        delta_x = int((w - x)/x_segments)
        delta_y = int((h - y)/y_segments)
        for i in range(0, x_segments):
            for j in range(0, y_segments):
                segements.extend([(x+i*delta_x, x+(i+1)*delta_x, y+j*delta_y, y+(j+1)*delta_y)])

        #Construct an eliptical mask representing the center of the image which is 75% of height and width of image
        image = bbox.scale(image)
        #Loop over the segements
        mask = []
        features = []
        for(startX, endX, startY, endY) in segements:
            #Construct a mask for each corner of the image subtracting the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            #Extract a color histogram from the image and update the feature vector
            if color_descriptor == 'histogram':
                hist = self.histogram(image, cornerMask)
                features.extend(hist)
            # moment feature
            if color_descriptor =='moment':
                feature = self.moment_descriptor(image[startX: endX, startY:endY])
                features.extend(feature)

            mask.append(cornerMask)
        #Return the feature vector
        if draw:
            return features, mask, segements
        else: return features


    def hist_plot(self, img, _bin, mask=None):
        img_split = cv2.split(img)
        h_hist = cv2.calcHist(img_split, [0], mask, [_bin], (0, 180))
        s_hist = cv2.calcHist(img_split, [1], mask, [_bin], (0, 256))
        v_hist = cv2.calcHist(img_split, [2], mask, [_bin], (0, 256))
        cv2.normalize(h_hist, h_hist, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(s_hist, s_hist, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(v_hist, v_hist, norm_type=cv2.NORM_MINMAX)
        plt.plot(h_hist, color='c')
        plt.plot(s_hist, color='m')
        plt.plot(v_hist, color='y')
        plt.legend()
        plt.show()

    def histogram(self, image, mask):

        #Extract a 3-D color histogram from the masked region of the image, using the number of bins supplied
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        #Returning histogram
        return hist
    def moment_descriptor(self, image):
        """
        color moment descriptor
        """
        mean_h = np.mean(image[0])
        mean_s = np.mean(image[1])
        mean_v = np.mean(image[2])
        std_h = np.std(image[0])
        std_s = np.std(image[1])
        std_v = np.std(image[2])
        skew_h = stats.skew(image[0].reshape(-1))
        skew_s = stats.skew(image[1].reshape(-1))
        skew_v = stats.skew(image[2].reshape(-1))
        feature = [mean_h, mean_s, mean_v, std_h, std_s, std_v, skew_h, skew_s, skew_v]
        return feature

def orb_descriptors(image):
    """
    ORB (Oriented FAST and Rotated BRIEF) descriptor
    """
    detector = cv2.ORB.create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, des = detector.detectAndCompute(gray, None)
    return des


class Searcher:

    def search(self, queryFeatures, path, limit=10, combine=False):
        #make a dictionary for thr results
        results = {}

        #open the index file for reading
        with open(path, 'r') as f:
            index = json.load(f)

            #loop over the rows in the index
            for imageID, feat in tqdm(index.items()):
                # parse out the imageID and features, then calculate the chi-squared distance between the saved features and the features of our image
                features = [float(x) for x in feat]
                distance = self.chi2_distance(features, queryFeatures, 1)

                # now we have the distance between the two feature vectors. we now update the results dictionary
                results[imageID] = distance

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])

        #return our results
        if combine:
            return results, index
        else: return results[:limit], index

    def orb_searcher(self, queryFeatures, path, limit=10, combine=False):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #open the index file for reading
        with open(path, 'r') as f:
            orb_index = json.load(f)
            x = []
            #loop over the rows in the index
            for imageID, feature in tqdm(orb_index.items()):
                matches = bf.match(np.array(feature).astype(np.uint8), queryFeatures)
                #keep top 10 matches
                matches = sorted(matches, key=lambda x: x.distance)[:10]
                x.append([matches, imageID])
            avg_distances = [(np.mean([y.distance for y in match[0]]), match[1]) for match in x]
            avg_distances = sorted(avg_distances, key=lambda x: x[0])
                #return image with lowest distance matches to query image
            if combine:
                return avg_distances, orb_index
            else: return avg_distances[:limit], orb_index

    def combined_search(self, searcher1, searcher2, limit=5):
        combines_results = {}
        searcher1 = sorted(searcher1, key=lambda x: x[1])
        searcher2 = sorted(searcher2, key=lambda x: x[1])
        for distance1, distance2 in zip(searcher1, searcher2):
            if distance1[1] == distance2[1]:
                combined_distance = np.mean([distance1[0], distance2[0]])
                combines_results[distance1[1]] = combined_distance
        # sort the results
        combines_results = sorted([(v, k) for (k, v) in combines_results.items()])
        return combines_results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        #calculating the chi squared distance
        d = 0.5 * np.sum([((a-b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

        #return the chi squared distance
        return d

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def draw_matches(image1, image2, n_matches=10):
    detector = cv2.ORB.create()
    image1 = cv2.resize(cv2.imread(image1), (600, 600))
    image2 = cv2.resize(cv2.imread(image2), (600, 600))
    kp1, des1 = detector.detectAndCompute(image1, None)
    kp2, des2 = detector.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:n_matches], image2, flags=2) # Show top 10 matches
    cv2.imshow('Matches', img_matches)
    cv2.waitKey()


def result(query, results):
    result = {}
    result['query'] = cv2.resize(query, (300, 300))
    for (score, resultID) in results:
        #load the result image and display it
        result[resultID] = cv2.resize(cv2.imread('image_fashion' + "/" + resultID), (300, 300))

    h_stack1 = np.hstack([img for k, img in result.items() if k in list(result.keys())[:3]])
    h_stack2 = np.hstack([img for k, img in result.items() if k in list(result.keys())[3:]])
    v_stack = np.vstack([h_stack1, h_stack2])
    return v_stack