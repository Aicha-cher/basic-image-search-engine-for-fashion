import numpy as np
import cv2
import json
from tqdm import tqdm
import numpy
from json import JSONEncoder

class Bounding_box:
    def __init__(self) -> None:
        self.thresh1=52
        self.thresh2=26
    def scale(self, image):
        width = int(image.shape[1] / 3)
        height = int(image.shape[0] / 3)
        scr_rescaled = cv2.resize(image, (width, height))
        return scr_rescaled

    def pre_processing(self, image):
        scr_rescaled = self.scale(image)
        # Convert image to gray and blur it
        image_gray = cv2.cvtColor(scr_rescaled, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.GaussianBlur(image_gray, (3,3),0)
        return image_gray

    def edge_detection(self, image):
        image_gray = self.pre_processing(image)
        img_canny = cv2.Canny(image_gray, self.thresh1, self.thresh2)
        kernel = np.ones((5,5))
        img_dill = cv2.dilate(img_canny,kernel, iterations=1)
        return img_dill

    def get_countours(self,img, ouputimage=None, draw=False):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:
                cv2.drawContours(ouputimage, contours, 0, (255,0,255),5)
                pr = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.005 * pr, True)
                x, y, w, h = cv2.boundingRect(approx)
                if draw:
                    cv2.rectangle(ouputimage, (x,y),(x + w, y + h), (0,255,0), 5)
        try: return x,y,w,h
        except: return 0,0,img.shape[1],img.shape[0]


class ColorDescriptor():
    def __init__(self, bins):
        #Storing number of bins for histogram
        self.bins = bins

    def describe(self, image):
        bbox = Bounding_box()
        #Convert the image into hsv and initialize the features to quantify the image
        image = image.astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        # Obtaining the dimensions and center of the image
        img_dill = bbox.edge_detection(image)
        x, y, h, w = bbox.get_countours(img_dill, draw=True, ouputimage=image )
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        #Divide the image into 4 segements(top-left,top-right,bottom-left,bottom-right,center)
        segements = [(x,cX,y,cY),(cX,w,y,cY),(cX,w,cY,h),(x,cX,cY,h)]
        image = bbox.scale(image)
        #Construct an eliptical mask representing the center of the image which is 75% of height and width of image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        #Loop over the segements
        for(startX, endX, startY, endY) in segements:
            #Construct a mask for each corner of the image subtracting the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255 ,-1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            #Extract a color histogram from the image and update the feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        #Extract a color histogram from the elliptical region and update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        #Return the feature vector
        return features

    def histogram(self,image,mask):

        #Extract a 3-D color histogram from the masked region of the image, using the number of bins supplied
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])

        hist = cv2.normalize(hist, hist).flatten()

        #Returning histogram
        return hist

def orb_descriptors(image):
    """
    ORB (Oriented FAST and Rotated BRIEF) descriptor
    """
    detector = cv2.ORB.create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,des = detector.detectAndCompute(gray, None)
    return des

class Searcher:
    def __init__(self,indexPath):
        #storing the index
        self.indexPath = indexPath

    def search(self,queryFeatures, limit=10):
        #make a dictionary for thr results
        results = {}

        #open the index file for reading
        with open(self.indexPath, 'r') as f:
            index = json.load(f)

            #loop over the rows in the index
            for imageID, feat in tqdm(index.items()):
                # parse out the imageID and features, then calculate the chi-squared distance between the saved features and the features of our image
                features = [float(x) for x in feat]
                distance = self.chi2_distance(features, queryFeatures)

                # now we have the distance between the two feature vectors. we now update the results dictionary
                results[imageID] = distance

        # sort the results
        results = sorted([(v,k) for (k,v) in results.items()])

        #return our results
        return results[:limit]

    def orb_searcher(self, queryFeatures, path, limit=10 ):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #open the index file for reading
        with open(path, 'r') as f:
            orb_index = json.load(f)
            x = []
            #loop over the rows in the index
            for imageID, descriptors in tqdm(orb_index.items()):
                matches = bf.match(np.array(descriptors).astype(np.uint8), queryFeatures)
                #keep top 10 matches
                matches = sorted(matches, key = lambda x: x.distance)[:10]
                x.append([matches, imageID])
                avg_distances = [(np.mean([y.distance for y in match[0]]), match[1]) for match in x]
                avg_distances = sorted(avg_distances, key=lambda x: x[0])
                #return image with lowest distance matches to query image
            return avg_distances[:limit]


    def chi2_distance(self, histA, histB, eps = 1e-10):
        #calculating the chi squared distance
        d = 0.5 * np.sum([((a-b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

        #return the chi squared distance
        return d
def draw_matches(image1, image2, n_matches=10):
  detector = cv2.ORB.create()
  image1 = cv2.imread(image1)
  image2 = cv2.imread(image2)
  kp1, des1 = detector.detectAndCompute(image1, None)
  kp2, des2 = detector.detectAndCompute(image2, None)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1, des2)
  matches = sorted(matches, key = lambda x: x.distance)
  img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:n_matches], image2, flags=2) # Show top 10 matches
  cv2.imshow('Matches', img_matches)
  cv2.waitKey()

class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, numpy.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)