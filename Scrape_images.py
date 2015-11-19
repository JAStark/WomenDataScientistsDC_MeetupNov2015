# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:01:27 2015

@author: jenniferstark
"""

from bs4 import BeautifulSoup
import os
import requests
import cv2
import numpy as np


baseUrl = 'https://identifyus.org'       
url_list = []
total_list = []

def get_images(cases):
    for case in cases:
        if os.path.isfile("./html_files2/case_" + str(case) + '.html') == True:
            with open(('./html_files2/case_' + str(case) + '.html'), 'r') as f:
                lines = len(f.readlines())
                if lines > 500:
                    with open(('./html_files2/case_' + str(case) + '.html'), 'r') as f:
                        soup = BeautifulSoup(f)
                        count = 0
                        little_list = []
                        individual = {}
                        print(case)
                        try:
                            # Individual Case level
                            for row in soup.find_all('td', attrs={'class':'thumbnail'}):
                                for col in row.find_all('img'):
                                    caseImage = soup.find(name='label', attrs={'for':'case_case_number'}).find_next('td').text.strip() + '_' + str(count) + '.jpg'
                                    count += 1 
                                    url_list.append(col.get('src'))
                                    imgUrl = baseUrl + col.get('src') + '/'
                                    with open((caseImage), 'wb') as q:
                                        res = requests.get(imgUrl)
                                        for chunk in res.iter_content(100000):
                                            q.write(chunk)
                                    little_list.append(find_face2(caseImage, cascPathFront, cascPathSide))
                            individual['face'] = little_list
                            individual['case'] = case
                        except: 
                            individual['face'] = 0
                            individual['case'] = case
                        total_list.append(individual)

get_images(range(10000, 16000))

'''
BASED ON https://realpython.com/blog/python/face-recognition-with-python/
    
Using profile lbp cascade and haar frontalface cascade detection classifier
'''
cascPathSide = '/Users/jenniferstark/opencv/data/lbpcascades/lbpcascade_profileface.xml'
cascPathFront =  '/Users/jenniferstark/opencv/data/haarcascades/haarcascade_frontalface_default.xml'


def find_face2(imagePath, cascPathFront, cascPathSide):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPathFront)
    
    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                        flags = cv2.CASCADE_SCALE_IMAGE)
                                        
    if (len(faces) > 0):
        print("front facing")
        return 1
    else:
        print('try side')
        faceCascade = cv2.CascadeClassifier(cascPathSide)
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                        flags = cv2.CASCADE_SCALE_IMAGE)
    
        if (len(faces) > 0):
            print("left facing")
            return 2
        else:
            print("flip")
            image_flip = cv2.flip(image,1) # the 1 says vertical axis, 0 would be horizontal axis
            gray = cv2.cvtColor(image_flip, cv2.COLOR_BGR2GRAY)
    
            faces = faceCascade.detectMultiScale(gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(30, 30),
                                                flags = cv2.CASCADE_SCALE_IMAGE)
                                                    
            if (len(faces) > 0):
                print("right facing")
                return 2
            else:
                print("no face!")
                return 0  