# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:01:27 2015

@author: jenniferstark
"""

from bs4 import BeautifulSoup
import os
import requests
import cv2

# the root URL
baseUrl = 'https://identifyus.org'    

# a list for all the image URLs...note required but might be useful if I want to check something later...   
url_list = []

# a list where data from each individual case gets appended. This will be a list of dictionaries
total_list = []

def get_images(cases):
    for case in cases:
        
        # check if I have the .html file for that case number on my local machine
        if os.path.isfile("./html_files2/case_" + str(case) + '.html') == True:
            
            #if I do, open it as 'f'. Check it is longer that 500 lines. If less than 500, it is a placeholder file for 
            # a case that currently does not exist
            with open(('./html_files2/case_' + str(case) + '.html'), 'r') as f:
                lines = len(f.readlines())
                if lines > 500:
                    
                    # if it is longer, open it as 'f', make it into 'soup' so we can explore the file
                    # using beautifulsoup commands
                    with open(('./html_files2/case_' + str(case) + '.html'), 'r') as f:
                        soup = BeautifulSoup(f)
                        
                        #this count is so that I can save each image associated with a 
                        # case with a number appended to it.
                        count = 0
                        
                        # each time face detection returns a result for each image, it gets 
                        # added to this list which is new for each case
                        little_list = []
                        
                        #each case will have its own dictionary with a 'face' count from 'little_list' 
                        # and a case number
                        individual = {}
                        
                        # love printing!! good to see how we're progressing through the case numbers
                        print(case)
                        
                        # we're going to 'try' to get image data, cos not every case will have images
                        try:
                            # Individual Case level
                            # for every image in current case
                            for row in soup.find_all('td', attrs={'class':'thumbnail'}):
                                for col in row.find_all('img'):
                                    
                                    # create a name to save the image as using case number and the current 'count'
                                    caseImage = soup.find(name='label', attrs={'for':'case_case_number'}).find_next('td').text.strip() + '_' + str(count) + '.jpg'

                                    # increase count by one for the next image loop
                                    count += 1
                                    
                                    # append image url to 'url_list'
                                    url_list.append(col.get('src'))
                                    
                                    # create full image url 
                                    imgUrl = baseUrl + col.get('src') + '/'
                                    
                                    # open the new, currenlty empty image file
                                    with open((caseImage), 'wb') as q:
                                        
                                        # get image from the link
                                        res = requests.get(imgUrl)
                                        
                                        # write the image to file in chunks
                                        for chunk in res.iter_content(100000):
                                            q.write(chunk)
                                            
                                    # call the find_face2 function to check for a face in current image. 
                                    # append answers to 'little_list'
                                    little_list.append(find_face2(caseImage, cascPathFront, cascPathSide))
                                
                            # after looping through all images for current case, append 'little_list' VALUE
                            # to 'face' KEY in 'individual' dictionary, and the case number for that case
                            individual['face'] = little_list
                            individual['case'] = case
                            
                        # if there were no images for current case at all, add the folling to the dictionary for 
                        # current case
                        except: 
                            individual['face'] = 0
                            individual['case'] = case
                            
                        # Add current case data to the 'total_list' 
                        total_list.append(individual)

get_images(range(0, 16000))

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
                                        flags = cv2.CASCADE_SCALE_IMAGE)  # this flag is updated for CV3, so it
                                                                          # is different from the code I based mine on
 
    # if face is detected with the frontalface cascade                                       
    if (len(faces) > 0):
        print("front facing")
        
        # this '1' gets appended to the 'little_list'
        return 1
        
    # if we dont find a frontalface, let's try 'profileface'
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
    
        # if face is detected with profile cascade
        if (len(faces) > 0):
            
            # the cascade only works on right facing images
            print("right facing")
            
            # this '2' gets appended to the 'little_list' for this case
            # this '2' gets replaced with '1' later for the modelling, but I wanted it
            # here so that I could check that it WAS even detecting profile faces
            # and I could check the cases to verify they were correct.
            return 2
            
        # if we dont find a profile face, maybe it is facing the wrong way, so let's flip it:
        else:
            print("flip")
            image_flip = cv2.flip(image,1) # the 1 says vertical axis, 0 would be horizontal axis
            gray = cv2.cvtColor(image_flip, cv2.COLOR_BGR2GRAY)
    
            faces = faceCascade.detectMultiScale(gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(30, 30),
                                                flags = cv2.CASCADE_SCALE_IMAGE)
                       
            # now if we find a profile face, we'll print it WAS left facing and return
            # '2' to be appended to 'little_list'                              
            if (len(faces) > 0):
                print("left facing")
                return 2
                
            # if none of that works, I guess there is no face
            else:
                print("no face!")
                
                # '0' gets appended to the 'little_list'
                return 0  