######################################################################################################################
#
# JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2017 by Laurent Itti, the University of Southern
# California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
#
# This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
# redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
# Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.  You should have received a copy of the GNU General Public License along with this program;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
# Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
######################################################################################################################

import libjevois as jevois
import cv2
import numpy as np
import math # for absolute value operations

## Calibration program that creates data files with color and size information for each main U.S. coin type
#
# This module should be run prior to running the "CoinCounting" module
# This module asks the user to place their coins in order and will then extract size and RGB values
# These values will be written into a file in the "/jevois/data" folder
# This file will then be accessed by the "CoinCounting" program
# The CoinCounting program will use this data to distinguish between coins
#
# Is it optional to run this module prior to using the "Coin Counting" module
# Instead of using this module, users can instead manually set the parameters in the "Coin Counting" module
#
# It is important to properly secure the JeVois in a stable position directly above the area coins will be placed
# The module will perform best when solid white or black backgrounds are used
#
# This module detects the coins by using a Simple Blob detector
# There are many other algorithms that can be substituted, examples are included in the related tutorial
# This module along with the "Coin Counting" module are meant to be used as a way to become familiar with OpenCV and
# with different  computer vision algorithms.
# 
#  Using this module
#  -----------------
#
# Check out [this tutorial](https://me-ghana.github.io/Coin-Counting/)
# For a shorter tutorial, check out this brief [user guide](https://me-ghana.github.io/Coin-Counting-User-Guide/)
#
# [Here's a link to the code on github](https://github.com/Me-ghana/Coin-Counter)
#
# Trying it out
# -------------
#
# Download the code from github and edit parameters on your machine. When ready, access the JeVois USB and edit the 
# configuration file in JeVois and load the module.  
#
# @author Laurent Itti
# 
# @displayname COIN CALIBRATION
# @videomapping YUYV 640 480 34.0 YUYV 640 480 34.0 JeVois CoinCalibration
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class CoinCalibration:
     # ###################################################################################################
     ## Constructor
     def __init__(self):
          # Instantiate a circular blob detector:
          params = cv2.SimpleBlobDetector_Params()
          params.filterByCircularity = True
          params.filterByArea = True
          params.minArea = 200.0
          self.detector = cv2.SimpleBlobDetector_create(params)

          # Parameters: pre-processing 
          self.threshWhiteBackground = cv2.THRESH_BINARY + cv2.THRESH_OTSU           # threshold type 
          self.threshBlackBackground = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU       # threshold type 
          self.kernelWidth = 5                                                       # width for blur
          self.kernelHeight = 5                                                      # height for blur

          # Parameters: instructions and marker placement 
          self.xVal = 80                               # x coordinate for first marker
          self.yVal = 300                              # y coordinate for first marker
          self.xDelta = 140                            # The distance between coins 
          self.yDelta = -100                           # The distance between coins and above text

     # ###################################################################################################
     ## Add text instructions on the image to help user place each coin type in the correct location
     def imageText(self, image, xVal, yVal, xDelta, yDelta):
          # Parameters: 
          # xVal and yVal are the right-most X and Y coordinates for coin placement
          # xDelta and yDelta are the distances by which each coin should be separated
          # text will be put on image, and image will be sent to the video display interface
          
          # Add title and instruction
          cv2.putText(image, "COIN CALIBRATION", (20, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          cv2.putText(image, "Center the corresponding coin at each X",(20, 40), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

          # Add coin names from left to right on the screen
          cv2.putText(image, "Dime", (xVal, yVal), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          cv2.putText(image, "Penny", (xVal + xDelta, yVal), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          cv2.putText(image, "Nickel", (xVal + 2*xDelta, yVal), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          cv2.putText(image, "Quarter", (xVal + 3*xDelta, yVal), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          
          # Add markers, or "X"s, where coins should be placed from left to right
          cv2.putText(image, "X", (xVal + 20, yVal + yDelta), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          cv2.putText(image, "X", (xVal + xDelta + 20, yVal + yDelta), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          cv2.putText(image, "X", (xVal + 2*xDelta + 20, yVal + yDelta), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          cv2.putText(image, "X", (xVal + 3*xDelta + 20, yVal + yDelta), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

          # All text has been added and image can now be returned
          return image

     # ###################################################################################################
     ##  Returns the name of the coin as a string given an integer
     def printCoinType(self, type):
          # Parameters:
          # type is the integer value of the closest marker
          # 0 is the leftmost marker while 3 is the rightmost marker 

          # Function returns the following strings corresponding to the integer type
          # Dime - 0
          # Penny - 1
          # Nickel - 2
          # Quarter - 3

          if type == 0:
               coin = 'Dime'
          if type == 1:
               coin = 'Penny'
          if type == 2:
               coin = 'Nickel'
          if type == 3:
               coin = 'Quarter'
          return coin

     # ###################################################################################################
     ##  Store R:G ratio, R:B ratio, and radius info data in three text files for each coin 
     def writeToFile(self,ratioRG,ratioRB,radius,coin):
          # Parameters:
          # ratio RG, ratio RB, and radius are the R:G, R:B, and radius (in pixels) value for the coin
          # coin specifies the type of coin (penny, dime, nickel, or quarter)

          # CAUTION: this function APPENDS to already existing data.
          # If you want to store calibration for a new environment, make sure to delete old files first
          # Go to /jevois/data/ and delete all coin related files

          fo = open("/jevois/data/" + coin+"_RG.txt","a")
          fo.write(str(ratioRG) + "\n")
          fo.close()

          fo = open("/jevois/data/" + coin+"_RB.txt","a")
          fo.write(str(ratioRB) + "\n")
          fo.close()

          fo = open("/jevois/data/" + coin+"_Radius.txt","a")
          fo.write(str(radius) + "\n")
          fo.close()  
     
     # ###################################################################################################
     ## Detects the type of coin based on the closest marker
     def detectCoinType(self, image, circleCenterXCoord,circleCenterYCoord, \
          targetXVal, targetYVal, xDelta, radius):  
          # Parameters:
          # (circleCenterXCoord,circleCenterYCoord) are the (x,y) coordinates of the coin center
          # (targetXVal, targetYVal) Pass in the (x,y) coordinates of the left most marker 
          # ratio RG, ratio RB, and radius are the R:G, R:B, and radius (in pixels) value for the coin
          # xDelta is the target spacing between coins
          # radius is coin radius in pixels

          # Get the height and width of the image
          height,width,depth = image.shape
          
          # Check each marker from left to right and see which the coin is closest too
          for i in range(0,4):
               # Calculate % difference b/w center of detected circle and center of coin target
               xDifference = abs(circleCenterXCoord-targetXVal)/targetXVal
               yDifference = abs(circleCenterYCoord-targetYVal)/targetYVal
         
               # Check if center is w/in 30% of the target marker
               if (xDifference < 0.3) and (yDifference < 0.3):              
                    # Call the printCoinType function to convert the coin type integer to a string name
                    coin = self.printCoinType(i)

                    # Get RGB values from the coin
                    # Radius alone isn't a great way to distinguish between the coins
                    # Looking at the R:G and R:B ratios helps distinguish pennies from the rest
                    # We use a circular area of half the radius to get the average RGB value
                    circle_img = np.zeros((height, width), np.uint8)
                    cv2.circle(circle_img, (circleCenterXCoord, circleCenterYCoord),\
                    int(radius/2), [255,255,255], -1)
                    circle_img = np.uint8(circle_img)
                    
                    # Compute the mean RGB value in this circle
                    mean_val = cv2.mean(image, circle_img)[::-1]

                    # Store these mean RGB values and use them to compute the ratios
                    temp = mean_val[1:]
                    red = temp[0]
                    green = temp[1]
                    blue = temp[2]
                    ratioRG = red/green
                    ratioRB = red/blue

                    # You can add more heuristics if you want, below is a commented out example 
                    # rgbSqrd =  math.sqrt(red*red + blue*blue + green*green)

                    # Write out the ratios and radii information for each coin
                    self.writeToFile(ratioRG,ratioRB,radius,coin)
                    
                    # Write out the calculated values on the screen
                    # This can help you troubleshoot if you notice the values are not what you expect
                    cv2.putText(image, "RG " + str("%.2f" % ratioRG), (targetXVal, targetYVal - 105), \
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(image, "RB " + str("%.2f" % ratioRB), (targetXVal, targetYVal - 130), \
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(image, "radius " + str("%.2f" % radius), (targetXVal, targetYVal - 80), \
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Write out the detected coin type values on the screen
                    # This allows you to double check you are detecting the expected coin
                    cv2.putText(image, coin, (targetXVal, targetYVal - 60), \
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

          
                    # We have detected the coin and added the necessary text to the image                    
                    # Now we can exit this function
                    return image
     
               # Every time the loops is incremented, we shift one marker to the left
               # As a result, we have to update the x coordinate of the marker
               targetXVal = targetXVal + xDelta
     
          # In this case, no coin was detected within 30% of a marker, and no data was stored                   
          return image

     # ###################################################################################################
     ## Process function with USB output
     def process(self, inframe, outframe):
          # Get the next camera image, may block until it is captured
          # Convert it to OpenCV BGR (for color output):
          img = inframe.getCvBGR()
          
          # Also convert it to grayscale for processing:
          grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
          # Get image width, height:
          height, width = grayImage.shape
          
          # Filter noise
          grayImage = cv2.GaussianBlur(grayImage, (self.kernelWidth, self.kernelWidth), 0, 0)
       
          # Apply automatic threshold
          ret, threshImage = cv2.threshold(grayImage, 0, 255, self.threshWhiteBackground)

          # Determine if the background is white or black, and then apply appropriate threshold
          whitePixels = cv2.countNonZero(threshImage)
          blackPixels = height*width - whitePixels

          if (blackPixels > whitePixels):
               ret, threshImage = cv2.threshold(grayImage, 0, 255, self.threshBlackBackground)

          # Blob detection
          keypoints = self.detector.detect(threshImage)
          nrOfBlobs = len(keypoints)
          
          # Draw keypoints
          im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
          
          # Create a loop that iterates once for each detected blob
          for x in range(0,len(keypoints)):
               # Draw a green border around the detected blob
               im_with_keypoints=cv2.circle(im_with_keypoints, (np.int(keypoints[x].pt[0]),\
                    np.int(keypoints[x].pt[1])), radius=np.int(keypoints[x].size/2), color=(0,0,255), thickness=2)
               
               # Call the "detectCoinType" function, to assign a coin type basd on the closest marker
               im_with_keypoints = self.detectCoinType(im_with_keypoints, np.int(keypoints[x].pt[0]),\
                    np.int(keypoints[x].pt[1]), self.xVal + 20 , self.yVal + self.yDelta, \
                    self.xDelta, np.int(keypoints[x].size/2))

          # Add text on the image to make more user-friendly
          im_with_keypoints = self.imageText(im_with_keypoints, self.xVal, self.yVal, self.xDelta, self.yDelta)

          # Convert our BGR image to video output format and send to host over USB:
          outframe.sendCvBGR(im_with_keypoints)