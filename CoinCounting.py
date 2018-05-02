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
import math # for sqr root  operations

## Coin Counting program that detects the main U.S. coin types and sums their value 
#
# Run this module after the "CoinCalibration" module, since it will access files created by "CoinCalibration"
# The "CoinCalibration" stores color and size data about the coins
# Every time the environment is changed, old calibration files should be deleted, and calibration should be re-run
# Calibration files are stored in "/jevois/data"
#
# If you do not want to use the calibration module, modify the parameters in the constructor
# Make sure you provide your own values to help this program distinguish between different types of coins

# It is important to properly secure the JeVois in a stable position directly above the area coins will be placed
# The module will perform best when solid white or black backgrounds are used
#
# This module detects the coins by using a Simple Blob detector
# There are many other algorithms that can be substituted, examples are included in the related tutorial
# This module along with the "CoinCalibration" module are meant to be used as a way to become familiar with OpenCV and
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
# @displayname Total Value of Coins: $--.--
# @videomapping YUYV 640 480 37.0 YUYV 640 480 37.0 JeVois CoinCalibration
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
class CoinCounting:
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
        self.threshWhiteBackground = cv2.THRESH_BINARY + cv2.THRESH_OTSU      # threshold type 
        self.threshBlackBackground = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU  # threshold type 
        self.kernelWidth = 5                                                  # width for blur
        self.kernelHeight = 5                                                 # height for blur

        # Parameters: instructions and marker placement 
        self.xVal = 80                            # x coordinate for first marker
        self.yVal = 300                           # y coordinate for first marker
        self.xDelta = 140                         # The distance between coins 
        self.yDelta = -100                        # The distance between coins and above text

        # Parameters:  adding coin statistics for trouble shooting (currently commented out)
        self.deltaY = 20                               # spacing between statistics
        self.initialY = 140                            # initial x coordinate 

        # Parameters: calibration will be used 
        self.calibration = True 
        # If the above is changed to false, you must supplement the information below:
        # self.pennyValues = np.array([radiusAvg,rgAvg,rbAvg,standardDeviation,\
        # standardDeviationRG,standardDeviationRB])
        # self.nickelValues = np.array([radiusAvg,rgAvg,rbAvg,standardDeviation,\
        # standardDeviationRG,standardDeviationRB])
        # self.dimeValues = np.array([radiusAvg,rgAvg,rbAvg,standardDeviation,\
        # standardDeviationRG,standardDeviationRB])
        # self.quarterValues = np.array([radiusAvg,rgAvg,rbAvg,standardDeviation,\
        # standardDeviationRG,standardDeviationRB])
     
    # ###################################################################################################
    ## Get the data values stored by the CoinCalibration program and assign values to appropriate coin
    def coinValues(self,coin):
        # Instantiate variables
        # The number of coins for which data is stored (number of lines in file)
        coinNum = 0
        # A running sum of the R:G ratio
        rgSum = 0
         # A running square of sums of the R:G ratio
        rgSqSum = 0
        # A running sum of the R:B ratio
        rbSum = 0
        # A running square of sums of the R:B ratio
        rbSqSum = 0
        # A running sum of the radii
        radiusSum = 0
        # A running square of sums of the radii
        radiusSqSum = 0
        
        # Open file containing R:G data and compute related variables
        with open("/jevois/data/" + coin + "_RG.txt","r") as fo:
            for line in fo:
                line = line.strip()
                if line:
                     rgSum = rgSum + float(line)
                     rgSqSum = float(line) * float(line) + rgSqSum
                     coinNum = coinNum + 1
        
        # Computed Statistics
        rgAvg = rgSum/coinNum
        rgSqAvg = rgSqSum/coinNum
        varianceRG = rgSqAvg - rgAvg*rgAvg
        standardDeviationRG = math.sqrt(varianceRG)

        # Open file containing R:B data and compute related variables
        with open("/jevois/data/" + coin + "_RB.txt","r") as fo:
            for line in fo: 
                line = line.strip()
                if line:
                     rbSum = rbSum + float(line)
                     rbSqSum = float(line) * float(line) + rbSqSum
        
        # Computed Statistics
        rbAvg = rbSum/coinNum
        rbSqAvg = rbSqSum/coinNum
        varianceRB = rbSqAvg - rbAvg*rbAvg
        standardDeviationRB = math.sqrt(varianceRB)

        # Open file containing radius data and compute related variables
        with open("/jevois/data/" + coin + "_Radius.txt","r") as fo:
            for line in fo: 
                line = line.strip()
                if line:
                     radiusSum = radiusSum + float(line)
                     radiusSqSum = float(line) * float(line) + radiusSqSum
    
        # Computed Statistics
        radiusAvg = radiusSum/coinNum
        radiusSqAvg = radiusSqSum/coinNum
        variance = radiusSqAvg - radiusAvg*radiusAvg
        standardDeviation = math.sqrt(variance)


        # Create an array storing this data and return array
        coinValues = np.array([radiusAvg,rgAvg,rbAvg,standardDeviation,\
            standardDeviationRG,standardDeviationRB])
        return coinValues

    # ###################################################################################################
    ## Add statistics to output image for trouble-shooting
    def addCoinStats(self,inimg,values,Coin,initialY,deltaY):
        # Adds statistics text to the screen
        # Can be used for seeing real-time values of coin data
        # Typically not used unless trouble-shooting
        if Coin == 'Penny' or Coin == 'Dime':
            cv2.putText(inimg, str(Coin) + " Radius " + str("%.2f" % values[0]),\
                (20, initialY+deltaY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " RG " + str("%.2f" % values[1]), (20, initialY+deltaY*2),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " RB " + str("%.3f" % values[2]), (20, initialY+deltaY*3),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " Radius Standard Dev " + str("%.3f" % values[3]),\
                (20, initialY+deltaY*4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " RG Standard Dev" + str("%.3f" % values[4]),\
                (20, initialY+deltaY*5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " RB Standard Dev " + str("%.3f" % values[5]),\
                (20, initialY+deltaY*6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        else:
            cv2.putText(inimg, str(Coin) + " Radius " + str("%.2f" % values[0]),\
                (20, initialY+deltaY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " Standard Dev " + str("%.3f" % values[3]),\
                (20, initialY+deltaY*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return inimg
        
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

        # Draw keypoints (used in tutorial, but we will later draw over this so commented out)
        # im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255, 0, 0),
        #                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Create array to store corresponding coin data
        if (self.calibration):
            pennyValues = self.coinValues('Penny')
            nickelValues = self.coinValues('Nickel')
            dimeValues = self.coinValues('Dime')
            quarterValues = self.coinValues('Quarter')

        # If Calibration module is NOT used, user must specify these values in the constructor 
        else:
            pennyValues = self.pennyValues
            nickelValues = self.nickelValues
            dimeValues = self.dimeValues
            quarterValues = self.quarterValues

        # Create variables to store number of each type of coin
        pennyNum = 0
        nickelNum = 0
        dimeNum = 0
        quarterNum = 0

        # Create a loop that iterates once for each detected blob
        for x in range(0,len(keypoints)):
            im_with_keypoints=cv2.circle(im_with_keypoints, (np.int(keypoints[x].pt[0]),\
                np.int(keypoints[x].pt[1])),radius=np.int(keypoints[x].size/2),\
            color=(0,0,255), thickness=2)

            # Get RGB values and radius from coin
            # We use a circular area of half the radius of the coin to get the average RGB value
            radius = np.int(keypoints[x].size/2)
            center = (np.int(keypoints[x].pt[0])-5,np.int(keypoints[x].pt[1]))
            circle_img = np.zeros((height, width), np.uint8)
            cv2.circle(circle_img, center, int(radius/2), [255,255,255], -1)
            circle_img = np.uint8(circle_img)

            # Compute the mean RGB value in this circle
            mean_val = cv2.mean(img, circle_img)[::-1]

            # Store these mean RGB values and use them to compute the ratios
            temp = mean_val[1:]
            red = temp[0]
            green = temp[1]
            blue = temp[2]
            ratioRG = red/green
            ratioRB = blue/green
            # If you want, you can add more heuristics, like the rgb sqrd value, which is commented out  
            # rgbSqrd =  math.sqrt(red*red + blue*blue + green*green)

            # To distinguish between nickels and quarters, we rely solely on radius
            # If the radius is greater than the average quarter radius, label this coin as a quarter
            if (radius > quarterValues[0]):
                cv2.putText(im_with_keypoints, "Q", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                quarterNum += 1
            # Assign the coin type based on whichever coin radius is closest in value
            elif (radius > nickelValues[0]):
                # For example: 
                # If the radius is greater than the average nickel radius, and closer to the average 
                # nickel radius than quarter radius, label this as as nickel
                if ((quarterValues[0] - radius) > (radius - nickelValues[0])):
                    cv2.putText(im_with_keypoints, "N", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    nickelNum += 1
                # For example: 
                # If the radius is greater than the average nickel radius,
                # but closer to the average quarter radius, label as a quarter                
                else:
                    cv2.putText(im_with_keypoints, "Q", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    quarterNum += 1
            

            # To distinguish between nickels and pennies, we rely largely on the R:G ratio
            # If you want, you can add in other dependencies, such as using the R:B ratio or the squared RGB values 
            elif (radius > pennyValues[0]):
                # If the average nickel and penny radius are within 15%, 
                # they are too similar, so we rely solely on the R:G ratio
                # Assign the coin type based on whichever coin has the closest R:G ratio value
                if (abs(nickelValues[0]-pennyValues[0])/pennyValues[0] < 0.15):
                    if ((abs(nickelValues[1]-ratioRG)) > (abs(pennyValues[1]-ratioRG)) ):
                        cv2.putText(im_with_keypoints, "P", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        pennyNum += 1
                    else:
                        cv2.putText(im_with_keypoints, "N", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        nickelNum += 1
                # If the average nickel and penny radius are greater than 15%, we rely on the radius
                elif ((nickelValues[0] - radius) > (radius - pennyValues[0])):
                    cv2.putText(im_with_keypoints, "P", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    pennyNum += 1
                else:
                    cv2.putText(im_with_keypoints, "N", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    nickelNum += 1

            # If the radius is less the the average dime radius, label as dime
            elif(radius < dimeValues[0]):
                cv2.putText(im_with_keypoints, "D", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                dimeNum += 1
            
            # To distinguish between dimes and pennies, we rely largely on the R:G ratio
            elif (abs(dimeValues[0]-pennyValues[0])/pennyValues[0] < 0.15):
                # If the average nickel and penny radius are within 15%, we rely solely on the R:G ratio
                # Assign the coin type based on whichever coin has the closest R:G ratio value
                if ((abs(pennyValues[1]-ratioRG)) > (abs(dimeValues[1]-ratioRG)) ):
                        cv2.putText(im_with_keypoints, "D", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        dimeNum += 1
                else:
                        cv2.putText(im_with_keypoints, "P", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        pennyNum += 1

            # If the average dime and penny radius are greater than 15%, we rely on the radius
            elif ((radius - dimeValues[0]) > (pennyValues[0]-radius)):
                cv2.putText(im_with_keypoints, "P", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                pennyNum += 1
            else:
                cv2.putText(im_with_keypoints, "D", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                dimeNum += 1

        # Compute the total value of all coins on screen
        totalVal = pennyNum*0.01 + dimeNum*0.1 + nickelNum*0.05 + quarterNum*0.25


        # Write out the total value and the number of each type of coin on the screen
        cv2.putText(im_with_keypoints, "Total Value of Coins: $" + str("%.2f" % totalVal), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(im_with_keypoints, "Pennies: " + str(pennyNum), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(im_with_keypoints, "Nickels: " + str(nickelNum), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(im_with_keypoints, "Dimes: " + str(dimeNum), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(im_with_keypoints, "Quarters: " + str(quarterNum), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


        # For troubleshooting, you can uncomment these lines and see what values you are basing coin identification on
        #cv2.putText(im_with_keypoints, "STATISTICS", (20, self.initialY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        #inimg = self.addCoinStats(im_with_keypoints,dimeValues,'Dime',self.initialY,self.deltaY)
        #inimg = self.addCoinStats(im_with_keypoints,pennyValues,'Penny',self.initialY+self.deltaY*6,self.deltaY)
        #inimg = self.addCoinStats(im_with_keypoints,nickelValues,'Nickel',self.initialY+self.deltaY*12,self.deltaY)
        #inimg = self.addCoinStats(im_with_keypoints,quarterValues,'Quarter',self.initialY+self.deltaY*14,self.deltaY)
    
        # Convert our BGR image to video output format and send to host over USB:
        outframe.sendCvBGR(im_with_keypoints)
