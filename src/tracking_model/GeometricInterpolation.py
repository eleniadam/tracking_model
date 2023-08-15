
# Lisa Brown
# Jan 2023
# Code to DrawContours from label image onto raw image
# and Perform Geometric Interpolation:
#   start with label image (anisotropic)
#   create new label image at higher resolution (isotropic)


import os
import cv2
import numpy as np
import tifffile as tiff
import copy
import math
#import h5py
import argparse

def DrawContourForLabel(label_im,im,ilabel,icolor):
    nslices,h,w = label_im.shape
    for islice in range(1,nslices):
        mask_indices = np.where(label_im[islice,:,:] == ilabel)
        #print(mask_indices[0].shape)
        this_mask_slice = np.zeros([h,w],dtype=np.uint8)
        this_mask_slice[mask_indices] = 255
        ret, thresh = cv2.threshold(this_mask_slice, 1, 255, 0, cv2.THRESH_BINARY)
        threshcopy = thresh.copy()
        contours, hierarchy = cv2.findContours(threshcopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #if (len(contours) > 0):
        #    print('number of contours ',len(contours)) # draw on image slice
        im[islice,:] = cv2.drawContours(im[islice,:], contours, -1, icolor, 3)
    return im



# one tube per label - just gives you the mask index
# from mask index - for each slice we need the mask..
# from mask - get points on border for two slices
# then interpolate to get masks for slices in between
# create tubes with new slice between each slice based on interpolating
def InterpolateTubes(label_img,upscale_factor):
    d,h,w = label_img.shape
    new_label_img = np.zeros([d*upscale_factor,h,w],dtype=np.uint8)
    unique_labels = np.unique(label_img)
    #print('unique labels ',unique_labels)
    for ilabel in unique_labels:  # for each tube
        if (ilabel != 0):
            ind = np.where(label_img == ilabel)
            minSlice = np.min(ind[0])
            # maybe we should get first larger 67slice with no pixels
            maxSlice = np.max(ind[0]) # this imyplies there are pixels in this slice (not in an inbetween slice ?)
            #print('ilabel min/max slice ',ilabel,minSlice,maxSlice)
            for islice in range(minSlice+1, maxSlice):
                # first slice with no pixels
                imask = np.where(label_img[islice, :, :] == ilabel)
                #print(islice,len(imask[0]))
                if (len(imask[0]) < 1):
                    maxSlice = islice
                    #print('new max slice ',maxSlice)
                    continue
            for islice in range(minSlice+1, maxSlice):
                imask1 = np.where(label_img[islice,:,:] == ilabel)
                imask2 = np.where(label_img[islice+1,:,:] == ilabel)
                #print('islice ',islice)
                nAngles = 100 # should be 100
                #print('len msk1 and msk2 ', len(imask1[0]), len(imask2[0]))
                if (len(imask1[0]) < 1):
                    pts1 = []
                else:
                    pts1 = GetMaskContourPointsForInterpolation(imask1,nAngles,ilabel,islice,h,w,label_img)
                if (len(imask2[0]) < 1):
                    pts2 = []
                    #print('returning imask2 no pixels',islice)
                    #continue
                    #return new_label_img
                else:
                    pts2 = GetMaskContourPointsForInterpolation(imask2,nAngles,ilabel,islice+1,h,w,label_img)
                    #print('len pts1 and pts2 ',len(pts1),len(pts2))
                    nNewSlices = upscale_factor
                    new_label_img = InterpolatePoints(pts1,pts2,nNewSlices,islice*nNewSlices, ilabel, new_label_img)
    return new_label_img

# nNewSlice is the upscale factor
# iStartSlice is the current slice we are expanding to
# p1 are the points from mask in below slice
# p2 are the points from mask in the above slice
def InterpolatePoints(p1,p2,nNewSlices,iStartSlice, label, label_img):
    #print('Interpolating Points')
    n = len(p1)
    h = label_img.shape[1]
    w = label_img.shape[2]
    #print('w,h ',w,h)

    for j in range(0,nNewSlices+1): # for each of the new slices
        mask_slice=np.zeros((w,h),dtype=np.uint8)
        newpts = np.zeros((n, 2), dtype=np.int32)
        for i in range(n): # interpolate each of n points
            y1 = p1[i,0]
            y2 = p2[i,0]
            x1 = p1[i,1]
            x2 = p2[i,1]
            sf1 = j/nNewSlices
            sf2 = (nNewSlices - j)/nNewSlices
            #print(j,i,sf1,sf2)
            newpts[i,1] = int ( y1*sf2 + y2*sf1 )
            newpts[i,0] = int ( x1*sf2 + x2*sf1 )
        # now put the contour in the label_img on the appropriate slice
        #print('newpts ',newpts)
        contours = [newpts]
        #print('contours ',contours)

        color = (255,255,255)
        mask_slice = cv2.drawContours(mask_slice, contours, -1, color, -1)
        #mask_slice =  cv2.cvtColor(mask_slice, cv2.COLOR_BGR2GRAY)
        mask_indices = np.nonzero(mask_slice)
        count = np.count_nonzero(mask_slice)
        #print('count for label for slice ',count,label,iStartSlice + j)
        this_slice = label_img[iStartSlice + j,:]
        this_slice[mask_indices] = label

    return label_img


# get N points around contour every 360/N degrees
# imask is now indices (x,y) of points at this label
def GetMaskContourPointsForInterpolation(imask,N,ilabel,islice,h,w,label_img):
    # m1 is mask
    cY = np.mean(imask[0])
    cX = np.mean(imask[1])
    # get the centroid
    #gray_image = cv2.cvtColor(m1, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray_image, 224, 255, 0)
    #M = cv2.moments(thresh)
    m1_centroid = (cY, cX)
    #print('centroid ',cY,cX)
    # for every 360/N degree, find point on contour (from centroid in direction of the theta - last point in region
    Pi = math.pi
    pts = np.zeros([N,2])
    for ipt in range(N):
        angle = (ipt/N) * 2 * Pi
        angle_in_degrees = angle * 180 / Pi
        #print('angle ',angle,angle_in_degrees)
        # line from centroid with angle
        # start at centroid, follow line for this angle until find a point not in mask
        step = .5
        for incr in range(5000):
            testy = int ( incr*step* math.cos(angle) + m1_centroid[0] )
            testx = int ( incr*step* math.sin(angle) + m1_centroid[1] )
            #print(testy,testx)
            #if  (m1[testx,testy,1] == 0):
            if (testy >= h) or (testx >= w) or (testy < 0) or (testx < 0):
                print('bad testy,testx ',testy,testx,incr,m1_centroid, angle_in_degrees)
                exit()
            if (label_img[islice,testy, testx] != ilabel):
                    #print('first point not in mask from centroid at this angle',m1_centroid,testx,testy)
                    pts[ipt,0] = testy
                    pts[ipt,1] = testx
                    break
    #print('incr ',incr,testy,testx)
    return pts
