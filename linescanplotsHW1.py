
# coding: utf-8

# # EECS192 Spring 2018 Track Finding from 1D line sensor data

# In[6]:

# changed to use 8 bit compressed line sensor values
# data format: 128 comma separated values, last value in line has space, not comma
# line samples are about 10 ms apart
#  csv file format time in ms, 128 byte array, velocity
# note AGC has already been applied to data, and camera has been calibrated for illumination effects


# In[8]:

import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
# import scipy.ndimage as ndi  # useful for 1d filtering functions
plt.close("all")   # try to close all open figs

# In[9]:

# Graphing helper function
def setup_graph(title='', x_label='', y_label='', fig_size=None):
    fig = plt.figure()
    if fig_size != None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# Line scan plotting function.
# 

# In[10]:

def plot_frame(linearray):
    nframes = np.size(linearray)/128
    n = range(0,128)
    print('number of frames', nframes)
    print('size of line', np.size(linearray[0,:]))
    for i in range(0, nframes-1):
        setup_graph(title='$x[n]$', x_label='$n$', y_label='row'+str(i)+' $ xa[n]$', fig_size=(15,2))
        plt.subplot(1,3,1)
        _ = plt.plot(n,linearray[0,:])
        plt.subplot(1,3,2)
        _ = plt.plot(n,linearray[i,:])
    # plot simple difference between frame i and first frame
        plt.subplot(1,3,3)
        _ = plt.plot(n,linearray[i,:] - linearray[0,:])
        plt.ylabel('Frame n - Frame 0')


# ### grayscale plotting of line function:
# 

# In[11]:

CAMERA_LENGTH = 128
INTENSITY_MIN = 0
INTENSITY_MAX = 255
def plot_gray(fig, camera_data):
  # x fencepost positions of each data matrix element
  x_mesh = []
  for i in range(0, len(camera_data)+1):
    x_mesh.append([i-0.5] * (CAMERA_LENGTH + 1))
  x_mesh = np.array(x_mesh)
  
  # y fencepost positions of each data matrix element
  y_array = range(0, CAMERA_LENGTH + 1)
  y_array = list(map(lambda x: x - 0.5, y_array))
  y_mesh = np.array([y_array] * (len(camera_data)+1))
    
  data_mesh = np.array(camera_data)
  vmax1 = np.max(data_mesh)
  data_mesh = INTENSITY_MAX * data_mesh/vmax1  # normalize intensity
  
  fig.set_xlim([-0.5, len(camera_data) - 0.5])
  fig.set_ylim([-8.5, CAMERA_LENGTH - 0.5])

  fig.pcolorfast(x_mesh, y_mesh, data_mesh,
      cmap='gray', vmin=INTENSITY_MIN, vmax=INTENSITY_MAX,
      interpolation='None')


# In[12]:

### inputs:
# linescans - An array of length n where each element is an array of length 128. Represents n frames of linescan data.

### outputs:
# track_center_list - A length n array of integers from 0 to 127. Represents the predicted center of the line in each frame.
# track_found_list - A length n array of booleans. Represents whether or not each frame contains a detected line.
# cross_found_list - A length n array of booleans. Represents whether or not each frame contains a crossing.

def find_track(linescans):
    n = len(linescans)
    print(np.size(linescans))
    print(n)
    track_center_list = n * [64]  # set to center pixel in line for test case
    track_left_list = n * [64]
    track_right_list = n * [64]
    track_found_list = n * [100]  # 100 for true
    max_matrix = n*[0]
    # track_found_list[200:300] = 10* np.ones(100) # 10 for false
    linemean = n * [0]
    spacemean = n * [0]
    linelength = n * [0]
    cross_found_list = n * [10]# 10 for false
    #cross_found_list[1000:1100] = 100* np.ones(100) # throw in a few random cross founds
    #  track_center_list[0] = np.argmax(linescans[0])
    # place holder function this is not robust
    argmax_matrix = n*[0]
    for i in range(0,n):
        argmax_matrix[i]=np.argmax(linescans[i])
    for i in range(0,n):
          max_matrix[i] = max(linescans[i])
    # gradient = 130*[0]
    max_gradient=n*[0]

    for i in range(0, n):
        gradienttemp = 130*[0]
        for j in range(0, 127):
            gradienttemp[j]=linescans[i][j+1]-linescans[i][j]
        if i==1000:
            f=1
        track_left_list[i] = np.argmax(gradienttemp)
        max_gradient[i]=max(gradienttemp)
        track_right_list[i] = np.argmin(gradienttemp)
    for i in range(0, n):
        # a = linemean[i]
        # b = spacemean[i]
        linelength[i] = track_right_list[i]-track_left_list[i]
        linemean[i] = np.mean(linescans[i][track_left_list[i]+1:track_right_list[i]+1])
        spacearray = np.concatenate((linescans[i][0:track_left_list[i]+1],linescans[i][track_right_list[i]+1:128]),axis=None)
        if i==1000:
            c=2
        spacemean[i] = np.mean(spacearray)
    for i in range(0, n):
        if linemean[i]-spacemean[i]>10 or i==0:
            track_center_list[i] = int(round((track_left_list[i] + track_right_list[i])/2))+1
        else:
            if max_gradient[i]>10:    ## Solving multi-white-object problem
                gradienttemp = 130 * [0]
                for j in range(0, 127):
                    gradienttemp[j] = linescans[i][j + 1] - linescans[i][j]
                sort_gradient=np.argpartition(gradienttemp, -5)[-5:]
                temp1 = abs(track_center_list[i-1]-sort_gradient)
                postemp= temp1.tolist().index(min(temp1))
                track_center_list[i]=sort_gradient[postemp]+1
                cross_found_list[i]=10


                #gradient>5 gradient<5
            # all gradient, calculate center moving difference, pick least difference one
            else:
                track_center_list[i] = track_center_list[i-1]
                track_found_list[i]=10





        # if a > b:
        #     track_found_list[i] = 100
        # else:
        #     track_found_list[i] = 10
    ## Goal:
    # track not found conditions:
    #
    # track found: when track is not found, use former information to compute
    #              track is not found only when it is all dark
    ## if left and right edge stands far away, choose the one near the former center.

    ## the average pixel value among the length of line is larger than others' mean
    ## Note: track found list should be existing line which length is limited to a certain number. And the pixel
    ##  is much bigger than others.
    ## Note: cross found list should be detecting length of current line, if the length is much larger than
    ##  the average of former lines.(20 pixels larger)
    ## Cross: length of line is larger than 3

    ### Code to be added here
    prev = 0
    preprev = 0
    count = 0
    # for i in range(0, n):
    #     linescans[i] = [100 if a_ > 29 else a_ for a_ in linescans[i]]
    for i in range(0, round(n/3)):
        linescans[i] = [100 if a_ > 29 else a_ for a_ in linescans[i]]
    for i in range(round(n/3), round(n * 3/7) ):
        linescans[i] = [100 if a_ > 33 else a_ for a_ in linescans[i]]
    for i in range(round(n*3 / 7), round(n * 3 / 6)):
        linescans[i] = [100 if a_ > 35 else a_ for a_ in linescans[i]]
    for i in range(round(n*3 /6) , round(n*7/8)):
        linescans[i] = [100 if a_ > 32 else a_ for a_ in linescans[i]]
    for i in range(round(n * 7 / 8), round(n)):
        linescans[i] = [100 if a_ > 30 else a_ for a_ in linescans[i]]
    for i in range(0,n-1):
        if (track_found_list[i]==100):
            cross_countup = 0;
            cross_countdown = 0;
            center=track_center_list[i]
            center_val=linescans[i][track_center_list[i]]

            for j in range(80):
                if(j+center>127):
                    break
                judge_value1=linescans[i][j+center]
                #judge_value2=abs(center_val-judge_value1)
                c2 = center_val * 4 / 8
                # if(0<i<n-2):
                #     judge_value2 = linescans[i+1][j+center]
                #     judge_value3 = linescans[i + 1][j + center]
                #     if (judge_value2 > c2 or judge_value3 > c2):
                #         continue
                #
                #



                if (judge_value1>c2):
                    cross_countdown+=1
                else:
                    break;
            for j in range(80):
                if (center-j)<=0:
                    break
                judge_value1 = linescans[i][center-j]
                #judge_value2 = abs(center_val-judge_value1)
                c2 = center_val * 4/8
                if (judge_value1>c2):
                    cross_countup+=1
                else:
                    break
            k=i-1
            for j in range(80):
                if(j+center>127):
                    break
                judge_value1=linescans[k][j+center]
                #judge_value2=abs(center_val-judge_value1)
                c2 = center_val * 4 / 8

                if (judge_value1>c2):
                    cross_countdown+=1
                else:
                    break;
            for j in range(80):
                if (center-j)<=0:
                    break
                judge_value1 = linescans[k][center-j]
                #judge_value2 = abs(center_val-judge_value1)
                c2 = center_val * 4/8
                if (judge_value1>c2):
                    cross_countup+=1
                else:
                    break
            k=i+1
            for j in range(80):
                if(j+center>127):
                    break
                judge_value1=linescans[k][j+center]
                #judge_value2=abs(center_val-judge_value1)
                c2 = center_val*4/8
                if (judge_value1>c2):
                    cross_countdown+=1
                else:
                    break;
            for j in range(80):
                if (center-j)<=0:
                    break
                judge_value1 = linescans[k][center-j]
                #judge_value2 = abs(center_val-judge_value1)
                c2 = center_val * 4/8
                if (judge_value1>c2):
                    cross_countup+=1
                else:
                    break

            #if(cross_countdown>70 or cross_countup>70 or(cross_countdown>40 and cross_countup>40)):
            cross = cross_countdown + cross_countup
            if(cross) >  30 :
                dex = i-0
                cross_found_list[(dex)]=100
                count +=1
            # else:
            #     if count > 4:
            #         for x in range(1,count+1):
            #             d = i - x
            #             cross_found_list[d] = 10
            #     count = 0
            #
            for i in range(0, n):
                if linemean[i] - spacemean[i] > 10 or i == 0:
                   e=1
                else:
                    if max_gradient[i] > 10:  ## Solving multi-white-object problem
                        cross_found_list[i] = 10
                    else:
                        for m in range(0, 10):
                            cross_found_list[i + m] = 10


            #preprev = prev
            #prev = cross
        # if i>5:
        #     formerlengtharr=np.array([linelength[i-1],linelength[i-2],linelength[i-3],linelength[i-4],linelength[i-5],linelength[i-6]])
        #     formerlength = np.mean(formerlengtharr)
        #     if (linelength[i]-formerlength)>50 and linemean[i]>spacemean[i]:
        #         cross_found_list[i] = 100
        #     else:
        #         cross_found_list[i] = 10
        # else:
        #     cross_found_list[i] = 10

    return track_center_list, track_found_list, cross_found_list, track_left_list, track_right_list




################
# need to use some different tricks to read csv file
import csv


def read_file():
  filename = './natcar2016_team1.csv'
  #filename = 'natcar2016_team1_short.csv'
  csvfile=open(filename, 'r')
  telemreader=csv.reader(csvfile, delimiter=',', quotechar='"')
  # Actual Spring 2016 Natcar track recording by Team 1.
  telemreader.__next__()  # discard first line
 # telemdata = telemreader.__next__()  # format time in ms, 128 byte array, velocity
  # linescans=[]  # linescan array
  #  times=[]  # should be global
  #  velocities=[]
  for row in telemreader:
      times.append(eval(row[0]))  # sample time
      velocities.append(eval(row[2]))  # measured velocity
      line = row[1]  # get scan data
      arrayline = np.array(eval(line))  # convert line to an np array
      linescans.append(arrayline)
  #print('scan line0:', linescans[0])
 # print('scan line1:', linescans[1])
  print(linescans)
  print(velocities)
  return np.array(linescans)

linescans=[]
times=[]
velocities=[]
filename = './natcar2016_team1.csv'
linescans = read_file()
n=len(linescans)

track_center_list, track_found_list, cross_found_list, left_gradient, right_gradient = find_track(linescans)
#for i, (track_center, track_found, cross_found) in enumerate(zip(track_center_list, track_found_list, cross_found_list)):
 #   print('scan # %d center at %d. Track_found = %s, Cross_found = %s' %(i,track_center,track_found, cross_found))

#
# ############# plots ###########
# #fig=plt.figure()
# fig = plt.figure(figsize = (16, 3))
# fig.set_size_inches(13, 4)
# fig.suptitle("%s\n" % (filename))
# ax = plt.subplot(1, 1, 1)
# #plot_gray(ax, linescans[0:1000])  # plot smaller range if hard too see
# plot_gray(ax, linescans)
# plt.show()
#
# ############# plot of velocities
# #fig = plt.figure(figsize = (8, 4))
# #fig.set_size_inches(13, 4)
# #fig.suptitle("velocities %s\n" % (filename))
# #plt.xlabel('time [ms]')
# #plt.ylabel('velocity (m/s)')
# #plt.plot(times,velocities)
#
# ###############plot of found track position
#
# #fig = plt.figure(figsize = (8, 4))
# #fig.set_size_inches(13, 4)
# #fig.suptitle("track center %s\n" % (filename))
# #plt.xlabel('time [ms]')
# #plt.ylabel('track center')
# #plt.plot(times,track_center_list)
#
# ######################################
# #############################  plot superimposed on grayscale
# fig = plt.figure(figsize=(35,15))
# fig.clf()
# fig.suptitle("HW1 Spring 2019")
# idx0 = 0
# idx1 = 2500
# ax = fig.add_subplot(411)
# ax.imshow(linescans[idx0:idx1].T, cmap='gray')
# ax.get_yaxis().set_visible(False)
# # ax.set_xlabel('Time')
# ax.set_title('Track Section')
# x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
#
# ax = fig.add_subplot(412)
# ax.imshow(linescans[idx0:idx1].T, cmap='gray')
# ax.get_yaxis().set_visible(False)
# ax.plot(track_center_list[idx0:idx1], 'r-')
# # ax.set_xlabel('Time')
# ax.set_title('Track Center')
# ax.set_xlim(x_lim); ax.set_ylim(y_lim)
#
#
# ax = fig.add_subplot(413)
# ax.plot(track_found_list[idx0:idx1])
# ax.imshow(linescans[idx0:idx1].T, cmap='gray')
# ax.plot(track_found_list[idx0:idx1],'r-')
# ax.get_yaxis().set_visible(False)
# # ax.set_xlabel('Time')
# ax.set_title('Track Found')
# ax.set_xlim(x_lim); ax.set_ylim(y_lim)
#
# ax = fig.add_subplot(414)
# ax.imshow(linescans[idx0:idx1].T, cmap='gray')
# ax.get_yaxis().set_visible(False)
# ax.plot(cross_found_list[idx0:idx1],'r-')
# ax.set_xlabel('Time')
# ax.set_title('Cross Found')
# ax.set_xlim(x_lim); ax.set_ylim(y_lim)
# fig.show()
#
# # ax = fig.add_subplot(413)
# # ax.imshow(linescans[idx0:idx1].T, cmap='gray')
# # ax.get_yaxis().set_visible(False)
# # ax.plot(left_gradient[idx0:idx1],'r-')
# # ax.set_xlabel('Time')
# # ax.set_title('Left Gradient')
# # ax.set_xlim(x_lim); ax.set_ylim(y_lim)
# # fig.show()
# #
# # ax = fig.add_subplot(414)
# # ax.imshow(linescans[idx0:idx1].T, cmap='gray')
# # ax.get_yaxis().set_visible(False)
# # ax.plot(right_gradient[idx0:idx1],'r-')
# # ax.set_xlabel('Time')
# # ax.set_title('Right Gradient')
# # ax.set_xlim(x_lim); ax.set_ylim(y_lim)
# # fig.show()