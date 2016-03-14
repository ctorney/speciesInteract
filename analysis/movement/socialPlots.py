import os
import csv
import numpy as np
from datetime import datetime

from numpy import array, empty
from numpy.random import randint, rand
import numpy as np

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
#import pandas as pd
import math
from math import *
from matplotlib import transforms
from viridis import viridis

#import ternary

plt.register_cmap(name='viridis', cmap=viridis)
viridis_r = matplotlib.colors.LinearSegmentedColormap( 'viridis_r', matplotlib.cm.revcmap(viridis._segmentdata))
plt.register_cmap(name='viridis_r', cmap=viridis_r)

plt.close('all')
dirs = ['zeb']
values = np.zeros((4,3))
plt.figure()
for idx,dd in enumerate(dirs): 
    
    aa = np.load(dd + '/alpha.npy')
    bb = np.load(dd + '/beta.npy')
    ev=(1-aa)*(1-bb)
    sv=bb
    mv=(aa)*(1-bb)
    values[idx,0]=(np.mean(ev))-0.6
    values[idx,1]=(np.mean(sv))
    values[idx,2]=(np.mean(mv))
    #plt.hist(mv,normed=True, label='memory')
    plt.hist(sv,normed=True, range=[0,0.3], bins=60,label=dd,alpha=0.5)
    #plt.hist(ev,normed=True, label='environment')
plt.title('strength of social interaction force')
plt.legend(loc='upper right')
plt.xlim(0,0.4)
plt.show()
plt.savefig('socialinteractions.png')


plt.figure()
for idx,dd in enumerate(dirs): 
    
    aa = np.load(dd + '/rho.npy')
    
    plt.hist(aa,normed=True,label=dd)#, range=[0.7,1], bins=1000,alpha=0.5)
        
    
plt.title('strength of bias')
plt.legend(loc='upper left')
#plt.xlim(0.8,1)
plt.show()
plt.savefig('bias.png')

#### Scatter Plot
#scale = 0.3
#fontsize=12
#figure, tax = ternary.figure(scale=scale)
#tax.set_title("Scatter Plot", fontsize=20)
#tax.boundary(linewidth=2.0)
#tax.gridlines(multiple=0.1, color="blue")
## Plot a few different styles with a legend
#
#tax.scatter([tuple(values[0,])], marker='s', color='red', label="wildebeest-wildebeest")
#tax.scatter([tuple(values[1,])], marker='D', color='black', label="zebra-zebra")
#tax.scatter([tuple(values[2,])], marker='o', color='blue', label="wildebeest-zebra")
#tax.scatter([tuple(values[3,])], marker='s', color='green', label="zebra-wildebeest")
#
##tax.scatter(values[0,], marker='D', color='green', label="Green Diamonds")
#tax.legend()
##tax.ticks(axis='lbr', linewidth=1, multiple=0.1)
#tax.left_axis_label("Left label $\\alpha^2$")
#tax.right_axis_label("Right label $\\beta^2$")
#tax.bottom_axis_label("Bottom label $\\Gamma - \\Omega$")
#tax.show()

#
### POLAR PLOT OF RELATIVE POSITIONS
##BL = is approx 32 pixels
binn2=19 # distance bins
binn1=72*8

dr = 0.5 # width of distance bins
sr = 0.25 # start point of distance
maxr=sr+(dr*binn2)

for idx,dd in enumerate(dirs): 
    
    aa = np.load(dd + '/ignore_length.npy')
    bb = np.load(dd + '/interaction_length.npy')
    cc = np.load(dd + '/interaction_angle.npy')
    ig=np.mean(aa)
    ir=np.mean(bb)
    vis_angle = np.mean(cc)
    theta2 = np.linspace(0,2.0 * np.pi, binn1+1)
##theta2 = np.linspace(-vis_angle,vis_angle, binn1+1)
    r2 = np.linspace(sr, maxr, binn2+1)
    areas = pi*((r2+dr)**2-r2**2)

    areas = (np.exp((1.0/ig)*(1-(r2/ir)**ig))*(r2/ir))#**ig
    areas=np.tile(areas,(binn1,1)).T
    
    for i in range(binn1):
        for j in range(binn2):
            if ((theta2[i])>vis_angle):
                if (theta2[i])<2*pi-vis_angle:
                    areas[j,i]=0;
    
    hista2 =areas
    
    size = 8
    # make a square figure
    
    fig1=plt.figure(figsize=(8,8))
    ax2=plt.subplot(projection="polar",frameon=False)
    im=ax2.pcolormesh(theta2,r2,hista2,lw=0.0,vmin=np.min(hista2),vmax=np.max(hista2),cmap='viridis')
    ax2.yaxis.set_visible(False)
    
    # angle lines
    ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)
    ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=ax2.get_theta_direction(), theta_offset=ax2.get_theta_offset())
    ax1.yaxis.set_visible(False)
    ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '', '',  '', 'back', '','', ''],frac=1.1)
    #colourbar
    position=fig1.add_axes([1.1,0.12,0.04,0.8])
    cbar=plt.colorbar(im,cax=position) 
    cbar.set_label('Social weighting', rotation=90,fontsize='xx-large',labelpad=15)      
    
    #body length legend - draws the ticks and 
    axes=ax2            
    factor = 0.98
    d = axes.get_yticks()[-1] #* factor
    r_tick_labels = [0] + axes.get_yticks()
    r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
    theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
    r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
    theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2
    
    # fixed offsets in x
    offset_spine = transforms.ScaledTranslation(-100, 0, axes.transScale)
    offset_ticklabels = transforms.ScaledTranslation(-10, 0, axes.transScale)
    offset_axlabel = transforms.ScaledTranslation(-40, 0, axes.transScale)
    
    # apply these to the data coordinates of the line/ticks
    trans_spine = axes.transData + offset_spine
    trans_ticklabels = trans_spine + offset_ticklabels
    trans_axlabel = trans_spine + offset_axlabel
    axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)
    
    # plot the 'tick labels'
    for ii in range(len(r_ticks)):
        axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)
    
    # plot the 'axis label'
    axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='xx-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')
    
    
    fig1.savefig(dd+"rules_w.png",bbox_inches='tight',dpi=100)
    # plot the 'spine'
    
    
