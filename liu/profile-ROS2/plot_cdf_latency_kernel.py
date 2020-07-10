#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
"""
Plot data from WattsUp power meter

Format is assumed to be space sperated containing:
YYYY-MM-DD HH:MM:SS.ssssss n W V A
where n is sample number, W is power in Watts, V volts, A current in amps

Usage: plot.py log.out [plot.png]

Requires numpy and matplotlib

Author: Kelsey Jordahl
Copyright: Kelsey Jordahl 2011
License: GPLv3
Time-stamp: <Fri Sep  2 17:11:38 EDT 2011>

    This program is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.  A copy of the GPL
    version 3 license can be found in the file COPYING or at
    <http://www.gnu.org/licenses/>.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage.filters import gaussian_filter1d

def plot_line(data,color,line,label):

    # Choose how many bins you want here
    num_bins = 100

    data = np.multiply(data,1000)

    counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)
    cdf = cdf / cdf.max()
    cdf_smooth = gaussian_filter1d(cdf,sigma=3)
    plt.plot(bin_edges[1:], cdf_smooth,color=color,label=label,linestyle=line)


def main():
    # 1024 65536 524288
    data_1 = np.loadtxt('0_power_mode/generic_latency_power0.log')
    data_2 = np.loadtxt('1_power_mode/generic_latency_power1.log')
    data_3 = np.loadtxt('2_power_mode/generic_latency_power2.log')
    data_4 = np.loadtxt('3_power_mode/generic_latency_power3.log')
    data_5 = np.loadtxt('4_power_mode/generic_latency_power4.log')
    data_6 = np.loadtxt('5_power_mode/generic_latency_power5.log')
    data_7 = np.loadtxt('6_power_mode/generic_latency_power6.log')

    data_8 = np.loadtxt('0_power_mode/rt_latency_power0.log')
    data_9 = np.loadtxt('1_power_mode/rt_latency_power1.log')
    data_10 = np.loadtxt('2_power_mode/rt_latency_power2.log')
    data_11 = np.loadtxt('3_power_mode/rt_latency_power3.log')
    data_12= np.loadtxt('4_power_mode/rt_latency_power4.log')
    data_13 = np.loadtxt('5_power_mode/rt_latency_power5.log')
    data_14 = np.loadtxt('6_power_mode/rt_latency_power6.log')

    # data = [data_1,data_2,data_3,data_4,data_5,data_6,data_7]
    data = [data_8,data_9,data_10,data_11,data_12,data_13,data_14]
    # color = ["green","blue","red","black","orange"]
    color = ["green","blue","red","black"]
    line_style = ["-",":"]
    label = ["mode 0", "mode 1", "mode 2", "mode 3", "mode 4", "mode 5", "mode 6"]

    for i in range(len(data)):
        print(i)
        col = i % len(color)
        line = int(i / len(color))
        plot_line(data[i],color[col],line_style[line],label[i])


    rcParams.update({'figure.autolayout': True})
    # plt.tick_params(labelsize=8)
    # plt.xlim((0, 2))
    plt.legend(ncol=1, fontsize = 10,loc='lower right')
    plt.xlabel('End to end latency of YOLOv3 on ROS2 and RT Kernel (ms)',fontsize=8,fontweight='bold')
    plt.ylabel('CDF', fontsize=10, fontweight='bold')

    # plt.savefig('cdf-latency-1k.png', bbox_inches = "tight")
    plt.show()

if __name__ == '__main__':
    main()