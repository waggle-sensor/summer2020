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

def main():

    data_0 = np.loadtxt('0_power_mode/generic_latency_power0.log')
    data_1 = np.loadtxt('0_power_mode/rt_latency_power0.log')
    data_2 = np.loadtxt('../profile-ROS1/0_power_mode/latency.log')
    data_3 = np.loadtxt('../profile-ROS1/0_power_mode/latency-rt.log')
    data_4 = np.loadtxt('4_power_mode/generic_latency_power4.log')
    data_5 = np.loadtxt('5_power_mode/generic_latency_power5.log')
    data_6 = np.loadtxt('6_power_mode/generic_latency_power6.log')

    # data_0 = np.loadtxt('0_power_mode/generic_latency_power0.log')
    # data_1 = np.loadtxt('1_power_mode/generic_latency_power1.log')
    # data_2 = np.loadtxt('2_power_mode/generic_latency_power2.log')
    # data_3 = np.loadtxt('3_power_mode/generic_latency_power3.log')
    # data_4 = np.loadtxt('4_power_mode/generic_latency_power4.log')
    # data_5 = np.loadtxt('5_power_mode/generic_latency_power5.log')
    # data_6 = np.loadtxt('6_power_mode/generic_latency_power6.log')

    data_0 = np.multiply(data_0,1000)
    print(data_0.max(),data_0.min(),data_0.mean())
    data_1 = np.multiply(data_1,1000)
    print(data_1.max(),data_1.min(),data_1.mean())
    data_2 = np.multiply(data_2,1000)
    print(data_2.max(),data_2.min(),data_2.mean())
    data_3 = np.multiply(data_3,1000)
    print(data_3.max(),data_3.min(),data_3.mean())
    data_4 = np.multiply(data_4,1000)
    print(data_4.max(),data_4.min(),data_4.mean())
    data_5 = np.multiply(data_5,1000)
    print(data_5.max(),data_5.min(),data_5.mean())
    data_6 = np.multiply(data_6,1000)
    print(data_6.max(),data_6.min(),data_6.mean())

    # Choose how many bins you want here
    num_bins = 100

    # Use the histogram function to bin the data
    # counts_base, bin_edges_base = np.histogram(data_base, bins=num_bins, normed=True)
    # counts_base, bin_edges_base = np.histogram(data_base, bins=num_bins, normed=True)
    counts_0, bin_edges_0 = np.histogram(data_0, bins=num_bins, normed=True)
    counts_1, bin_edges_1 = np.histogram(data_1, bins=num_bins, normed=True)
    counts_2, bin_edges_2 = np.histogram(data_2, bins=num_bins, normed=True)
    counts_3, bin_edges_3 = np.histogram(data_3, bins=num_bins, normed=True)
    counts_4, bin_edges_4 = np.histogram(data_4, bins=num_bins, normed=True)
    counts_5, bin_edges_5 = np.histogram(data_5, bins=num_bins, normed=True)
    counts_6, bin_edges_6 = np.histogram(data_6, bins=num_bins, normed=True)

    # cdf_base = np.cumsum(counts_base)
    cdf_0 = np.cumsum(counts_0)
    cdf_1 = np.cumsum(counts_1)
    cdf_2 = np.cumsum(counts_2)
    cdf_3 = np.cumsum(counts_3)
    cdf_4 = np.cumsum(counts_4)
    cdf_5 = np.cumsum(counts_5)
    cdf_6 = np.cumsum(counts_6)

    # cdf_base = cdf_base / cdf_base.max()
    cdf_0 = cdf_0 / cdf_0.max()
    cdf_1 = cdf_1 / cdf_1.max()
    cdf_2 = cdf_2 / cdf_2.max()
    cdf_3 = cdf_3 / cdf_3.max()
    cdf_4 = cdf_4 / cdf_4.max()
    cdf_5 = cdf_5 / cdf_5.max()
    cdf_6 = cdf_6 / cdf_6.max()

    # cdf_base_smooth = gaussian_filter1d(cdf_base,sigma=3)
    cdf_0_smooth = gaussian_filter1d(cdf_0,sigma=3)
    cdf_1_smooth = gaussian_filter1d(cdf_1,sigma=3)
    cdf_2_smooth = gaussian_filter1d(cdf_2,sigma=3)
    cdf_3_smooth = gaussian_filter1d(cdf_3,sigma=3)
    cdf_4_smooth = gaussian_filter1d(cdf_4,sigma=3)
    cdf_5_smooth = gaussian_filter1d(cdf_5,sigma=3)
    cdf_6_smooth = gaussian_filter1d(cdf_6,sigma=3)

    # plt.plot(bin_edges_base[1:], cdf_base_smooth,label="Baseline", color="black", linestyle="-",linewidth=3)
    # plt.plot(bin_edges_0[1:], cdf_0,color="red", label="mode 0",linestyle="-")
    # plt.plot(bin_edges_1[1:], cdf_1,color="green", label="mode 1",linestyle="-")
    # plt.plot(bin_edges_2[1:], cdf_2,color="blue",label="mode 2",linestyle="-")
    # plt.plot(bin_edges_3[1:], cdf_3,color="gray",label="mode 3",linestyle="-")
    # plt.plot(bin_edges_4[1:], cdf_4,color="black",label="mode 4",linestyle="-")
    # plt.plot(bin_edges_5[1:], cdf_5,color="orange",label="mode 5",linestyle="-")
    # plt.plot(bin_edges_6[1:], cdf_6,color="purple",label="mode 6",linestyle="-")

    # plt.plot(bin_edges_base[1:], cdf_base_smooth,label="Baseline", color="black", linestyle="-",linewidth=3)
    plt.plot(bin_edges_0[1:], cdf_0_smooth,color="red", label="ROS2 generic kernel",linestyle="-")
    plt.plot(bin_edges_1[1:], cdf_1_smooth,color="red", label="ROS2 RT kernel",linestyle=":")
    plt.plot(bin_edges_2[1:], cdf_2_smooth,color="blue",label="ROS1 generic kernel",linestyle="-")
    plt.plot(bin_edges_3[1:], cdf_3_smooth,color="blue",label="ROS1 RT kernel",linestyle=":")
    # plt.plot(bin_edges_4[1:], cdf_4_smooth,color="black",label="mode 4",linestyle="-")
    # plt.plot(bin_edges_5[1:], cdf_5_smooth,color="orange",label="mode 5",linestyle="-")
    # plt.plot(bin_edges_6[1:], cdf_6_smooth,color="purple",label="mode 6",linestyle="-")


    rcParams.update({'figure.autolayout': True})
    plt.tick_params(labelsize=12)
    plt.legend(ncol=1, fontsize = 10)
    plt.xlabel('ms',fontsize=12,fontweight='bold')
    plt.xlabel('End to end latency of YOLOv3 in power mode 0 (ms)',fontsize=8,fontweight='bold')
    plt.ylabel('CDF', fontsize=12, fontweight='bold')

    # plt.savefig('Xavier-Yolov3-latency.png', bbox_inches = "tight")
    plt.show()

if __name__ == '__main__':
    main()