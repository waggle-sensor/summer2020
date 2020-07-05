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

    data_0 = np.loadtxt('latency_image_file.log')
    data_1 = np.loadtxt('latency_image_usb.log')

    data_0 = np.multiply(data_0,1000)
    print(data_0.max(),data_0.min(),data_0.mean())
    data_1 = np.multiply(data_1,1000)
    print(data_1.max(),data_1.min(),data_1.mean())

    # Choose how many bins you want here
    num_bins = 100

    # Use the histogram function to bin the data
    # counts_base, bin_edges_base = np.histogram(data_base, bins=num_bins, normed=True)
    # counts_base, bin_edges_base = np.histogram(data_base, bins=num_bins, normed=True)
    counts_0, bin_edges_0 = np.histogram(data_0, bins=num_bins, normed=True)
    counts_1, bin_edges_1 = np.histogram(data_1, bins=num_bins, normed=True)

    # cdf_base = np.cumsum(counts_base)
    cdf_0 = np.cumsum(counts_0)
    cdf_1 = np.cumsum(counts_1)

    # cdf_base = cdf_base / cdf_base.max()
    cdf_0 = cdf_0 / cdf_0.max()
    cdf_1 = cdf_1 / cdf_1.max()

    # cdf_base_smooth = gaussian_filter1d(cdf_base,sigma=3)
    cdf_0_smooth = gaussian_filter1d(cdf_0,sigma=3)
    cdf_1_smooth = gaussian_filter1d(cdf_1,sigma=3)

    # plt.plot(bin_edges_base[1:], cdf_base_smooth,label="Baseline", color="black", linestyle="-",linewidth=3)
    # plt.plot(bin_edges_0[1:], cdf_0,color="red", label="mode 0",linestyle="-")
    # plt.plot(bin_edges_1[1:], cdf_1,color="green", label="mode 1",linestyle="-")
    # plt.plot(bin_edges_2[1:], cdf_2,color="blue",label="mode 2",linestyle="-")
    # plt.plot(bin_edges_3[1:], cdf_3,color="gray",label="mode 3",linestyle="-")
    # plt.plot(bin_edges_4[1:], cdf_4,color="black",label="mode 4",linestyle="-")
    # plt.plot(bin_edges_5[1:], cdf_5,color="orange",label="mode 5",linestyle="-")
    # plt.plot(bin_edges_6[1:], cdf_6,color="purple",label="mode 6",linestyle="-")

    # plt.plot(bin_edges_base[1:], cdf_base_smooth,label="Baseline", color="black", linestyle="-",linewidth=3)
    plt.plot(bin_edges_0[1:], cdf_0_smooth,color="red", label="image file",linestyle="-")
    plt.plot(bin_edges_1[1:], cdf_1_smooth,color="green", label="USB camera",linestyle="-")

    rcParams.update({'figure.autolayout': True})
    plt.tick_params(labelsize=12)
    plt.legend(ncol=1, fontsize = 10)
    plt.xlabel('ms',fontsize=12,fontweight='bold')
    plt.ylabel('CDF', fontsize=12, fontweight='bold')

    plt.savefig('Xavier-USB-latency.png', bbox_inches = "tight")
    plt.show()

if __name__ == '__main__':
    main()