#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

def plot_massbalance(C, evaporated, biodegraded, zc, tc, ax = None, title=None, savefig = False, filename='massbalance', xlabel=None, ylabel=None, return_ax=False, legend=False, xticks=None, skip_small=1e-8, dashed_lines=None, set_ticks=True):

    # Calculate derived quantities
    # Integrate concentration to get total dissolved amount
    dissolved = simps(C, x = zc, axis = 1)
    # Calculate direct transport as 1 - dissolved at t=0
    direct = 1 - dissolved[0]

    # Define some nice colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # Create figure and axis, unless suppled as kwarg
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (5, 3.5))

    # Create y-series needed to plot with fill_between
    y0 = np.zeros_like(tc)
    y1 = biodegraded
    y2 = biodegraded + dissolved
    y3 = biodegraded + dissolved + evaporated
    y4 = biodegraded + dissolved + evaporated + direct*np.ones_like(tc)

    # Plot
    if skip_small:
        if np.any(direct > skip_small):
            ax.fill_between(tc/(24*3600), y3, y4, color = colors[0], alpha = 0.5, label = 'Direct to surface')
        if np.any(evaporated > skip_small):
            ax.fill_between(tc/(24*3600), y2, y3, color = colors[0], label = 'Evaporated')
        if np.any(dissolved > skip_small):
            ax.fill_between(tc/(24*3600), y1, y2, color = colors[1], label = 'Dissolved')
        if np.any(biodegraded > skip_small):
            ax.fill_between(tc/(24*3600), y0, y1, color = colors[2], label = 'Biodegraded')
    else:
        ax.fill_between(tc/(24*3600), y3, y4, color = colors[0], alpha = 0.5, label = 'Direct to surface')
        ax.fill_between(tc/(24*3600), y2, y3, color = colors[0], label = 'Evaporated')
        ax.fill_between(tc/(24*3600), y1, y2, color = colors[1], label = 'Dissolved')
        ax.fill_between(tc/(24*3600), y0, y1, color = colors[2], label = 'Biodegraded')


    if set_ticks:
        if xticks is None:
            ax.set_xticks(np.arange(0, tc[-1]+1, 30*24*3600) / (24*3600))
        else:
            ax.set_xticks(xticks)

        if xlabel is None:
            ax.set_xlabel('Time [days]')
        else:
            ax.set_xlabel(xlabel)
        if ylabel is None:
            ax.set_ylabel('Mass fraction')
        else:
            ax.set_ylabel(ylabel)

    if dashed_lines is not None:
        for d in dashed_lines:
            plt.plot([d, d], [0, 1], '--', c='k', lw=1, alpha=0.8)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, tc[-1]/(24*3600) )
    casedict = {
        'case1' : 'Case 1',
        'case2' : 'Case 2',
        'case3' : 'Case 3',
    }

    if legend:
        ax.legend(loc=legend, fontsize=9)

    if savefig:
        if title is not None:
            ax.set_title(title)
        plt.subplots_adjust(left = 0.11, bottom = 0.12, right=0.97, top = 0.93)
        plt.savefig(filename + '.png', dpi = 240)
        plt.savefig(filename + '.pdf')

    if return_ax:
        return ax


def plot_parameter_scan(direct, dissolved, evaporated, biodegraded, xvalues, xlabel, xticks, filename=None, loc=None, skip_small=1e-6, title=None, dashed_lines=None):

    # Create figure and define some nice colors
    fig = plt.figure(figsize = (5, 3.5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Create y-series needed to plot with fill_between
    y0 = np.zeros_like(xvalues)
    y1 = biodegraded
    y2 = biodegraded + dissolved
    y3 = biodegraded + dissolved + evaporated
    y4 = biodegraded + dissolved + evaporated + direct*np.ones_like(xvalues)

    # Plot
    if skip_small:
        if np.any(direct > skip_small):
            plt.fill_between(xvalues, y3, y4, color = colors[0], alpha = 0.5, label = 'Direct to surface')
        if np.any(evaporated > skip_small):
            plt.fill_between(xvalues, y2, y3, color = colors[0], label = 'Evaporated')
        if np.any(dissolved > skip_small):
            plt.fill_between(xvalues, y1, y2, color = colors[1], label = 'Dissolved')
        if np.any(biodegraded > skip_small):
            plt.fill_between(xvalues, y0, y1, color = colors[2], label = 'Biodegraded')
    else:
        plt.fill_between(xvalues, y3, y4, color = colors[0], alpha = 0.5, label = 'Direct to surface')
        plt.fill_between(xvalues, y2, y3, color = colors[0], label = 'Evaporated')
        plt.fill_between(xvalues, y1, y2, color = colors[1], label = 'Dissolved')
        plt.fill_between(xvalues, y0, y1, color = colors[2], label = 'Biodegraded')

    if dashed_lines is not None:
        for d in dashed_lines:
            plt.plot([d, d], [0, 1], '--', c='k', lw=1, alpha=0.8)

    if loc is None:
        loc = 'best'
    plt.legend(loc=loc)

    plt.xscale('log')

    plt.xlabel(xlabel)
    plt.ylabel('Mass fraction')
    plt.ylim(0, 1)
    plt.xlim(xvalues[0], xvalues[-1])
    plt.xticks(xticks, xticks)
    if title is None:
        plt.subplots_adjust(left = 0.11, bottom = 0.125, right=0.97, top = 0.96)
    else:
        plt.title(title)
        plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
