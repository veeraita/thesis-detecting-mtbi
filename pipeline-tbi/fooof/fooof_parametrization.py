# -*- coding: utf-8 -*-

from fooof import FOOOF, Bands
from fooof.analysis import get_band_peak_fm
import numpy as np
import mne
import os
import sys
import pandas as pd

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from visualize.visualize import visualize_fooof_fit


def fooof_inparcel(subject, psd, fmin, fmax, aperiodic_mode, labels, output_fname, force_calculate=False, visualize=True):

    fm = FOOOF(peak_width_limits=[.2, 8], min_peak_height=.05, max_n_peaks=6, aperiodic_mode=aperiodic_mode)
    #fm.set_debug_mode(True)
    freq_range = [fmin, fmax]

    bands = Bands({'delta': [1, 4],
                   'theta': [4, 8],
                   'alpha': [8, 12],
                   'beta': [15, 30]})
    
    freqs = psd.times
    
    fig_dir = os.path.join(os.path.dirname(output_fname), 'fig')
    os.makedirs(fig_dir, exist_ok=True)
    
    if not os.path.exists(output_fname + ".npy") or force_calculate:
        
        print("Calculating FOOOF for parcels for subject {}, background mode = {}".format(subject, aperiodic_mode))

        # Split the peak parameters by parcels (location in cortex)
        results = np.array([{} for _ in range(len(labels))])
        results_by_parcel = {}
        
        for i, label in enumerate(labels):
            
            spectrum = psd.in_label(label).data.mean(axis=0)
            fm.fit(freqs, spectrum, freq_range)

            for band in bands.labels:
                results[i][f'{band}_peak_params'] = get_band_peak_fm(fm, bands[band])

            results[i]['aperiodic_params'] = fm.aperiodic_params_
            results[i]['error'] = fm.error_
            results[i]['r_squared'] = fm.r_squared_
            
            if i % 100 == 0:
                print("Label: " + str(i))
                fig_fname = os.path.join(fig_dir, os.path.basename(output_fname) + f'-{label.name}.png')
                if visualize:
                    try:
                        visualize_fooof_fit(fm, fig_fname)
                    except Exception as e:
                        print(e)
        
            label_results = results[i] 
            results_by_parcel[label.name] = label_results
        #print(results_by_parcel)
        np.save(output_fname, results_by_parcel)
    
    else:
        print('Already exists: Subject ' + subject)
        results_by_parcel = np.load(output_fname + ".npy", allow_pickle=True).item()
    
    return results_by_parcel
