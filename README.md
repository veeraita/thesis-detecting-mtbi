## Detecting mild traumatic brain injury from MEG data using machine learning

This repository contains the code used for the Master's thesis by Veera Itälinna.

### Repository structure


*  `dicom2nifti`

    Contains a MATLAB script for converting DICOM files to NIfTI format.

*  `pipeline-tbi`

    Contains the steps of the source modelling pipeline (adapted from Rantala, 2020).

*  `model`

    Contains the code for training and evaluating the model and visualizing the results.
    
    
### The source modelling pipeline

1.  Run FreeSurfer on the MRI images (`pipeline-tbi/freesurfer`)
2.  Make BEM meshes with Watershed algorithm (`pipeline-tbi/watershed`)
3.  Perform ICA on the raw MEG measurements, and apply the ICA solutions also to the empty room recordings (`pipeline-tbi/ica`)
4.  Calculate noise covariance matrices from the empty room recordings (`pipeline-tbi/noisecov`)
5.  (Optional: Increase the dataset size using a sliding window approach on the MEG measurements (`pipeline-tbi/window`))
6.  Compute the coordinate transformations (`pipeline-tbi/trans`)
7.  Compute source PSDs and morph to an average brain (`pipeline-tbi/psd`)
 

### Creating the dataset used in the thesis

1.  Calculate mean and standard deviation matrices from the normative samples (`pipeline-tbi/averages`)
2.  Calculate Z-score maps (`pipeline-tbi/zmap`)
3.  Parcellate the Z-maps (`pipeline-tbi/parc`)
4.  Collect all parcellated Z-maps into a single dataset (`pipeline-tbi/zmap/create_zmap_dataset.py`)
    
    
### References

Rantala, A. (2020). Creating a normative database of resting-state brain activity from a large number of MEG recordings [Aalto University School of Science]. http://urn.fi/URN:NBN:fi:aalto-202101311732


