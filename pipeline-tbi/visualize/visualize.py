import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_bem(fig_fname, subjects_dir, subject, src=None):
    print('Visualizing BEM surfaces...')
    plt.figure()
    mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                     brain_surfaces='white', orientation='coronal', src=src)
    print('Saving figure...')
    plt.savefig(fig_fname)
    print('Figure saved in', fig_fname)
    plt.close()


def visualize_coregistration(raw_fname, trans_fname, fiducials_fname, fig_fname, subjects_dir, subject):
    print('Visualizing coregistration...')
    info = mne.io.read_info(raw_fname)
    bem = mne.read_bem_surfaces('/outputs/recon/sub-CC110033/bem/sub-CC110033-head.fif')
    plt.figure()
    mne.viz.plot_alignment(info=info, trans=trans_fname, subject=subject, dig=True,
                           meg=['helmet', 'sensors'], subjects_dir=subjects_dir,
                           surfaces='brain', mri_fiducials=fiducials_fname, bem=bem,
                           verbose=True)
    print('Saving figure...')
    plt.savefig(fig_fname)
    print('Figure saved in', fig_fname)
    plt.close()


def visualize_source_space(src_fname, fig_fname, subjects_dir, subject):
    print('Visualizing source space...')
    src = mne.read_source_spaces(src_fname)
    plt.figure()
    mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                     brain_surfaces='white', src=src, orientation='coronal', show=False)
    print('Saving figure...')
    plt.savefig(fig_fname)
    print('Figure saved in', fig_fname)
    plt.close()


def visualize_stc(stc_fname, fig_fname, subjects_dir, subject, initial_time=10., colormap='auto', clim='auto', spacing='ico4', fsaverage=True):
    print('Visualizing source estimate...')
    stc_subject = 'fsaverage' if fsaverage else subject
    stc = mne.read_source_estimate(stc_fname, subject=stc_subject)
    plt.figure()
    stc.plot(subjects_dir=subjects_dir, backend='matplotlib', initial_time=initial_time, hemi='lh', colormap=colormap, clim=clim, spacing=spacing)
    lh_fig_fname = fig_fname.replace('.png', '-lh.png')
    plt.savefig(lh_fig_fname)
    print('Saving figure (left hemisphere)...')
    plt.savefig(lh_fig_fname)
    print('Figure saved in', lh_fig_fname)

    stc.plot(subjects_dir=subjects_dir, backend='matplotlib', initial_time=initial_time, hemi='rh', colormap=colormap, clim=clim, spacing=spacing)
    print('Saving figure (right hemisphere)...')
    rh_fig_fname = fig_fname.replace('.png', '-rh.png')
    plt.savefig(rh_fig_fname)
    print('Figure saved in', rh_fig_fname)
    plt.close()


def visualize_psd(stc_fname, fig_fname, subject, stc_avg_fname=None, stc_var_fname=None, fsaverage=True):
    print('Visualizing PSD...')
    stc_subject = 'fsaverage' if fsaverage else subject
    stc = mne.read_source_estimate(stc_fname, subject=stc_subject)
    plt.figure()
    plt.plot(stc.times, 10 * np.log10(stc.data.mean(axis=0)), label=subject)
    if stc_avg_fname:
        stc_avg = mne.read_source_estimate(stc_avg_fname, subject=stc_subject)
        x = stc_avg.times
        y = 10 * np.log10(stc_avg.data.mean(axis=0))
        plt.plot(x, y, label='normative avg')
        if stc_var_fname:
            stc_var = mne.read_source_estimate(stc_var_fname, subject=stc_subject)
            dy = np.sqrt(10 * np.log10(stc_var.data.mean(axis=0)))
            plt.fill_between(x, y - dy, y + dy, color='gray', alpha=0.2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title('Source Power Spectrum (PSD)')
    plt.legend()
    print('Saving figure...')
    plt.savefig(fig_fname)
    print('Figure saved in', fig_fname)
    plt.close()


def visualize_fooof_fit(fooof_model, fig_fname):
    fooof_model.plot(plot_peaks='shade', add_legend=True, save_fig=True, file_path=os.path.dirname(fig_fname),
                     file_name=os.path.basename(fig_fname))
    plt.close()
