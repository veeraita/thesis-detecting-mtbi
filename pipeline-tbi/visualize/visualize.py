import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize_bem(fig_fname, subjects_dir, subject):
    print('Visualizing BEM surfaces...')
    plt.figure()
    mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                     brain_surfaces='white', orientation='coronal')
    print('Saving figure...')
    plt.savefig(fig_fname)
    print('Figure saved in', fig_fname)


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


def visualize_source_space(src_fname, fig_fname, subjects_dir, subject):
    print('Visualizing source space...')
    src = mne.read_source_spaces(src_fname)
    plt.figure()
    mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                     brain_surfaces='white', src=src, orientation='coronal', show=False)
    print('Saving figure...')
    plt.savefig(fig_fname)
    print('Figure saved in', fig_fname)


def visualize_stc(stc_fname, fig_fname, subjects_dir, subject, initial_time=10., colormap='auto', clim='auto'):
    print('Visualizing source estimate...')
    stc = mne.read_source_estimate(stc_fname, subject=subject)
    plt.figure()
    stc.plot(subjects_dir=subjects_dir, backend='matplotlib', initial_time=initial_time, hemi='lh', colormap=colormap, clim=clim)
    lh_fig_fname = fig_fname.replace('.png', '-lh.png')
    plt.savefig(lh_fig_fname)
    print('Saving figure (left hemisphere)...')
    plt.savefig(lh_fig_fname)
    print('Figure saved in', lh_fig_fname)

    stc.plot(subjects_dir=subjects_dir, backend='matplotlib', initial_time=initial_time, hemi='rh', clim=clim)
    print('Saving figure (right hemisphere)...')
    rh_fig_fname = fig_fname.replace('.png', '-rh.png')
    plt.savefig(rh_fig_fname)
    print('Figure saved in', rh_fig_fname)


def visualize_psd(stc_fname, fig_fname, subject):
    print('Visualizing PSD...')
    stc = mne.read_source_estimate(stc_fname, subject=subject)
    plt.figure()
    plt.plot(stc.times, stc.data.mean(axis=0))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB)')
    plt.title('Source Power Spectrum (PSD)')
    print('Saving figure...')
    plt.savefig(fig_fname)
    print('Figure saved in', fig_fname)
