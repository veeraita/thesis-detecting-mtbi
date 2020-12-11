import mne
import sys
import os
import os.path as op


def main(args):
    subj = args[1]
    outdir = args[2]

    data_dir = os.path.join(outdir, subj, 'ica')
    fname = os.path.join(data_dir, f'{subj}-emptyroom-ica-recon.fif')

    raw = mne.io.Raw(fname)
    cov = mne.compute_raw_covariance(raw)
    outfile = op.join(outdir, subj, 'cov', subj + '-cov.fif')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    print("Saving to", outfile)
    cov.save(outfile)


if __name__ == "__main__":
    main(sys.argv)
