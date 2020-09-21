import mne
import sys
import os
import os.path as op


def main(args):
    fname = args[1]
    outdir = args[2]
    if 'camcan' in fname:
        subj = 'sub-' + op.dirname(fname).split('/')[-1]
    else:
        subj = op.basename(fname).split('_')[0]
    raw = mne.io.Raw(fname)
    cov = mne.compute_raw_covariance(raw)
    outfile = op.join(outdir, subj, 'cov', subj + '-cov.fif')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    print("Saving to", outfile)
    cov.save(outfile)


if __name__ == "__main__":
    main(sys.argv)
