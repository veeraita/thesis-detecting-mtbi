import os
import pandas as pd

from get_avg import get_cohorts

outfile = '../subject_demographics.csv'
tbi_dir = '/scratch/nbe/tbi-meg/veera/processed'
camcan_dir = '/scratch/nbe/restmeg/veera/processed'

cohorts = get_cohorts(tbi_dir, 'all', camcan_dir)

df = pd.DataFrame(cohorts.items(), columns=['cohort', 'subject']).explode('subject').sort_values('subject').set_index('subject')

df.to_csv(outfile, header=True)