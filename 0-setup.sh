#!/bin/sh
module load apps/binapps/anaconda3/2022.10
source activate arp_env
Rscript -e 'install.packages("remotes", repos="https://cloud.r-project.org/");remotes::install_version("CEGO", version = "2.4.2", upgrade="never")'
python3 -m pip install --isolated -r python-requirements.txt

