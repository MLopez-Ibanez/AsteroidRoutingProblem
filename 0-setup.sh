#!/bin/sh
Rscript -e 'install.packages(c("remotes", "quadprog"), repos="https://cloud.r-project.org/");remotes::install_version("CEGO", version = "2.4.2")'
python3 -m pip install -r python-requirements.txt
