#!/bin/sh
Rscript -e 'devtools::install_version("CEGO", version = "2.4.2")'
python3 -m pip install -r python-requirements.txt