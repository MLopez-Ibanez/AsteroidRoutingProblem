#!bin/bash

zipfile="zenodo.zip"
if [ -r $zipfile ]; then
    rm -f $zipfile
fi

cat << EOF | zip --exclude \*~ \*__pycache__ \*.pyc -r $zipfile -@
README.md
0-setup.sh
1-launch_exps.sh
2-collect_results.py
4-analysis.ipynb
5-visualize_sol.ipynb
6-statistics.R
arp.py
arp_vis.py
ast_orbits.pkl.gz
cego.py
greedy_nn.py
img
mallows_kendall.py
mallows_model.py
poliastro
problem.py
python-requirements.txt
results
r_problem.py
runner.py
space_util.py
target-runner-cego.py
target-runner-greedynn.py
target-runner-umm.py
timer.py
umm.py
EOF

