The Asteroid Routing Problem: A Benchmark for Expensive Black-Box Permutation Optimization
============================

> Manuel López-Ibáñez, Francisco Chicano, Rodrigo Gil-Merino. **[The Asteroid Routing Problem: A Benchmark for Expensive Black-Box Permutation Optimization](https://arxiv.org/abs/2203.15708)**. In _Applications of Evolutionary Computation 2022_, LNCS 13224, Springer Nature, Switzerland, 2022. [doi:10.1007/978-3-031-02462-7_9](https://doi.org/10.1007/978-3-031-02462-7_9)  | [arXiv:2203.15708](https://arxiv.org/abs/2203.15708)

### Steps to reproduce ###

 * `0-setup.sh` : Install all required R and Python packages. Needs python 3 and R

 * `1-launch_exps.sh`: Launch experiments (this takes significant time). You may need to edit the contents of the file to match your computer setup.

 * `2-collect_results.py`: Run once after all the experiments have completed.

 * `3-analysis.ipynb`: Main analysis

 * `4-visualize_sol.ipynb`: Visualization of solutions.

 * `5-statistics.R`: Statistical tests.

### Examples of solutions ###

* Greedy Solution:
  
  ![Greedy Solution](/img/sol_greedy_10_73.svg)

 * CEGO+Greedy Solution:
 
  ![CEGO+Greedy Solution](/img/sol_cego-greedy-er1_10_73.svg)

 * UMM+Greedy Solution:

  ![UMM+Greedy Solution](/img/sol_umm-greedy-er0_10_73.svg)

### License ###


[![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](./LICENSE)

This software is Copyright (C) 2022 Manuel López-Ibáñez, Francisco Chicano and Rodrigo Gil-Merino under the MIT license. Please refer to the [LICENSE](./LICENSE) file.

CEGO is Copyright (C) 2021 Martin Zaefferer distributed under the GPL v3:
https://cran.r-project.org/package=CEGO

UMM is Copyright (C) 2021 Ekhine Irurozki and Manuel López-Ibáñez distributed
under the GPL v3: https://github.com/ekhiru/BB_optim

poliastro is Copyright (c) 2012-2021 Juan Luis Cano Rodríguez and the poliastro development team, released under the MIT license: https://github.com/poliastro/poliastro/

Other packages used or required by this software may have their own licenses.

**IMPORTANT NOTE:** Please be aware that the fact that this program is released as
Free Software does not excuse you from scientific propriety, which obligates
you to give appropriate credit! If you write a scientific paper describing
research that made substantive use of this program, it is your obligation as a
scientist to (a) mention the fashion in which this software was used in the
Methods section; (b) mention the appropriate citation in the References section:

    Manuel López-Ibáñez, Francisco Chicano, Rodrigo Gil-Merino. The Asteroid Routing Problem: A Benchmark for Expensive Black-Box Permutation Optimization. In Applications of Evolutionary Computation 2022, LNCS 13224, Springer Nature, Switzerland, 2022.

