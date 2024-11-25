# -*- coding: utf-8 -*-
"""
Created Oct 2024

This script executes all the CADET-Verification tests for CADET-Core.
Modify the input in the 'User Input' section if needed.
To test if the script works, specify rdm_debug_mode and small_test as true.

Only specify rdm_debug_mode as False if you are sure that this run shall be
saved to the output repository!

@author: jmbr
""" 
  
#%% Include packages
import os
import sys
from pathlib import Path
import re
from joblib import Parallel, delayed
import numpy as np

from cadet import Cadet
from cadetrdm import ProjectRepo

import utility.convergence as convergence
import bench_func
import bench_configs

import chromatography
import crystallization
import MCT

#%% User Input

commit_message = f"Full run of CADET-Core verification"

rdm_debug_mode = 1 # Run CADET-RDM in debug mode to test if the script works

small_test = 1 # Defines a smaller test set (less numerical refinement steps)

n_jobs = -1 # For parallelization on the number of simulations

run_chromatography_tests = 1
run_crystallization_tests = 0
run_MCT_tests = 0

database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
    "/-/raw/core_tests/cadet_config/test_cadet-core/"

sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core"

# The get_cadet_path function searches for the cadet-cli. If you want to use a specific source build, please define the path below
cadet_path = convergence.get_cadet_path() # path to root folder of bin\cadet-cli 
 

# %% Run with CADET-RDM

with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):
    
    if run_chromatography_tests:
        chromatography.chromatography_tests(
            n_jobs=n_jobs, database_path=database_path+"chromatography/",
            small_test=small_test, sensitivities=True,
            output_path=str(output_path) + "/chromatography", cadet_path=cadet_path
            )
    
    if run_crystallization_tests:
        crystallization.crystallization_tests(
            n_jobs=n_jobs, database_path=database_path+"crystallization/",
            small_test=small_test,
            output_path=str(output_path) + "/crystallization", cadet_path=cadet_path
            )
    
    if run_MCT_tests:
        MCT.MCT_tests(
            n_jobs=n_jobs, database_path=database_path+"mct/",
            small_test=small_test,
            output_path=str(output_path) + "/mct", cadet_path=cadet_path
            )
    
