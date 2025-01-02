# -*- coding: utf-8 -*-
"""
Created on Nov 2024

This file contains the software verification code for the FV implementation of
the 2DGRM. The results of this convergence analysis are published in Rao et al.
    'Two-dimensional general rate model with particle size distribution in CADET
    calibrated with high-definition CFD simulated intra-column data' (2025)

@author: jmbr
"""

#%%

import utility.convergence as convergence
import re
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np

from cadet import Cadet
from cadetrdm import ProjectRepo

import bench_func
import bench_configs
import utility.convergence as convergence

import settings_2Dchromatography

# database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
#     "/-/raw/core_tests/cadet_config/test_cadet-core/chromatography/"
database_path = None # TODO use database for model setup


sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core" / "2D_chromatography"

# The get_cadet_path function searches for the cadet-cli. If you want to use a specific source build, please define the path below
# TODO We use the source build here since one bug fix is not released yet, which was added in commit 0887fcb
cadet_path = convergence.get_cadet_path() # path to root folder of bin\cadet-cli 
commit_message = f"Benchmarks for 2DGRM FV 3-zone radial inlet variance convergence"

use_CASEMA_reference = True
n_jobs = -1

# small_test is set to true to define a minimal benchmark, which can be used
# to see if the simulations still run and see first results.
# To run the full extensive benchmarks, this needs to be set to false.

small_test = True
rdm_debug_mode = False
rerun_sims = True

#%% We define multiple settings convering binding modes, surface diffusion and
### multiple particle types. All settings consider three radial zones.

nRadialZones = 3
target_zone = 0 # only required for analytical solution: Here, we only consider the solution of one radial zone (volume average)

settings = [
    { # PURE COLUMN TRANSPORT CASE
    'film_diffusion' : 0.0,
    # 'col_dispersion_radial' : 0.0,
    'analytical_reference' : use_CASEMA_reference, # If set to true, solution time 0.0 is ignored since its not computed by the analytical solution (CADET-Semi-Analytic)
    'nRadialZones' : 3,
    'name' : '2DGRM3Zone_noBnd_1Comp',
    'adsorption_model' : 'NONE',
    'par_surfdiffusion' : 0.0
    },
    { # 1parType, dynamic binding, no surface diffusion
    'analytical_reference' : use_CASEMA_reference,
    'nRadialZones' : 3,
    'name' : '2DGRM3Zone_dynLin_1Comp',
    'adsorption_model' : 'LINEAR',
    'adsorption.is_kinetic' : 1,
    'par_surfdiffusion' : 0.0
    },
    { # 1parType, dynamic binding, with surface diffusion
    'analytical_reference' : use_CASEMA_reference,
    'nRadialZones' : 3,
    'name' : '2DGRMsd3Zone_dynLin_1Comp',
    'adsorption_model' : 'LINEAR',
    'adsorption.is_kinetic' : 1,
    'par_surfdiffusion' : 1e-11
    },
    { # 1parType, req binding, no surface diffusion
    'analytical_reference' : use_CASEMA_reference,
    'nRadialZones' : 3,
    'name' : '2DGRM3Zone_reqLin_1Comp',
    'adsorption_model' : 'LINEAR',
    'adsorption.is_kinetic' : 0,
    'par_surfdiffusion' : 0.0,
    'init_cp' : [0.0],
    'init_cs' : [0.0]
    },
    { # 1parType, req binding, with surface diffusion
    'analytical_reference' : use_CASEMA_reference,
    'nRadialZones' : 3,
    'name' : '2DGRMsd3Zone_reqLin_1Comp',
    'adsorption_model' : 'LINEAR',
    'adsorption.is_kinetic' : 0,
    'par_surfdiffusion' : 1e-11,
    'init_cp' : [0.0],
    'init_cs' : [0.0]
    },
    { # 4parType: 
    'analytical_reference' : use_CASEMA_reference,
    'nRadialZones' : 3,
    'name' : '2DGRM2parType3Zone_1Comp' if small_test else'2DGRM4parType3Zone_1Comp',
    'npartype' : 2 if small_test else 4,
    'par_type_volfrac' : [0.5, 0.5] if small_test else [0.3, 0.35, 0.15, 0.2],
    'par_radius' : [45E-6, 75E-6] if small_test else [45E-6, 75E-6, 25E-6, 60E-6],
    'par_porosity' : [0.75, 0.7] if small_test else [0.75, 0.7, 0.8, 0.65],
    'nbound' : [1, 1] if small_test else [1, 1, 0, 1],
    'init_cp' : [0.0, 0.0] if small_test else [0.0, 0.0, 0.0, 0.0],
    'init_cs' : [0.0, 0.0] if small_test else [0.0, 0.0, 0.0], # unbound component is ignored
    'film_diffusion' : [6.9E-6, 6E-6] if small_test else [6.9E-6, 6E-6, 6.5E-6, 6.7E-6],
    'par_diffusion' : [5E-11, 3E-11] if small_test else [6.07E-11, 5E-11, 3E-11, 4E-11],
    'par_surfdiffusion' : [5E-11, 0.0] if small_test else [1E-11, 5E-11, 0.0], # unbound component is ignored
    'adsorption_model' : ['LINEAR', 'LINEAR'] if small_test else ['LINEAR', 'LINEAR', 'NONE', 'LINEAR'],
    'adsorption.is_kinetic' : [0, 1] if small_test else [0, 1, 0, 0],
    'adsorption.lin_ka' : [35.5, 4.5] if small_test else [35.5, 4.5, 0, 0.25],
    'adsorption.lin_kd' : [1.0, 0.15] if small_test else [1.0, 0.15, 0, 1.0]
    }
    ]

ref_file_names = ['data/CASEMA_reference/ref_2DGRM3Zone_noBnd_1Comp_radZ3.h5',
                  'data/CASEMA_reference/ref_2DGRM3Zone_dynLin_1Comp_radZ3.h5',
                  'data/CASEMA_reference/ref_2DGRMsd3Zone_dynLin_1Comp_radZ3.h5',
                  'data/CASEMA_reference/ref_2DGRM3Zone_reqLin_1Comp_radZ3.h5',
                  'data/CASEMA_reference/ref_2DGRMsd3Zone_reqLin_1Comp_radZ3.h5',
                  'data/CASEMA_reference/ref_2DGRM2parType3Zone_1Comp_radZ3.h5' if small_test else 'data/CASEMA_reference/ref_2DGRM4parType3Zone_1Comp_radZ3.h5'
                  ]
    
with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):

    os.makedirs(output_path, exist_ok=True)
    
    # %% Define benchmarks
    
    cadet_configs = []
    config_names = []
    include_sens = []
    ref_files = []
    
    if use_CASEMA_reference:
        
        for idx in range(len(settings)):
            
            ref_files.extend(
                [convergence.get_solution(
                    str(project_repo.output_path.parent / ref_file_names[idx]), unit='unit_000', which='outlet_port_' + str(target_zone).zfill(3)
                    )]
                )
            
        ref_files = [ref_files]
        
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    rad_methods = []
    rad_discs = []
    par_methods = []
    par_discs = []
    refinement_IDs = []

    
    def GRM2D_FV_Benchmark(small_test=False, **kwargs):

        nDisc = 4 if small_test else 6
        nRadialZones=kwargs.get('nRadialZones', 3)
        
        benchmark_config = {
            'cadet_config_jsons': [
                settings_2Dchromatography.SamDiss_2DVerificationSetting(
                    radNElem=nRadialZones,
                    rad_inlet_profile=None,
                    USE_MODIFIED_NEWTON=0, axMethod=0, **kwargs)
            ],
            'include_sens': [
                False
            ],
            'ref_files': [
                [None]
            ],
            'refinement_ID': [
                '000'
            ],
            'unit_IDs': [
                str(nRadialZones + 1 + target_zone).zfill(3) if kwargs.get('analytical_reference', 0) else '000'
            ],
            'which': [
                'outlet' if kwargs.get('analytical_reference', 0) else 'radial_outlet' # outlet_port_000
            ],
            'idas_abstol': [
                [1e-10]
            ],
            'ax_methods': [
                [0]
            ],
            'ax_discs': [
                [bench_func.disc_list(4, nDisc)]
            ],
            'rad_methods': [
                [0]
            ],
            'rad_discs': [
                [bench_func.disc_list(nRadialZones, nDisc)]
            ],
            'par_methods': [
                [0]
            ],
            'par_discs': [ # same number of particle cells as radial cells
                [bench_func.disc_list(nRadialZones, nDisc)]
            ]
        }

        return benchmark_config
    
    # %% create benchmark configurations
    
    for setting in settings:
        addition = GRM2D_FV_Benchmark(small_test=small_test, **setting)
        
        bench_configs.add_benchmark(
            cadet_configs, include_sens, ref_files, unit_IDs, which,
            idas_abstol,
            ax_methods, ax_discs, rad_methods=rad_methods, rad_discs=rad_discs,
            par_methods=par_methods, par_discs=par_discs,
            refinement_IDs=refinement_IDs,
            addition=addition)
    
        config_names.extend([setting['name']])
    
    # %% Run convergence analysis
    
    Cadet.cadet_path = cadet_path
    
    bench_func.run_convergence_analysis(
        database_path=database_path, output_path=output_path,
        cadet_path=cadet_path,
        cadet_configs=cadet_configs,
        cadet_config_names=config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        rad_methods=rad_methods, rad_discs=rad_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        n_jobs=n_jobs,
        rad_inlet_profile=None,
        rerun_sims=rerun_sims,
        refinement_IDs=refinement_IDs,
        analytical_reference=settings[0].get('analytical_reference', False)
    )
