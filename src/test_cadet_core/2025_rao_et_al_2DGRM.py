# -*- coding: utf-8 -*-
"""
Created on Nov 2024

This file contains the software verification code for the FV implementation of
the 2DGRM. The results of this convergence analysis are published in Rao et al.
    'Two-dimensional general rate model with particle size distribution in CADET
    calibrated with high-definition CFD simulated intra-column data' (2025)

@author: jmbr
"""

# %% import packages and files
import utility.convergence as convergence
import re
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import json
import shutil

from cadet import Cadet
from cadetrdm import ProjectRepo

from utility import convergence
import bench_func
import bench_configs
import settings_2Dchromatography


# %% set variables for evaluation (can be modified)

sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "paper" / "2025_Rao_et_al_2DGRM"

# The get_cadet_path function searches for the cadet-cli. If you want to use a specific source build, please define the path below
# path to root folder of bin\cadet-cli
cadet_path = convergence.get_cadet_path()
commit_message = f"Benchmarks for 2DGRM FV 3-zone radial inlet variance convergence"

use_CASEMA_reference = True  # Use analytical reference provided in data folder
n_jobs = -1

# small_test is set to true to define a minimal benchmark, which can be used
# to see if the simulations still run and see first results.
# To run the full extensive benchmarks, this needs to be set to false.
small_test = True
rdm_debug_mode = False
rerun_sims = True


# %% We define multiple settings convering binding modes, surface diffusion and
# multiple particle types. All settings consider three radial zones.
# An analytical solution can be provided and the EOC is computed for three
# radial zones. Ultimately, the discrete maximum norm of the zonal errors is
# considered to compute the EOC.
def GRM2D_linBnd_tests(
        n_jobs, database_path, small_test,
        output_path, cadet_path, reference_data_path=None,
        use_CASEMA_reference=True, rerun_sims=True):

    os.makedirs(output_path, exist_ok=True)

    nRadialZones = 3
    n_settings = 6

    references = [None] * n_settings

    if use_CASEMA_reference:

        references = []
        ref_file_names = ['CASEMA_reference/ref_2DGRM3Zone_noBnd_1Comp_radZ3.h5',
                          'CASEMA_reference/ref_2DGRM3Zone_dynLin_1Comp_radZ3.h5',
                          'CASEMA_reference/ref_2DGRMsd3Zone_dynLin_1Comp_radZ3.h5',
                          'CASEMA_reference/ref_2DGRM3Zone_reqLin_1Comp_radZ3.h5',
                          'CASEMA_reference/ref_2DGRMsd3Zone_reqLin_1Comp_radZ3.h5',
                          'CASEMA_reference/ref_2DGRM2parType3Zone_1Comp_radZ3.h5' if small_test else 'data/CASEMA_reference/ref_2DGRM4parType3Zone_1Comp_radZ3.h5'
                          ]

        # Note: All zones will be considered when use_CASEMA_reference is true.
        # We start with the first and compute the other two in a second step.
        # Finally, we compute a discrete norm of the zonal errors to compute the EOC.
        for idx in range(n_settings):
            # note that we consider radial zone 0
            references.extend(
                [convergence.get_solution(
                    reference_data_path + '/' + ref_file_names[idx], unit='unit_000', which='outlet_port_' + str(0).zfill(3)
                )]
            )

    def get_settings():
        return [
            {  # PURE COLUMN TRANSPORT CASE
                'film_diffusion': 0.0,
                # 'col_dispersion_radial' : 0.0,
                # If set to true, solution time 0.0 is ignored since its not computed by the analytical solution (CADET-Semi-Analytic)
                'analytical_reference': use_CASEMA_reference,
                'nRadialZones': 3,
                'name': '2DGRM3Zone_noBnd_1Comp',
                'adsorption_model': 'NONE',
                'par_surfdiffusion': 0.0,
                'reference': references[0]
            },
            {  # 1parType, dynamic binding, no surface diffusion
                'analytical_reference': use_CASEMA_reference,
                'nRadialZones': 3,
                'name': '2DGRM3Zone_dynLin_1Comp',
                'adsorption_model': 'LINEAR',
                'adsorption.is_kinetic': 1,
                'par_surfdiffusion': 0.0,
                'reference': references[1]
            },
            {  # 1parType, dynamic binding, with surface diffusion
                'analytical_reference': use_CASEMA_reference,
                'nRadialZones': 3,
                'name': '2DGRMsd3Zone_dynLin_1Comp',
                'adsorption_model': 'LINEAR',
                'adsorption.is_kinetic': 1,
                'par_surfdiffusion': 1e-11,
                'reference': references[2]
            },
            {  # 1parType, req binding, no surface diffusion
                'analytical_reference': use_CASEMA_reference,
                'nRadialZones': 3,
                'name': '2DGRM3Zone_reqLin_1Comp',
                'adsorption_model': 'LINEAR',
                'adsorption.is_kinetic': 0,
                'par_surfdiffusion': 0.0,
                'init_cp': [0.0],
                'init_cs': [0.0],
                'reference': references[3]
            },
            {  # 1parType, req binding, with surface diffusion
                'analytical_reference': use_CASEMA_reference,
                'nRadialZones': 3,
                'name': '2DGRMsd3Zone_reqLin_1Comp',
                'adsorption_model': 'LINEAR',
                'adsorption.is_kinetic': 0,
                'par_surfdiffusion': 1e-11,
                'init_cp': [0.0],
                'init_cs': [0.0],
                'reference': references[4]
            },
            {  # 4parType:
                'analytical_reference': use_CASEMA_reference,
                'nRadialZones': 3,
                'name': '2DGRM2parType3Zone_1Comp' if small_test else '2DGRM4parType3Zone_1Comp',
                'npartype': 2 if small_test else 4,
                'par_type_volfrac': [0.5, 0.5] if small_test else [0.3, 0.35, 0.15, 0.2],
                'par_radius': [45E-6, 75E-6] if small_test else [45E-6, 75E-6, 25E-6, 60E-6],
                'par_porosity': [0.75, 0.7] if small_test else [0.75, 0.7, 0.8, 0.65],
                'nbound': [1, 1] if small_test else [1, 1, 0, 1],
                'init_cp': [0.0, 0.0] if small_test else [0.0, 0.0, 0.0, 0.0],
                # unbound component is ignored
                'init_cs': [0.0, 0.0] if small_test else [0.0, 0.0, 0.0],
                'film_diffusion': [6.9E-6, 6E-6] if small_test else [6.9E-6, 6E-6, 6.5E-6, 6.7E-6],
                'par_diffusion': [5E-11, 3E-11] if small_test else [6.07E-11, 5E-11, 3E-11, 4E-11],
                # unbound component is ignored
                'par_surfdiffusion': [5E-11, 0.0] if small_test else [1E-11, 5E-11, 0.0],
                'adsorption_model': ['LINEAR', 'LINEAR'] if small_test else ['LINEAR', 'LINEAR', 'NONE', 'LINEAR'],
                'adsorption.is_kinetic': [0, 1] if small_test else [0, 1, 0, 0],
                'adsorption.lin_ka': [35.5, 4.5] if small_test else [35.5, 4.5, 0, 0.25],
                'adsorption.lin_kd': [1.0, 0.15] if small_test else [1.0, 0.15, 0, 1.0],
                'reference': references[5]
            }
        ]

    # %% Define benchmarks

    settings = get_settings()

    cadet_configs = []
    config_names = []
    include_sens = []
    ref_files = []
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
        nRadialZones = kwargs.get('nRadialZones', 3)

        benchmark_config = {
            'cadet_config_jsons': [
                settings_2Dchromatography.GRM2D_linBnd_benchmark1(
                    radNElem=nRadialZones,
                    rad_inlet_profile=None,
                    USE_MODIFIED_NEWTON=0, axMethod=0, **kwargs)
            ],
            'include_sens': [
                False
            ],
            'ref_files': [
                [kwargs.get('reference', None)]
            ],
            'refinement_ID': [
                '000'
            ],
            'unit_IDs': [  # note that we consider radial zone 0
                str(nRadialZones + 1 +
                    0).zfill(3) if kwargs.get('analytical_reference', 0) else '000'
            ],
            'which': [
                'outlet' if kwargs.get(
                    'analytical_reference', 0) else 'radial_outlet'  # outlet_port_000
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
            'par_discs': [  # same number of particle cells as radial cells
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
        analytical_reference=use_CASEMA_reference
    )

    # For the analytical solution, we compute the discrete norm of the errors from each zone
    if use_CASEMA_reference:

        def copy_json_file(source_file, destination_file):
            try:
                # Copy the file
                shutil.copy(source_file, destination_file)
            #     print(f"Copied {source_file} to {destination_file}")
            except FileNotFoundError:
                print(f"File {source_file} not found!")
            except Exception as e:
                print(f"An error occurred: {e}")

        def rename_json_file(original_file, new_file):

            # Rename the file
            try:
                os.rename(original_file, new_file)
            #     print(f"Renamed {original_file} to {new_file}")
            except FileNotFoundError:
                print(f"File {original_file} not found!")
            except Exception as e:
                print(f"An error occurred: {e}")

        # save old results under new name for corresponding port
        for idx in range(len(settings)):

            old_name = str(output_path) + '/convergence_' + \
                settings[idx]['name'] + '.json'
            new_name = str(output_path) + '/convergence_' + 'port' + \
                str(0).zfill(3) + '_' + settings[idx]['name'] + '.json'
            rename_json_file(old_name, new_name)

        for target_zone in range(1, nRadialZones):

            references = []

            for idx in range(n_settings):

                # get the references at the other ports
                references.extend(
                    [convergence.get_solution(
                        reference_data_path + '/' + ref_file_names[idx], unit='unit_000', which='outlet_port_' + str(target_zone).zfill(3)
                    )]
                )

            unit_IDs = [str(4 + target_zone).zfill(3)] * \
                n_settings  # 4 + target_zone

            # calculate results for next port

            ref_files = [[references[0]], [references[1]], [references[2]],
                         [references[3]], [references[4]], [references[5]]]

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
                rerun_sims=False,
                refinement_IDs=refinement_IDs,
                analytical_reference=use_CASEMA_reference
            )

            # save new results under new name for corresponding port

            for idx in range(len(settings)):

                old_name = str(output_path) + '/convergence_' + \
                    settings[idx]['name'] + '.json'
                new_name = str(output_path) + '/convergence_' + 'port' + \
                    str(target_zone).zfill(3) + '_' + \
                    settings[idx]['name'] + '.json'
                rename_json_file(old_name, new_name)

        # Calculate Discrete Maximum Norm over all radial zones

        for idx in range(len(settings)):

            # create target file based off the first file
            target_name = str(output_path) + '/convergence_' + \
                settings[idx]['name'] + '.json'
            copy_name = str(output_path) + '/convergence_' + \
                'port000_' + settings[idx]['name'] + '.json'
            copy_json_file(copy_name, target_name)

            for target_zone in range(nRadialZones):

                file_name = str(output_path) + '/convergence_' + 'port' + \
                    str(target_zone).zfill(3) + '_' + \
                    settings[idx]['name'] + '.json'

                with open(file_name, "r") as file:
                    data = json.load(file)

                if target_zone == 0:
                    disc = data['convergence']['FV']['outlet']['$N_e^z$']
                    maxError = np.array(
                        data['convergence']['FV']['outlet']['Max. error'])
                    L1Error = np.array(
                        data['convergence']['FV']['outlet']['$L^1$ error'])
                    L2Error = np.array(
                        data['convergence']['FV']['outlet']['$L^2$ error'])
                else:  # maximum norm
                    maxError = np.maximum(maxError, np.array(
                        data['convergence']['FV']['outlet']['Max. error']))
                    L1Error = np.maximum(L1Error, np.array(
                        data['convergence']['FV']['outlet']['$L^1$ error']))
                    L2Error = np.maximum(L2Error, np.array(
                        data['convergence']['FV']['outlet']['$L^2$ error']))

            maxEOC = np.insert(
                convergence.calculate_eoc(disc, maxError), 0, 0.0)
            L1EOC = np.insert(convergence.calculate_eoc(disc, L1Error), 0, 0.0)
            L2EOC = np.insert(convergence.calculate_eoc(disc, L2Error), 0, 0.0)

            with open(target_name, "r") as file:
                target_data = json.load(file)

            target_data['convergence']['FV']['outlet']['Max. error'] = maxError.tolist()
            target_data['convergence']['FV']['outlet']['Max. EOC'] = maxEOC.tolist()
            target_data['convergence']['FV']['outlet']['$L^1$ error'] = L1Error.tolist()
            target_data['convergence']['FV']['outlet']['$L^1$ EOC'] = L1EOC.tolist()
            target_data['convergence']['FV']['outlet']['$L^2$ error'] = L2Error.tolist()
            target_data['convergence']['FV']['outlet']['$L^2$ EOC'] = L2EOC.tolist()

            print("jojo setting no. ", idx)
            print(target_data)
            with open(target_name, "w") as file:
                # Write with pretty formatting
                json.dump(target_data, file, indent=4)


# %% Execute convergence analysis with CADET-RDM


with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):

    GRM2D_linBnd_tests(
        n_jobs=n_jobs, database_path=None, small_test=small_test,
        output_path=output_path, cadet_path=cadet_path,
        reference_data_path=str(project_repo.output_path.parent / 'data'),
        use_CASEMA_reference=use_CASEMA_reference, rerun_sims=rerun_sims)
