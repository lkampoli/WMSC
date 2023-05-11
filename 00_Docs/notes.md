# Wall-mounted square cylinder data

The following are brief descriptions of the sent data. Please ask if anything is unclear.

* `01_Data`:

    * This is the high-fidelity hybrid RANS/LES data folder

    * The folder is mostly unchanged from how I received it from Ali

    * There are some additional postprocessing files in `postProcessing/sets`

    * `kOmegaSSTNewFrozenAppro` should be the "frozen" solver to derive `omega` and `Rterm`

    * `frozenFoam` might be the previous version of that solver

    * `20059.95` should be the data resulting from `kOmegaSSTNewFrozenAppro`

    * `training_data`: training data, derived by Ali

    * `training`: Ali's GEP training files, I never used them

    * The remaining files/folder should be OpenFOAM/Slurm standard

* `02_Training`:

    * This folder contains scripts to extract training data and run GEP training

    * Example: `rnd11` refers to the training domain: 1.5 <= x < 8, -5 <= y < 0, z <= 5

    * Use `generate_points.py` to create `training_data_rnd11/points_rnd11.H`

    * Run OpenFOAM postprocessing to sample at the generated points (the corresponding sampleDict is unfortunately lost on former cluster)

    * Apply `points2edf.py` to `01_Data/postProcessing/sets/20059.95/points_rnd11_*` to create `training_data_rnd11` data and rename it via `TtoV.py`

    * Run `split_train_test.py` to create `training_data_rnd11_1e4`, `training_data_rnd11_1e5`, `testing_data_rnd11_1e4` and `testing_data_rnd11_1e5`

    * `training_rnd11_1e5_p100_g4_AC5_LM_l5_no10_comp` is an exemplary complete training folder

* `03_Simulation`:

    * This folder contains OpenFOAM solvers and exemplary RANS calculations

    * `smsc_rans_half_Baseline_k_omega_SST`: Ali's baseline RANS folder

    * `WMSC_BSL_kOmegaSST`: Baseline RANS folder

    * `kOmegaSSTx`: Solver to read in anisotropy (`Ax_`) and turbulence production correction (`Rx_`) models from `lib/nonLinearModel.H` files

    * `kOmegaSSTxc`: Extension of above solver that applies model only in constrained domain

    * `WMSC_template` and `WMSC_template_xc`: Template folders to set up simulations with the above solvers

    * `WMSC_BSL_kOmegaSSTx` and `WMSC_BSL_kOmegaSSTxc`: Baseline folders applying the above solvers

    * `WMSC_rnd611_1e5_p100_g4_AC5_LM_l5_no10_comp_As0c79_Rs2c28`: Simulation applying trained `Ax_` and `Rx_` models

