import os
from random import shuffle


def generate_combinations():
    model_sizes = [
        {'n_layers': 2, 'n_heads': 2, 'd_model': 128, 'd_ff': 512},
        {'n_layers': 4, 'n_heads': 4, 'd_model': 256, 'd_ff': 1024},
        {'n_layers': 6, 'n_heads': 8, 'd_model': 512, 'd_ff': 2048},
        # {'n_layers': 8, 'n_heads': 16, 'd_model': 1024, 'd_ff': 4096}  # too big for m2 mac mini to run reasonably fast
    ]
    dataset_sizes = [100, 500, 1_000, 5_000]
    n_bins_list = [128, 512, 2_048, 8_192]  # [128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384]
    coordinate_systems = ['spherical', 'cartesian']
    lr_list = [1e-3, 1e-4, 1e-5]
    batch_sizes = [16, 32, 64]

    combinations = []
    for model in model_sizes:
        for dataset_size in dataset_sizes:
            for n_bins in n_bins_list:
                for coord_sys in coordinate_systems:
                    for lr in lr_list:
                        for bs in batch_sizes:
                            comb = {
                                'n_layers': model['n_layers'],
                                'n_heads': model['n_heads'],
                                'd_model': model['d_model'],
                                'd_ff': model['d_ff'],
                                'dataset_size': dataset_size,
                                'n_bins': n_bins,
                                'coordinate_system': coord_sys,
                                'learning_rate': lr,
                                'batch_size': bs,
                                'input_length': 32,
                                'epochs': 10
                            }
                            combinations.append(comb)

    print(f"Total combinations: {len(combinations)}")
    shuffle(combinations)
    return combinations


def generate_bash_files(combinations):
    combs_per_machine = len(combinations) // N_MACHINES
    machine_combs = [combinations[i * combs_per_machine:(i + 1) * combs_per_machine] for i in range(N_MACHINES)]
    for machine_id, combs in enumerate(machine_combs):
        script_path = os.path.join(TARGET_DIR, f"run_machine_{machine_id}.sh")
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            for comb in combs:
                cmd = "python 2_train.py --trade_name \"scaling_laws_v1\" "
                for key, value in comb.items():
                    cmd += f"--{key} {value} "
                cmd += "\n"
                f.write(cmd)
    return


if __name__ == "__main__":

    N_MACHINES = 4
    TARGET_DIR = "."

    combinations = generate_combinations()

    generate_bash_files(combinations)
