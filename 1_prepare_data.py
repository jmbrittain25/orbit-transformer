import os
import pandas as pd
from astropy import units as u

import orbit_transformer as ot


def make_datasets(target_dir, n_orbits, n_bins):

    print(target_dir, n_orbits, n_bins)

    total_raw_csv_path = os.path.join(target_dir, f"orbits_HEO_only_dataset_{n_orbits}_raw.csv")

    train_raw_csv_path = total_raw_csv_path.replace("total", "train")
    val_raw_csv_path = total_raw_csv_path.replace("total", "val")
    test_raw_csv_path = total_raw_csv_path.replace("total", "test")

    if os.path.exists(total_raw_csv_path):
        df = pd.read_csv(total_raw_csv_path)
    else:
        df = ot.generate_orbits_dataset(
            n_orbits=n_orbits,
            orbit_types=["HEO"],  # "LEO", "MEO", "GEO"
            time_step=60*u.s,
            out_csv=total_raw_csv_path
        )

    if os.path.exists(train_raw_csv_path):
        df_train = pd.read_csv(train_raw_csv_path)
        df_val = pd.read_csv(val_raw_csv_path)
        df_test = pd.read_csv(test_raw_csv_path)
    else:
        df_train, df_val, df_test = ot.split_orbits_by_id(df)

        df_train.to_csv(train_raw_csv_path, index=False)
        df_val.to_csv(val_raw_csv_path, index=False)
        df_test.to_csv(test_raw_csv_path, index=False)



if __name__ == "__main__":

    target_dir = os.path.join(".", "data")

    # n_orbits = 100
    # target_path = os.path.join(target_dir, f"HEO_only_val_dataset_{n_orbits}_orbits.csv")

    for n_orbits in [50_000, 100_000]:
        target_path = os.path.join(target_dir, f"orbits_HEO_only_dataset_{n_orbits}_raw.csv")

        ot.generate_orbits_dataset(
            n_orbits=n_orbits,
            orbit_types=["HEO"],  # "LEO", "MEO", "GEO"
            time_step=60*u.s,
            out_csv=target_path,
            num_workers=os.cpu_count() - 1
        )

    # for n_orbits in [10_000]:  # 100, 500, 1_000, 5_000, , 50_000, 100_000
    #     for n_bins in [128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384]:

    #         make_datasets(target_dir, n_orbits, n_bins)


