import os
import pandas as pd
from astropy import units as u

import orbit_transformer as ot


def make_datasets(target_dir, n_orbits, r_bins, theta_bins, phi_bins):

    print(target_dir, n_orbits, r_bins, theta_bins, phi_bins)

    total_raw_csv_path = os.path.join(target_dir, f"orbits_dataset_total_{n_orbits}_raw.csv")

    train_raw_csv_path = total_raw_csv_path.replace("total", "train")
    val_raw_csv_path = total_raw_csv_path.replace("total", "val")
    test_raw_csv_path = total_raw_csv_path.replace("total", "test")

    train_tokenized_csv_path = train_raw_csv_path.replace("raw.csv", f"tokenized_{r_bins}_{theta_bins}_{phi_bins}.csv")
    val_tokenized_csv_path = val_raw_csv_path.replace("raw.csv", f"tokenized_{r_bins}_{theta_bins}_{phi_bins}.csv")
    test_tokenized_csv_path = test_raw_csv_path.replace("raw.csv", f"tokenized_{r_bins}_{theta_bins}_{phi_bins}.csv")

    if os.path.exists(total_raw_csv_path):
        df = pd.read_csv(total_raw_csv_path)
    else:
        df = ot.generate_orbits_dataset(
            n_orbits=n_orbits,
            orbit_types=["LEO", "MEO", "HEO"],  # , "GEO"
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


    if os.path.exists(train_tokenized_csv_path):
        return

    tokenizer = ot.SphericalCoordinateTokenizer(
        r_bins=r_bins,
        theta_bins=theta_bins,
        phi_bins=phi_bins,
    )

    df_train_tokenized = tokenizer.transform(df_train)
    df_train_tokenized.to_csv(train_tokenized_csv_path, index=False)

    df_val_tokenized = tokenizer.transform(df_val)
    df_val_tokenized.to_csv(val_tokenized_csv_path, index=False)

    df_test_tokenized = tokenizer.transform(df_test)
    df_test_tokenized.to_csv(test_tokenized_csv_path, index=False)


if __name__ == "__main__":

    target_dir = os.path.join(".", "data")

    for n_orbits in [100, 500, 1_000, 5_000, 10_000]:
        for n_bins in [10, 50, 100, 500, 1_000]:

            r_bins = n_bins
            theta_bins = n_bins
            phi_bins = n_bins

            make_datasets(target_dir, n_orbits, r_bins, theta_bins, phi_bins)


    # generate_raw = False
    # generate_splits = False
    # tokenize_splits = True
    # visualize_tokenizer = False

    # total_raw_csv_path = os.path.join(".", "data", "orbits_dataset_1000_raw.csv")

    # train_raw_csv_path = total_raw_csv_path.replace("raw.csv", "train_raw.csv")
    # val_raw_csv_path = total_raw_csv_path.replace("raw.csv", "val_raw.csv") 
    # test_raw_csv_path = total_raw_csv_path.replace("raw.csv", "test_raw.csv")
    
    # train_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "train_tokenized.csv")
    # val_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "val_tokenized.csv")
    # test_tokenized_csv_path = total_raw_csv_path.replace("raw.csv", "test_tokenized.csv")

    # if generate_raw:
    #     df = ot.generate_orbits_dataset(
    #         n_orbits=1_000,
    #         orbit_types=["LEO", "MEO", "HEO", "GEO"],
    #         time_step=60*u.s,
    #         out_csv=total_raw_csv_path
    #     )

    # if generate_splits:

    #     df = pd.read_csv(total_raw_csv_path)

    #     df_train, df_val, df_test = ot.split_orbits_by_id(df)

    #     df_train.to_csv(train_raw_csv_path, index=False)
    #     df_val.to_csv(val_raw_csv_path, index=False)
    #     df_test.to_csv(test_raw_csv_path, index=False)

    # if tokenize_splits:
    #     tokenizer = ot.SphericalCoordinateTokenizer(
    #         r_bins=200,
    #         theta_bins=180,
    #         phi_bins=360,
    #         theta_min=0.0,
    #         theta_max=180.0,
    #         phi_min=-180.0,
    #         phi_max=180.0,
    #     )

    #     if visualize_tokenizer:
    #         tokenizer.visualize_all_bins_3d()

    #     df_train = pd.read_csv(train_raw_csv_path)
    #     df_train_tokenized = tokenizer.transform(df_train)
    #     df_train_tokenized.to_csv(train_tokenized_csv_path, index=False)

    #     df_val = pd.read_csv(val_raw_csv_path)
    #     df_val_tokenized = tokenizer.transform(df_val)
    #     df_val_tokenized.to_csv(val_tokenized_csv_path, index=False)

    #     df_test = pd.read_csv(test_raw_csv_path)
    #     df_test_tokenized = tokenizer.transform(df_test)
    #     df_test_tokenized.to_csv(test_tokenized_csv_path, index=False)
