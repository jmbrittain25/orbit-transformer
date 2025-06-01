import numpy as np
import pandas as pd
from astropy import units as u
from poliastro.constants import J2000
from astropy.time import Time
from astropy.coordinates import (
    GCRS, ITRS, CartesianRepresentation, CartesianDifferential
)
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from tqdm import tqdm
import multiprocessing
from functools import partial


def generate_random_orbit(epoch, orbit_type="Random"):
    """
    Your existing random orbit generator. 
    Returns a poliastro.twobody.Orbit object.
    """
    if orbit_type == "Random":
        orbit_type = np.random.choice(["LEO", "MEO", "HEO", "GEO"])

    # Define altitude ranges for each orbital regime
    if orbit_type == "LEO":
        periapsis_alt_km_min, periapsis_alt_km_max = 160, 2000
        apoapsis_alt_km_min, apoapsis_alt_km_max = 160, 2000
    elif orbit_type == "MEO":
        periapsis_alt_km_min, periapsis_alt_km_max = 2000, 35786
        apoapsis_alt_km_min, apoapsis_alt_km_max = 2000, 35786
    elif orbit_type == "HEO":
        periapsis_alt_km_min, periapsis_alt_km_max = 160, 2000
        apoapsis_alt_km_min, apoapsis_alt_km_max = 2000, 35786
    elif orbit_type == "GEO":
        periapsis_alt_km_min, periapsis_alt_km_max = 35786, 35786
        apoapsis_alt_km_min, apoapsis_alt_km_max = 35786, 35786
    else:
        raise ValueError("Invalid orbit type. Must be 'LEO', 'MEO', 'HEO', 'GEO' or 'Random'.")

    # Random altitudes
    periapsis_alt = np.random.uniform(periapsis_alt_km_min, periapsis_alt_km_max)
    apoapsis_alt = np.random.uniform(apoapsis_alt_km_min, apoapsis_alt_km_max)

    # Ensure apoapsis >= periapsis
    while apoapsis_alt < periapsis_alt:
        apoapsis_alt = np.random.uniform(apoapsis_alt_km_min, apoapsis_alt_km_max)

    # Calculate distances from Earth's center
    periapsis = 6371 + periapsis_alt
    apoapsis = 6371 + apoapsis_alt

    # Semi-major axis and eccentricity
    a = (periapsis + apoapsis) / 2 * u.km
    ecc = (apoapsis - periapsis) / (apoapsis + periapsis) * u.one

    # Random angles
    inc = np.random.uniform(0, 180) * u.deg
    raan = np.random.uniform(0, 360) * u.deg
    argp = np.random.uniform(0, 360) * u.deg
    nu = np.random.uniform(0, 360) * u.deg

    return Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch=epoch)

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates (km) to spherical (r, theta, phi).
    - r (km)
    - theta (deg) = polar angle from z-axis (colatitude)
    - phi (deg)   = azimuth angle in x-y plane
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    if r < 1e-12:
        return (0.0, 0.0, 0.0)
    theta = np.degrees(np.arccos(z / r))  # 0 to 180
    phi = np.degrees(np.arctan2(y, x))    # -180 to 180
    return (r, theta, phi)


def propagate_orbit_to_df(orbit_obj, orbit_id, orbit_regime, time_step=60*u.s):
    """
    Propagate a single poliastro Orbit at fixed time intervals.
    Collect position & velocity in both ECI (GCRS) and ECEF (ITRS).
    Return a pandas DataFrame with one row per timestep.

    Parameters
    ----------
    orbit_obj : poliastro.twobody.Orbit
    orbit_id : str
        A unique identifier string for the orbit
    orbit_regime : str
        e.g. 'LEO', 'MEO', 'HEO', 'GEO', or 'Random'
    time_step : astropy.units.Quantity
        Timestep, e.g. 60*u.s

    Returns
    -------
    df : pandas.DataFrame
        Columns: orbit_id, orbit_regime, epoch, period_s, time_s,
                 x_eci_km, y_eci_km, z_eci_km, vx_eci_km_s, vy_eci_km_s, vz_eci_km_s,
                 r_eci_km, theta_eci_deg, phi_eci_deg,
                 x_ecef_km, y_ecef_km, z_ecef_km, vx_ecef_km_s, vy_ecef_km_s, vz_ecef_km_s,
                 r_ecef_km, theta_ecef_deg, phi_ecef_deg,
                 sma_km, ecc, inc_deg, raan_deg, argp_deg, nu_deg
    """
    rows = []
    period_s = orbit_obj.period.to_value(u.s)
    epoch_str = orbit_obj.epoch.isot  # epoch in ISO format

    # We'll propagate one full period minus one time_step
    num_steps = int(period_s // time_step.to_value(u.s)) - 1

    for step in range(num_steps):
        dt = step * time_step
        # Propagate the orbit to the desired offset from epoch
        new_orbit = orbit_obj.propagate(dt)

        # ECI coordinates (poliastro's representation is effectively GCRS)
        r_eci = new_orbit.r.to(u.km).value  # [x, y, z]
        v_eci = new_orbit.v.to(u.km/u.s).value  # [vx, vy, vz]

        x_eci, y_eci, z_eci = r_eci
        vx_eci, vy_eci, vz_eci = v_eci

        # Convert ECI -> ECEF using GCRS -> ITRS transform
        # We attach obstime = new_orbit.epoch + dt
        current_time = new_orbit.epoch + dt
        gcrs_coord = GCRS(
            x=x_eci*u.km, y=y_eci*u.km, z=z_eci*u.km,
            v_x=vx_eci*(u.km/u.s), v_y=vy_eci*(u.km/u.s), v_z=vz_eci*(u.km/u.s),
            representation_type=CartesianRepresentation,
            differential_type=CartesianDifferential,
            obstime=current_time
        )
        itrs_coord = gcrs_coord.transform_to(ITRS(obstime=current_time))

        x_ecef = itrs_coord.x.to_value(u.km)
        y_ecef = itrs_coord.y.to_value(u.km)
        z_ecef = itrs_coord.z.to_value(u.km)
        vx_ecef = itrs_coord.v_x.to_value(u.km/u.s)
        vy_ecef = itrs_coord.v_y.to_value(u.km/u.s)
        vz_ecef = itrs_coord.v_z.to_value(u.km/u.s)

        # Spherical coords in ECI
        r_eci_val, theta_eci_val, phi_eci_val = cartesian_to_spherical(x_eci, y_eci, z_eci)
        # Spherical coords in ECEF
        r_ecef_val, theta_ecef_val, phi_ecef_val = cartesian_to_spherical(x_ecef, y_ecef, z_ecef)

        # Classical orbital elements (use new_orbit)
        sma_km   = new_orbit.a.to_value(u.km)
        ecc_val  = new_orbit.ecc.value
        inc_deg  = new_orbit.inc.to_value(u.deg)
        raan_deg = new_orbit.raan.to_value(u.deg)
        argp_deg = new_orbit.argp.to_value(u.deg)
        nu_deg   = new_orbit.nu.to_value(u.deg)

        row = {
            "orbit_id": orbit_id,
            "orbit_regime": orbit_regime,
            "epoch": epoch_str,
            "period_s": period_s,
            "time_s": dt.to_value(u.s),

            # ECI (GCRS) cartesian
            "x_eci_km": x_eci,
            "y_eci_km": y_eci,
            "z_eci_km": z_eci,
            "vx_eci_km_s": vx_eci,
            "vy_eci_km_s": vy_eci,
            "vz_eci_km_s": vz_eci,

            # ECI spherical
            "r_eci_km": r_eci_val,
            "theta_eci_deg": theta_eci_val,
            "phi_eci_deg": phi_eci_val,

            # ECEF (ITRS) cartesian
            "x_ecef_km": x_ecef,
            "y_ecef_km": y_ecef,
            "z_ecef_km": z_ecef,
            "vx_ecef_km_s": vx_ecef,
            "vy_ecef_km_s": vy_ecef,
            "vz_ecef_km_s": vz_ecef,

            # ECEF spherical
            "r_ecef_km": r_ecef_val,
            "theta_ecef_deg": theta_ecef_val,
            "phi_ecef_deg": phi_ecef_val,

            # Classical elements
            "sma_km": sma_km,
            "ecc": ecc_val,
            "inc_deg": inc_deg,
            "raan_deg": raan_deg,
            "argp_deg": argp_deg,
            "nu_deg": nu_deg,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

# def generate_orbits_dataset(n_orbits=2, orbit_types=("LEO", "MEO", "HEO", "GEO"), time_step=60*u.s, out_csv="orbits_dataset.csv"):
#     """
#     Generate multiple random orbits, propagate each, and store data in a single CSV.
#     """
#     all_dfs = []

#     # Use current time as reference epoch or pick a custom one
#     base_epoch = J2000

#     for i in tqdm(range(n_orbits), desc="Generating orbits", unit="orbit"):
#         chosen_type = np.random.choice(orbit_types)
#         # Generate random orbit
#         orb = generate_random_orbit(base_epoch, orbit_type=chosen_type)
#         orbit_id = f"Orbit_{i+1}"
#         df = propagate_orbit_to_df(
#             orbit_obj=orb,
#             orbit_id=orbit_id,
#             orbit_regime=chosen_type,
#             time_step=time_step,
#         )
#         all_dfs.append(df)

#     final_df = pd.concat(all_dfs, ignore_index=True)
#     final_df.to_csv(out_csv, index=False)
#     print(f"Saved {len(final_df)} rows to {out_csv}")
#     return final_df


def generate_single_orbit(i, orbit_types, time_step, base_epoch):
    """
    Generate and propagate a single orbit.
    
    Args:
        i (int): Orbit index.
        orbit_types (tuple): Tuple of orbit types to choose from.
        time_step (astropy.units.quantity.Quantity): Time step for propagation.
        base_epoch (astropy.time.Time): Base epoch for the orbit.
    
    Returns:
        pd.DataFrame: DataFrame containing the propagated orbit data.
    """
    np.random.seed(i)  # Set unique seed for reproducibility
    chosen_type = np.random.choice(orbit_types)
    orb = generate_random_orbit(base_epoch, orbit_type=chosen_type)
    orbit_id = f"Orbit_{i+1}"
    df = propagate_orbit_to_df(
        orbit_obj=orb,
        orbit_id=orbit_id,
        orbit_regime=chosen_type,
        time_step=time_step,
    )
    return df

def generate_orbits_dataset(n_orbits=2, orbit_types=("LEO", "MEO", "HEO", "GEO"), time_step=60*u.s, out_csv="orbits_dataset.csv", num_workers=None):
    """
    Generate multiple random orbits in parallel, propagate each, and store data in a single CSV.
    
    Args:
        n_orbits (int): Number of orbits to generate.
        orbit_types (tuple): Tuple of orbit types to choose from.
        time_step (astropy.units.quantity.Quantity): Time step for propagation.
        out_csv (str): Output CSV file path.
        num_workers (int, optional): Number of worker processes. Defaults to CPU core count.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame of all generated orbits.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    base_epoch = Time("J2000", scale="tt")
    
    # Create a partial function with fixed arguments
    generate_func = partial(generate_single_orbit, orbit_types=orbit_types, time_step=time_step, base_epoch=base_epoch)
    
    # Use multiprocessing pool to generate orbits in parallel
    with multiprocessing.Pool(num_workers) as pool:
        results = []
        # Process orbits and show progress
        for df in tqdm(pool.imap_unordered(generate_func, range(n_orbits)), total=n_orbits, desc="Generating orbits", unit="orbit"):
            results.append(df)
    
    # Combine all DataFrames
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved {len(final_df)} rows to {out_csv}")
    return final_df


def split_orbits_by_id(df, train_ratio=0.8, val_ratio=0.1):
    unique_ids = df["orbit_id"].unique()
    np.random.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    train_ids = unique_ids[:n_train]
    val_ids   = unique_ids[n_train:n_train+n_val]
    test_ids  = unique_ids[n_train+n_val:]

    df_train = df[df["orbit_id"].isin(train_ids)].copy()
    df_val   = df[df["orbit_id"].isin(val_ids)].copy()
    df_test  = df[df["orbit_id"].isin(test_ids)].copy()

    return df_train, df_val, df_test
