import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.iod import izzo

from ..constants import (
    KILOMETER, MINUTE, HOUR, DAY, R_EARTH,
    MIN_R_MAG, MAX_R_MAG, MIN_V_MAG, MAX_V_MAG,
    MIN_THRUST, MAX_THRUST, MIN_TIME_STEP_SEC, MAX_TIME_STEP_SEC,
)


def create_orbit_from_sv_meters(r_m, v_mps):
    # poliastro wants float64 for this operation!!
    r = r_m.astype(np.float64) * u.m
    v = v_mps.astype(np.float64) * u.m / u.s
    return Orbit.from_vectors(Earth, r, v)


def create_orbit_from_orbital_elements_meters(inc, a, ecc, raan, argp, nu):
    # poliastro wants float64 for this operation!!
    inc = inc.astype(np.float64) * u.deg
    a = a.astype(np.float64) * u.m
    ecc = ecc.astype(np.float64) * u.one
    raan = raan.astype(np.float64) * u.deg
    argp = argp.astype(np.float64) * u.deg
    nu = nu.astype(np.float64) * u.deg
    return Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)


def get_rv_from_orbit(orbit):
    position_eci = orbit.r.to(u.m).value        # Convert position to meters
    velocity_eci = orbit.v.to(u.m / u.s).value  # Convert velocity to meters per second
    return position_eci, velocity_eci


def get_orbital_elements_from_orbit(orbit):
    inc = orbit.inc.to(u.deg).value
    a = orbit.a.to(u.m).value
    ecc = orbit.ecc.value
    raan = orbit.raan.to(u.deg).value
    argp = orbit.argp.to(u.deg).value
    nu = orbit.nu.to(u.deg).value
    nu_normalized = nu % 360
    return inc, a, ecc, raan, argp, nu_normalized


# TODO
def get_expanded_orbital_elements_from_orbit(orbit):
    inc, a, ecc, raan, argp, nu = get_orbital_elements_from_orbit(orbit)
    p = orbit.p.to(u.km).value
    raise NotImplemented('need to update before use')


def apply_speed_change(orbit, speed_change):
    # Calculate the burn direction (along the current velocity vector)
    burn_dir_eci = orbit.v.value / np.linalg.norm(orbit.v.value)
    # Create a Maneuver object for the burn
    dv = burn_dir_eci * speed_change * u.m / u.s
    impulsive_burn = Maneuver.impulse(dv)
    # Apply the maneuver to the orbit
    new_orbit = orbit.apply_maneuver(impulsive_burn)
    return new_orbit


def apply_burn_eci(orbit, burn_dir_eci, burn_mag):
    burn_unit_eci = burn_dir_eci / np.linalg.norm(burn_dir_eci)  # Normalize the direction
    burn_eci = burn_unit_eci * burn_mag
    # Create a Maneuver object for the burn
    dv = burn_eci * u.m / u.s  # Convert burn vector to Astropy Quantity
    impulsive_burn = Maneuver.impulse(dv)
    # Apply the maneuver to the orbit
    new_orbit = orbit.apply_maneuver(impulsive_burn)
    return new_orbit


def orbit_to_eci_unit_vectors(orbit):
    r = orbit.r.to(u.km).value
    r_unit = r / np.linalg.norm(r)
    v = orbit.v.to(u.km / u.s).value
    v_unit = v / np.linalg.norm(v)
    return r_unit, v_unit


def ecef_to_eci(ecef_coords, obstime):
    itrs = ITRS(CartesianRepresentation(ecef_coords * u.one), obstime=obstime)
    gcrs = itrs.transform_to(GCRS(obstime=obstime))
    return gcrs.cartesian.xyz


def lvlh_to_eci(orbit, lvlh_vector):
    r = orbit.r.to(u.km).value  # Position vector in ECI frame
    v = orbit.v.to(u.km / u.s).value  # Velocity vector in ECI frame
    # Compute LVLH basis vectors
    z_lvlh = -r / np.linalg.norm(r)  # Towards Earth
    y_lvlh = np.cross(r, v) / np.linalg.norm(np.cross(r, v))  # Orbit normal
    x_lvlh = np.cross(y_lvlh, z_lvlh)  # Completes right-handed system
    rot_matrix = np.column_stack((x_lvlh, y_lvlh, z_lvlh))
    # Convert LVLH maneuver direction to ECI
    eci_vector = rot_matrix @ lvlh_vector
    return eci_vector


def ipop_to_eci(orbit, in_plane_angle, out_plane_angle):
    r = orbit.r.to(u.km).value  # Position vector in ECI frame
    v = orbit.v.to(u.km / u.s).value  # Velocity vector in ECI frame
    # Compute the orbital plane's normal vector
    h = np.cross(r, v)
    h_unit = h / np.linalg.norm(h)
    # Compute the radial unit vector
    r_unit = r / np.linalg.norm(r)
    # Compute the in-plane perpendicular unit vector
    p_unit = np.cross(h_unit, r_unit)
    # Convert angles to radians
    in_plane_rad = np.radians(in_plane_angle)
    out_plane_rad = np.radians(out_plane_angle)
    # Compute the burn vector in the orbital plane reference frame
    burn_vector = np.array([
        np.cos(in_plane_rad) * np.cos(out_plane_rad),
        np.sin(in_plane_rad) * np.cos(out_plane_rad),
        np.sin(out_plane_rad)
    ])
    # Transform the burn vector to ECI
    burn_eci = (burn_vector[0] * r_unit +
                burn_vector[1] * p_unit +
                burn_vector[2] * h_unit)
    return burn_eci


def get_distance_between_orbits(orbit1, orbit2):
    position1, _ = get_rv_from_orbit(orbit1)
    position2, _ = get_rv_from_orbit(orbit2)
    delta = position2 - position1
    distance = np.linalg.norm(delta)
    return distance


def offset_orbit(orbit, max_offset_distance):
    # Get the current position vector (r) of the orbit
    r = orbit.r.to(u.m).value
    # Generate a random offset vector within the specified range
    offset = np.random.uniform(-max_offset_distance, max_offset_distance, size=3)  # Offset in m
    # Apply the offset to the current position vector
    new_r = r + offset
    # Create a new orbit with the offset position vector
    new_orbit = Orbit.from_vectors(Earth, new_r * u.m, orbit.v)
    return new_orbit


def solve_lamberts_problem(initial_orbit, target_orbit, tof_min=MINUTE,
                           tof_max=2 * HOUR, step_size=10 * MINUTE, revs=0):

    v0 = initial_orbit.v.to(u.m / u.s)

    best_dv = np.inf
    best_v1 = None
    best_tof = None
    for tof in np.arange(tof_min, tof_max + step_size, step_size):

        target_orbit_at_intercept = target_orbit.propagate(tof * u.s)

        try:
            v1, v2 = izzo.lambert(Earth.k, initial_orbit.r, target_orbit_at_intercept.r, tof * u.s, M=revs)
        except RuntimeError:
            print("failed to converge, skipping")
            continue

        v1 = v1.to(u.m / u.s)

        # Calculate Δv required for this solution
        dv1 = np.linalg.norm((v1 - v0).value)

        total_dv = dv1

        # Check if this solution is better than the current best
        if total_dv < best_dv:
            best_dv = total_dv
            best_v1 = v1
            best_tof = tof

    # Compute burn direction and magnitude based on best solution
    burn_direction = (best_v1 - v0).value
    burn_direction /= np.linalg.norm(burn_direction)  # Normalize

    if any(np.isnan(burn_direction)):
        burn_direction = np.array([0, 0, 0], dtype=burn_direction.dtype)

    burn_magnitude = best_dv  # Δv magnitude

    return burn_direction, burn_magnitude, best_tof


def generate_random_orbit(epoch, orbit_type="Random"):

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

    # Randomly generate altitudes
    periapsis_alt = np.random.uniform(periapsis_alt_km_min, periapsis_alt_km_max)
    apoapsis_alt = np.random.uniform(apoapsis_alt_km_min, apoapsis_alt_km_max)

    while apoapsis_alt < periapsis_alt:
        apoapsis_alt = np.random.uniform(apoapsis_alt_km_min, apoapsis_alt_km_max)

    # Calculate periapsis and apoapsis distances from Earth's center
    periapsis = 6371 + periapsis_alt
    apoapsis = 6371 + apoapsis_alt

    # Calculate semi-major axis and eccentricity
    a = (periapsis + apoapsis) / 2 * u.km
    ecc = (apoapsis - periapsis) / (apoapsis + periapsis) * u.one

    # Randomly generate other orbital elements
    inc = np.random.uniform(0, 180) * u.deg
    raan = np.random.uniform(0, 360) * u.deg
    argp = np.random.uniform(0, 360) * u.deg
    nu = np.random.uniform(0, 360) * u.deg

    # Create and return the orbit
    return Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch=epoch)


def generate_ephemeris_dict(start_epoch, duration_sec, orbit_dict, step_size=MINUTE):
    ephemeris = {}
    start = 0
    end = duration_sec
    for time_delta_sec in np.arange(start, end + step_size, step_size):
        # Find the most recent orbit before the current time
        recent_orbit_time = max(t for t in orbit_dict if t <= time_delta_sec)
        recent_orbit = orbit_dict[recent_orbit_time]
        cur_time = start_epoch + time_delta_sec * u.s
        prop_time = cur_time - recent_orbit.epoch
        # Propagate the orbit to the current time
        cur_orbit = recent_orbit.propagate(prop_time)
        # Grab position and store
        position_eci, _ = get_rv_from_orbit(cur_orbit)
        ephemeris[time_delta_sec] = position_eci
    return ephemeris


def dump_ephemeris_to_csv(start_epoch, duration_sec, orbit_dict, path):
    lines = list()
    header = '{time},{rx},{ry},{rz}\n'
    lines.append(header)
    ephem_dict = generate_ephemeris_dict(start_epoch, duration_sec, orbit_dict)
    for time, position_eci in ephem_dict.items():
        rx, ry, rz = position_eci
        line = f'{time},{rx},{ry},{rz}\n'
        lines.append(line)
    with open(path, 'w') as f:
        f.writelines(lines)


def dump_orbits_to_csv(path, orbit_dict):
    lines = list()
    header = '{time},{rx},{ry},{rz},{vx},{vy},{vz}\n'
    lines.append(header)
    for time, orbit in orbit_dict.items():
        position_eci, velocity_eci = get_rv_from_orbit(orbit)
        rx, ry, rz = position_eci
        vx, vy, vz = velocity_eci
        line = f'{time},{rx},{ry},{rz},{vx},{vy},{vz}\n'
        lines.append(line)
    with open(path, 'w') as f:
        f.writelines(lines)
