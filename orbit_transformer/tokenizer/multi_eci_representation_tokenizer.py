import numpy as np

from .tokenizer import Tokenizer
from ..constants import MIN_R_MAG, MAX_R_MAG, MAX_V_MAG, KILOMETER


class MultiEciRepresentationTokenizer(Tokenizer):
    def __init__(
        self,
        position_spherical_bins,
        position_cartesian_bins,
        velocity_spherical_bins,
        velocity_cartesian_bins,
        r_min=MIN_R_MAG/ KILOMETER,
        r_max=MAX_R_MAG / KILOMETER,
        v_max=MAX_V_MAG / KILOMETER,
        theta_min=0.0,
        theta_max=180.0,
        phi_min=-180.0,
        phi_max=180.0
    ):
        """
        Tokenizer for multiple representations of position and velocity in ECI frame.

        Parameters:
        - position_spherical_bins: Tuple (r_bins, theta_bins, phi_bins)
        - position_cartesian_bins: Tuple (x_bins, y_bins, z_bins)
        - velocity_spherical_bins: Tuple (vr_bins, vtheta_bins, vphi_bins)
        - velocity_cartesian_bins: Tuple (vx_bins, vy_bins, vz_bins)
        - r_min, r_max: Range for radial distance (km)
        - v_max: Magnitude of maximum velocity (km/s)
        - theta_min, theta_max: Range for polar angle (degrees)
        - phi_min, phi_max: Range for azimuthal angle (degrees)
        """
        self.position_spherical_bins = position_spherical_bins
        self.position_cartesian_bins = position_cartesian_bins
        self.velocity_spherical_bins = velocity_spherical_bins
        self.velocity_cartesian_bins = velocity_cartesian_bins
        self.r_min = r_min
        self.r_max = r_max
        self.v_max = v_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.phi_min = phi_min
        self.phi_max = phi_max

        # Precompute bin edges
        self.r_edges = np.linspace(r_min, r_max, position_spherical_bins[0] + 1)
        self.theta_edges = np.linspace(theta_min, theta_max, position_spherical_bins[1] + 1)
        self.phi_edges = np.linspace(phi_min, phi_max, position_spherical_bins[2] + 1)
        self.x_edges = np.linspace(-r_max, r_max, position_cartesian_bins[0] + 1)
        self.y_edges = np.linspace(-r_max, r_max, position_cartesian_bins[1] + 1)
        self.z_edges = np.linspace(-r_max, r_max, position_cartesian_bins[2] + 1)
        self.vr_edges = np.linspace(-v_max, v_max, velocity_spherical_bins[0] + 1)
        self.vtheta_edges = np.linspace(-v_max, v_max, velocity_spherical_bins[1] + 1)
        self.vphi_edges = np.linspace(-v_max, v_max, velocity_spherical_bins[2] + 1)
        self.vx_edges = np.linspace(-v_max, v_max, velocity_cartesian_bins[0] + 1)
        self.vy_edges = np.linspace(-v_max, v_max, velocity_cartesian_bins[1] + 1)
        self.vz_edges = np.linspace(-v_max, v_max, velocity_cartesian_bins[2] + 1)

    def transform(self, df):
        """
        Add tokenized columns for all representations to the DataFrame.
        Assumes df has columns: x_eci_km, y_eci_km, z_eci_km, vx_eci_km_s, vy_eci_km_s, vz_eci_km_s.
        """
        required_cols = ['x_eci_km', 'y_eci_km', 'z_eci_km', 'vx_eci_km_s', 'vy_eci_km_s', 'vz_eci_km_s']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Compute spherical position
        r = np.sqrt(df['x_eci_km']**2 + df['y_eci_km']**2 + df['z_eci_km']**2)
        theta = np.arccos(df['z_eci_km'] / r) * (180 / np.pi)
        phi = np.arctan2(df['y_eci_km'], df['x_eci_km']) * (180 / np.pi)

        # Tokenize spherical position
        df['eci_r_token'] = np.clip(np.digitize(r, self.r_edges) - 1, 0, self.position_spherical_bins[0] - 1)
        df['eci_theta_token'] = np.clip(np.digitize(theta, self.theta_edges) - 1, 0, self.position_spherical_bins[1] - 1)
        df['eci_phi_token'] = np.clip(np.digitize(phi, self.phi_edges) - 1, 0, self.position_spherical_bins[2] - 1)

        # Tokenize Cartesian position
        df['eci_x_token'] = np.clip(np.digitize(df['x_eci_km'], self.x_edges) - 1, 0, self.position_cartesian_bins[0] - 1)
        df['eci_y_token'] = np.clip(np.digitize(df['y_eci_km'], self.y_edges) - 1, 0, self.position_cartesian_bins[1] - 1)
        df['eci_z_token'] = np.clip(np.digitize(df['z_eci_km'], self.z_edges) - 1, 0, self.position_cartesian_bins[2] - 1)

        # Compute spherical velocity components
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)

        # Spherical unit vectors
        e_r = np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]).T
        e_theta = np.array([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta]).T
        e_phi = np.array([-sin_phi, cos_phi, np.zeros_like(phi)]).T

        # Velocity vector
        v = np.array([df['vx_eci_km_s'], df['vy_eci_km_s'], df['vz_eci_km_s']]).T

        # Project velocity
        vr = np.sum(v * e_r, axis=1)
        vtheta = np.sum(v * e_theta, axis=1)
        vphi = np.sum(v * e_phi, axis=1)

        # Tokenize spherical velocity
        df['eci_vr_token'] = np.clip(np.digitize(vr, self.vr_edges) - 1, 0, self.velocity_spherical_bins[0] - 1)
        df['eci_vtheta_token'] = np.clip(np.digitize(vtheta, self.vtheta_edges) - 1, 0, self.velocity_spherical_bins[1] - 1)
        df['eci_vphi_token'] = np.clip(np.digitize(vphi, self.vphi_edges) - 1, 0, self.velocity_spherical_bins[2] - 1)

        # Tokenize Cartesian velocity
        df['eci_vx_token'] = np.clip(np.digitize(df['vx_eci_km_s'], self.vx_edges) - 1, 0, self.velocity_cartesian_bins[0] - 1)
        df['eci_vy_token'] = np.clip(np.digitize(df['vy_eci_km_s'], self.vy_edges) - 1, 0, self.velocity_cartesian_bins[1] - 1)
        df['eci_vz_token'] = np.clip(np.digitize(df['vz_eci_km_s'], self.vz_edges) - 1, 0, self.velocity_cartesian_bins[2] - 1)

        return df

    def to_dict(self):
        """Serialize parameters to a dictionary."""
        return {
            "class_name": "MultiRepresentationTokenizer",
            "position_spherical_bins": self.position_spherical_bins,
            "position_cartesian_bins": self.position_cartesian_bins,
            "velocity_spherical_bins": self.velocity_spherical_bins,
            "velocity_cartesian_bins": self.velocity_cartesian_bins,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "v_max": self.v_max,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "phi_min": self.phi_min,
            "phi_max": self.phi_max
        }
