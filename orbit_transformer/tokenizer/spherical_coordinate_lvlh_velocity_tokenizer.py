import numpy as np

from ..constants import MIN_R_MAG, MAX_R_MAG, MAX_V_MAG, KILOMETER
from .tokenizer import Tokenizer


class SphericalCoordinateLVLHVelocityTokenizer(Tokenizer):
    def __init__(
        self,
        r_bins=128,
        theta_bins=128,
        phi_bins=128,
        vx_bins=128,
        vy_bins=128,
        vz_bins=128,
        r_min=MIN_R_MAG / KILOMETER,
        r_max=MAX_R_MAG / KILOMETER,
        theta_min=0.0,
        theta_max=180.0,
        phi_min=-180.0,
        phi_max=180.0,
        v_max=MAX_V_MAG / KILOMETER
    ):
        """
        Tokenizer for spherical position (r, theta, phi) and LVLH velocity (vx, vy, vz).
        
        Parameters:
        - r_bins, theta_bins, phi_bins: Number of bins for position components.
        - vx_bins, vy_bins, vz_bins: Number of bins for velocity components.
        - r_min, r_max: Radial distance range (km).
        - theta_min, theta_max: Polar angle range (degrees).
        - phi_min, phi_max: Azimuthal angle range (degrees).
        - v_max: Velocity magnitude range (km/s).
        """
        self.r_bins = r_bins
        self.theta_bins = theta_bins
        self.phi_bins = phi_bins
        self.vx_bins = vx_bins
        self.vy_bins = vy_bins
        self.vz_bins = vz_bins
        self.r_min = r_min
        self.r_max = r_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.v_min = -v_max
        self.v_max = v_max

        # Precompute bin edges
        self.r_edges = np.linspace(r_min, r_max, r_bins + 1)
        self.theta_edges = np.linspace(theta_min, theta_max, theta_bins + 1)
        self.phi_edges = np.linspace(phi_min, phi_max, phi_bins + 1)
        self.vx_edges = np.linspace(-v_max, v_max, vx_bins + 1)
        self.vy_edges = np.linspace(-v_max, v_max, vy_bins + 1)
        self.vz_edges = np.linspace(-v_max, v_max, vz_bins + 1)

    def eci_to_lvlh_velocity(self, r_eci_km, v_eci_km_s):
        """
        Convert ECI velocity to LVLH frame given position and velocity.
        Returns (vx, vy, vz) in km/s.
        """
        # LVLH basis vectors
        z_lvlh = -r_eci_km / np.linalg.norm(r_eci_km)  # Radial (toward Earth)
        y_lvlh = np.cross(r_eci_km, v_eci_km_s) / np.linalg.norm(np.cross(r_eci_km, v_eci_km_s))  # Normal
        x_lvlh = np.cross(y_lvlh, z_lvlh)  # Along-track
        
        # Rotation matrix from ECI to LVLH
        rot_matrix = np.column_stack((x_lvlh, y_lvlh, z_lvlh))
        
        # Project velocity into LVLH frame
        v_lvlh = rot_matrix.T @ v_eci_km_s
        return v_lvlh[0], v_lvlh[1], v_lvlh[2]

    def transform(self, df, coordinate_prefix="eci"):
        """
        Transform DataFrame by adding tokenized spherical position and LVLH velocity.
        Assumes df has columns: x_eci_km, y_eci_km, z_eci_km, vx_eci_km_s, vy_eci_km_s, vz_eci_km_s.
        """
        r_col = f"r_{coordinate_prefix}_km"
        theta_col = f"theta_{coordinate_prefix}_deg"
        phi_col = f"phi_{coordinate_prefix}_deg"
        x_col = f"x_{coordinate_prefix}_km"
        y_col = f"y_{coordinate_prefix}_km"
        z_col = f"z_{coordinate_prefix}_km"
        vx_col = f"vx_lvlh_km_s"
        vy_col = f"vy_lvlh_km_s"
        vz_col = f"vz_lvlh_km_s"

        required_cols = [r_col, theta_col, phi_col, x_col, y_col, z_col, vx_col, vy_col, vz_col]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Column {c} not found in DataFrame.")

        # Tokenize position (r, theta, phi)
        r_tokens = np.digitize(df[r_col].values, self.r_edges) - 1
        theta_tokens = np.digitize(df[theta_col].values, self.theta_edges) - 1
        phi_tokens = np.digitize(df[phi_col].values, self.phi_edges) - 1

        r_tokens = np.clip(r_tokens, 0, self.r_bins - 1)
        theta_tokens = np.clip(theta_tokens, 0, self.theta_bins - 1)
        phi_tokens = np.clip(phi_tokens, 0, self.phi_bins - 1)

        # Compute LVLH velocities
        v_lvlh_x = []
        v_lvlh_y = []
        v_lvlh_z = []
        for i in range(len(df)):
            r_eci = np.array([df[x_col].iloc[i], df[y_col].iloc[i], df[z_col].iloc[i]])
            v_eci = np.array([df[vx_col].iloc[i], df[vy_col].iloc[i], df[vz_col].iloc[i]])
            vx, vy, vz = self.eci_to_lvlh_velocity(r_eci, v_eci)
            v_lvlh_x.append(vx)
            v_lvlh_y.append(vy)
            v_lvlh_z.append(vz)

        v_lvlh_x = np.array(v_lvlh_x)
        v_lvlh_y = np.array(v_lvlh_y)
        v_lvlh_z = np.array(v_lvlh_z)

        # Tokenize velocities
        vx_tokens = np.digitize(v_lvlh_x, self.vx_edges) - 1
        vy_tokens = np.digitize(v_lvlh_y, self.vy_edges) - 1
        vz_tokens = np.digitize(v_lvlh_z, self.vz_edges) - 1

        vx_tokens = np.clip(vx_tokens, 0, self.vx_bins - 1)
        vy_tokens = np.clip(vy_tokens, 0, self.vy_bins - 1)
        vz_tokens = np.clip(vz_tokens, 0, self.vz_bins - 1)

        # Add tokens to DataFrame
        df[f"{coordinate_prefix}_r_token"] = r_tokens
        df[f"{coordinate_prefix}_theta_token"] = theta_tokens
        df[f"{coordinate_prefix}_phi_token"] = phi_tokens
        df[f"lvlh_vx_token"] = vx_tokens
        df[f"lvlh_vy_token"] = vy_tokens
        df[f"lvlh_vz_token"] = vz_tokens

        # Optionally store raw LVLH velocities for reference
        df[f"{coordinate_prefix}_vx_lvlh_km_s"] = v_lvlh_x
        df[f"{coordinate_prefix}_vy_lvlh_km_s"] = v_lvlh_y
        df[f"{coordinate_prefix}_vz_lvlh_km_s"] = v_lvlh_z

        return df

    def get_bin_centers(self):
        """Return bin centers for all six components."""
        r_centers = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        theta_centers = 0.5 * (self.theta_edges[:-1] + self.theta_edges[1:])
        phi_centers = 0.5 * (self.phi_edges[:-1] + self.phi_edges[1:])
        vx_centers = 0.5 * (self.vx_edges[:-1] + self.vx_edges[1:])
        vy_centers = 0.5 * (self.vy_edges[:-1] + self.vy_edges[1:])
        vz_centers = 0.5 * (self.vz_edges[:-1] + self.vz_edges[1:])
        return r_centers, theta_centers, phi_centers, vx_centers, vy_centers, vz_centers

    def to_dict(self):
        """Serialize parameters to a dictionary."""
        return {
            "class_name": "SphericalLVLHVelocityTokenizer",
            "r_bins": self.r_bins,
            "theta_bins": self.theta_bins,
            "phi_bins": self.phi_bins,
            "vx_bins": self.vx_bins,
            "vy_bins": self.vy_bins,
            "vz_bins": self.vz_bins,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "phi_min": self.phi_min,
            "phi_max": self.phi_max,
            "v_min": self.v_min,
            "v_max": self.v_max
        }
    