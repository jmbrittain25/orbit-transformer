import numpy as np

from ..constants import MIN_R_MAG, MAX_R_MAG, KILOMETER
from .tokenizer import Tokenizer


class SphericalCoordinateTokenizer(Tokenizer):

    def __init__(
        self,
        r_bins=200,
        theta_bins=180,
        phi_bins=360,
        r_min=MIN_R_MAG / KILOMETER,
        r_max=MAX_R_MAG / KILOMETER,
        theta_min=0.0,
        theta_max=180.0,
        phi_min=-180.0,
        phi_max=180.0,
        composite_tokens=False
    ):
        self.r_bins = r_bins
        self.theta_bins = theta_bins
        self.phi_bins = phi_bins
        self.r_min = r_min
        self.r_max = r_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.composite_tokens = composite_tokens

        # Precompute bin edges
        self.r_edges     = np.linspace(r_min, r_max, r_bins + 1)
        self.theta_edges = np.linspace(theta_min, theta_max, theta_bins + 1)
        self.phi_edges   = np.linspace(phi_min, phi_max, phi_bins + 1)

        # If composite, we create an ID space for the triple
        # total_vocabulary_size = r_bins * theta_bins * phi_bins
        if composite_tokens:
            self.total_vocab_size = self.r_bins * self.theta_bins * self.phi_bins
        else:
            self.total_vocab_size = None  # Not strictly necessary if we keep them separate

    def transform(self, df, coordinate_prefix="eci"):
        r_col = f"r_{coordinate_prefix}_km"
        theta_col = f"theta_{coordinate_prefix}_deg"
        phi_col = f"phi_{coordinate_prefix}_deg"

        for c in [r_col, theta_col, phi_col]:
            if c not in df.columns:
                raise ValueError(f"Column {c} not found in DataFrame. Check data columns or prefix.")

        # Digitize each dimension
        r_tokens     = np.digitize(df[r_col].values, self.r_edges) - 1
        theta_tokens = np.digitize(df[theta_col].values, self.theta_edges) - 1
        phi_tokens   = np.digitize(df[phi_col].values, self.phi_edges) - 1

        # Clip out-of-bounds
        r_tokens     = np.clip(r_tokens, 0, self.r_bins - 1)
        theta_tokens = np.clip(theta_tokens, 0, self.theta_bins - 1)
        phi_tokens   = np.clip(phi_tokens, 0, self.phi_bins - 1)

        if not self.composite_tokens:
            # Add separate columns
            df[f"{coordinate_prefix}_r_token"] = r_tokens
            df[f"{coordinate_prefix}_theta_token"] = theta_tokens
            df[f"{coordinate_prefix}_phi_token"] = phi_tokens
        else:
            # Create a single composite token ID:
            # ID = r_bin + theta_bin * r_bins + phi_bin * (r_bins * theta_bins)
            # This is basically enumerating the 3D grid in row-major order
            composite = (r_tokens 
                         + theta_tokens * self.r_bins 
                         + phi_tokens  * (self.r_bins * self.theta_bins))
            df[f"{coordinate_prefix}_composite_token"] = composite

        return df

    def get_bin_centers(self):
        """
        Return the 1D arrays of bin centers for r, theta, phi.
        """
        r_centers     = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        theta_centers = 0.5 * (self.theta_edges[:-1] + self.theta_edges[1:])
        phi_centers   = 0.5 * (self.phi_edges[:-1] + self.phi_edges[1:])
        return r_centers, theta_centers, phi_centers

    def visualize_all_bins_3d(self, coordinate_prefix="eci", max_points=50_000):
        """
        Plot ALL possible bin centers in 3D space. This shows
        the entire discretized domain for (r, theta, phi).
        
        If the total number of bins (r_bins * theta_bins * phi_bins) 
        exceeds max_points, we sample for easier plotting.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # ensures 3D support

        r_centers, theta_centers, phi_centers = self.get_bin_centers()

        # Create the full meshgrid of bin centers
        # Shapes: (r_bins, ), (theta_bins, ), (phi_bins, )
        # We want to produce all combinations: size = r_bins * theta_bins * phi_bins
        R, TH, PH = np.meshgrid(r_centers, theta_centers, phi_centers, indexing='ij')

        # Flatten
        Rf  = R.ravel()
        THf = TH.ravel()
        PHf = PH.ravel()

        total_bin_points = Rf.shape[0]
        if total_bin_points > max_points:
            # Randomly sample
            idx = np.random.choice(total_bin_points, size=max_points, replace=False)
            Rf  = Rf[idx]
            THf = THf[idx]
            PHf = PHf[idx]

        # Convert spherical -> cartesian for plotting
        x_vals, y_vals, z_vals = spherical_to_cartesian(Rf, THf, PHf)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_vals, y_vals, z_vals, s=1, alpha=0.3, c='blue')

        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.set_title(f"All Bin Centers for {coordinate_prefix.upper()} (r,theta,phi)")

        # Attempt equal aspect ratio
        max_range = np.array([x_vals.max()-x_vals.min(),
                              y_vals.max()-y_vals.min(),
                              z_vals.max()-z_vals.min()]).max()
        mid_x = (x_vals.max()+x_vals.min()) * 0.5
        mid_y = (y_vals.max()+y_vals.min()) * 0.5
        mid_z = (z_vals.max()+z_vals.min()) * 0.5
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        plt.show()

    def to_dict(self):
        return {
            "class_name": "SphericalCoordinateTokenizer",
            "r_bins": self.r_bins,
            "theta_bins": self.theta_bins,
            "phi_bins": self.phi_bins,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "phi_min": self.phi_min,
            "phi_max": self.phi_max,
            "composite_tokens": self.composite_tokens
        }


# Utility: spherical -> cartesian
def spherical_to_cartesian(r, theta_deg, phi_deg):
    """
    Convert spherical (r, theta, phi) to Cartesian (x, y, z).
    - r in km
    - theta in [0..180], from +Z axis
    - phi in [-180..180], in XY plane
    """
    theta = np.radians(theta_deg)
    phi   = np.radians(phi_deg)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z
