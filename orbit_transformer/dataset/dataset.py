import pandas as pd
import torch
from torch.utils.data import Dataset


import pandas as pd
import torch
from torch.utils.data import Dataset

class OrbitTokenDataset(Dataset):
    def __init__(
        self,
        csv_path,
        orbit_id_col="orbit_id",
        time_col="time_s",
        input_length=32,
        output_length=1,
        stride=1
    ):
        super().__init__()
        self.csv_path = csv_path
        self.orbit_id_col = orbit_id_col
        self.time_col = time_col
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride
        self.token_col = ["eci_r_token", "eci_theta_token", "eci_phi_token"]

        # Load data
        df = pd.read_csv(csv_path)

        # Group by orbit
        self.groups = df.groupby(orbit_id_col, sort=False)

        # Store examples as (orbit_id, start_idx, end_idx, out_end_idx)
        self.examples = []

        # Build examples
        for orbit_id, group_df in self.groups:
            group_df = group_df.sort_values(by=self.time_col)
            tokens_array = group_df[self.token_col].values  # Shape: (num_timesteps, 3)

            total_len = tokens_array.shape[0]
            seq_len = input_length + output_length

            max_start = total_len - seq_len
            if max_start < 0:
                continue

            for start_idx in range(0, max_start + 1, stride):
                end_idx = start_idx + input_length
                out_end_idx = end_idx + output_length
                self.examples.append((orbit_id, start_idx, end_idx, out_end_idx))



class OrbitTokenDataset(Dataset):
    """
    A PyTorch Dataset that:
      - Loads tokenized orbital data from a CSV.
      - Groups rows by orbit_id.
      - For each orbit, creates sliding-window sequences.
        * input_length tokens are "the past context"
        * output_length tokens are "the future to predict"

    Parameters
    ----------
    csv_path : str
        Path to the tokenized CSV file.
    orbit_id_col : str
        Name of the column that uniquely identifies each orbit (e.g., "orbit_id").
    token_col : str or list of str
        Name(s) of the column(s) containing the token(s). 
        For composite tokens, might be "eci_composite_token".
        For separate tokens, you might pass a list ["eci_r_token", "eci_theta_token", "eci_phi_token"].
    time_col : str
        Name of the column specifying time order (e.g., "time_s"). Used for sorting.
    input_length : int
        Number of past tokens in each sliding window.
    output_length : int
        Number of future tokens to predict (e.g., 1 for next-step, or more).
    stride : int
        Step size of the sliding window.
    """

    def __init__(
        self,
        csv_path,
        orbit_id_col="orbit_id",
        token_col=["eci_r_token", "eci_theta_token", "eci_phi_token"],
        time_col="time_s",
        input_length=32,
        output_length=1,
        stride=1
    ):
        super().__init__()
        self.csv_path = csv_path
        self.orbit_id_col = orbit_id_col
        self.token_col = token_col if isinstance(token_col, list) else [token_col]
        self.time_col = time_col
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride

        # Load data
        df = pd.read_csv(csv_path)

        # Group by orbit
        self.groups = df.groupby(orbit_id_col, sort=False)

        # We'll store a list of (orbit_id, start_idx, end_idx, out_end_idx)
        # each describing one training example in terms of array indices.
        self.examples = []

        # Build examples
        for orbit_id, group_df in self.groups:
            group_df = group_df.sort_values(by=self.time_col)
            tokens_array = group_df[self.token_col].values  # Shape: (num_timesteps, 3)

            total_len = tokens_array.shape[0]
            seq_len = input_length + output_length

            max_start = total_len - seq_len
            if max_start < 0:
                continue

            for start_idx in range(0, max_start + 1, stride):
                end_idx = start_idx + input_length
                out_end_idx = end_idx + output_length
                self.examples.append((orbit_id, start_idx, end_idx, out_end_idx))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        orbit_id, start_idx, end_idx, out_end_idx = self.examples[index]
        group_df = self.groups.get_group(orbit_id).sort_values(by=self.time_col)

        tokens_array = group_df[self.token_col].values  # Shape: (N, 3)
        input_seq = tokens_array[start_idx:end_idx]     # Shape: (input_length, 3)
        output_seq = tokens_array[end_idx:out_end_idx]  # Shape: (output_length, 3)

        input_seq_tensor = torch.tensor(input_seq, dtype=torch.long)
        output_seq_tensor = torch.tensor(output_seq, dtype=torch.long)

        return {
            "input": input_seq_tensor,   # Shape: (input_length, 3)
            "output": output_seq_tensor  # Shape: (output_length, 3)
        }

    def to_dict(self):
        return {
            "csv_path": self.csv_path,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "stride": self.stride,
            "token_col": self.token_col,
            "time_col": self.time_col,
            "orbit_id_col": self.orbit_id_col,
            "groups": self.groups,
            "examples": self.examples
        }


def collate_fn(batch):
    """
    Custom collate function if needed. 
    For simple (input, output) pairs of uniform length, default_collate works fine.
    But if you want to handle variable-length sequences, you could pad here.
    """
    inputs = [ex["input"] for ex in batch]
    outputs = [ex["output"] for ex in batch]

    # Stack them into tensors of shape (batch_size, seq_length)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)

    return inputs, outputs
