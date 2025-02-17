import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


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

        # If token_col is a string => single token dimension (composite)
        # If it's a list => multiple columns. We'll handle both cases.
        self.token_col = token_col if isinstance(token_col, list) else [token_col]

        # Build examples
        for orbit_id, group_df in self.groups:
            # Sort by time to ensure correct temporal order
            group_df = group_df.sort_values(by=self.time_col)

            # For each orbit, gather the token arrays
            # shape = (num_timesteps, num_token_dims)
            tokens_array = group_df[self.token_col].values
            if len(self.token_col) > 1:
                # If multiple columns, tokens_array is shape (N, len(token_col))
                pass
            else:
                # If single column, reshape for consistency => (N, 1)
                tokens_array = tokens_array.reshape(-1, 1)

            total_len = tokens_array.shape[0]
            seq_len = input_length + output_length

            max_start = total_len - seq_len
            if max_start < 0:
                # Not enough tokens in this orbit to form a single sequence
                continue

            # Create sliding windows
            for start_idx in range(0, max_start+1, stride):
                end_idx = start_idx + input_length
                out_end_idx = end_idx + output_length
                # We'll store indices + orbit_id, so we can retrieve on __getitem__
                self.examples.append((orbit_id, start_idx, end_idx, out_end_idx))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # Retrieve the orbit + indices
        orbit_id, start_idx, end_idx, out_end_idx = self.examples[index]
        # Get that orbit's subset
        group_df = self.groups.get_group(orbit_id).sort_values(by=self.time_col)

        # Convert to array
        tokens_array = group_df[self.token_col].values
        if len(self.token_col) > 1:
            # shape (N, len(token_col))
            pass
        else:
            # shape (N,) => reshape to (N,1)
            tokens_array = tokens_array.reshape(-1, 1)

        # Slice
        input_seq = tokens_array[start_idx:end_idx]
        output_seq = tokens_array[end_idx:out_end_idx]

        # Convert to torch tensor
        # If composite tokens => integer IDs => dtype=torch.long
        # If multiple tokens => you might keep them as long or some other dtype
        input_seq_tensor = torch.tensor(input_seq, dtype=torch.long)
        output_seq_tensor = torch.tensor(output_seq, dtype=torch.long)

        # Return a dict or a (input, output) tuple
        return {
            "input": input_seq_tensor.squeeze(-1),   # shape (input_length,) if composite
            "output": output_seq_tensor.squeeze(-1)  # shape (output_length,)
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
