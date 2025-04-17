import torch
from torch.utils.data import Dataset


class OrbitTokenDataset(Dataset):
    def __init__(
        self,
        df,
        orbit_id_col="orbit_id",
        token_cols=None,
        time_col="time_s",
        input_length=32,
        output_length=1,
        stride=1
    ):
        super().__init__()

        if token_cols is None:
            token_cols = ["eci_r_token", "eci_theta_token", "eci_phi_token"]

        self.orbit_id_col = orbit_id_col
        self.time_col = time_col
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride
        self.token_cols = token_cols

        # Group by orbit
        self.groups = df.groupby(orbit_id_col, sort=False)

        # Store examples as (orbit_id, start_idx, end_idx, out_end_idx)
        self.examples = []

        # Build examples
        for orbit_id, group_df in self.groups:
            group_df = group_df.sort_values(by=self.time_col)
            tokens_array = group_df[self.token_cols].values  # Shape: (num_timesteps, num_components)

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

        tokens_array = group_df[self.token_cols].values  # Shape: (N, 3)
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
            "input_length": self.input_length,
            "output_length": self.output_length,
            "stride": self.stride,
            "token_cols": self.token_cols,
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
