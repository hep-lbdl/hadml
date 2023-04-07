import itertools
import pickle
from functools import reduce
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

GAN_INPUT_DATA_TYPE = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]


def shuffle(array: np.ndarray):
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence

    np_rs = RandomState(MT19937(SeedSequence(123456789)))
    np_rs.shuffle(array)


def read_dataframe(filename, sep=",", engine=None):
    if type(filename) == list:
        print(filename)
        df_list = [
            pd.read_csv(f, sep=sep, header=None, names=None, engine=engine)
            for f in filename
        ]
        df = pd.concat(df_list, ignore_index=True)
        filename = filename[0]
    else:
        df = pd.read_csv(filename, sep=sep, header=None, names=None, engine=engine)
    return df


def split_to_float(df, sep=","):
    out = df
    if type(df.iloc[0]) == str:
        out = df.str.split(sep, expand=True).astype(np.float32)
    return out


def xyz2hep(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    theta = np.arccos(pz / p)
    eta = -np.log(np.tan(0.5 * theta))
    return pt, eta, phi


def calculate_mass(lorentz_vector: np.ndarray) -> float:
    """To calculate the invariant mass of a particle given its 4-vector.

    Args:
        lorentz_vector: 4 vector [E, px, py, pz]

    Returns:
        invariant mass
    """
    sum_p2 = sum([lorentz_vector[idx] ** 2 for idx in range(1, 4)])
    return np.sqrt(lorentz_vector[0] ** 2 - sum_p2)


def create_boost_fn(cluster_4vec: np.ndarray):
    mass = calculate_mass(cluster_4vec)
    E0, p0 = cluster_4vec[0], cluster_4vec[1:]
    gamma = E0 / mass

    velocity = p0 / gamma / mass
    v_mag = np.sqrt(sum([velocity[idx] ** 2 for idx in range(3)]))
    n = velocity / v_mag

    def boost_fn(lab_4vec: np.ndarray):
        """4vector [E, px, py, pz] in lab frame"""
        E = lab_4vec[0]
        p = lab_4vec[1:]
        n_dot_p = np.sum((n * p))
        E_prime = gamma * (E - v_mag * n_dot_p)
        P_prime = p + (gamma - 1) * n_dot_p * n - gamma * E * v_mag * n
        return np.array([E_prime] + P_prime.tolist())

    def inv_boost_fn(boost_4vec: np.ndarray):
        """4vecot [E, px, py, pz] in boost frame (aka cluster frame)"""
        E_prime = boost_4vec[0]
        P_prime = boost_4vec[1:]
        n_dot_p = np.sum((n * P_prime))
        E = gamma * (E_prime + v_mag * n_dot_p)
        p = P_prime + (gamma - 1) * n_dot_p * n + gamma * E_prime * v_mag * n
        return np.array([E] + p.tolist())

    return boost_fn, inv_boost_fn


def boost(a_row: np.ndarray):
    """boost all particles to the rest frame of the first particle in the list"""

    assert a_row.shape[0] % 4 == 0, "a_row should be a 4-vector"
    boost_fn, _ = create_boost_fn(a_row[:4])
    n_particles = len(a_row) // 4
    results = [boost_fn(a_row[4 * x : 4 * (x + 1)]) for x in range(n_particles)]
    return list(itertools.chain(*[x.tolist() for x in results]))


def inv_boost(a_row: np.ndarray):
    """boost all particles to the rest frame of the first particle in the list"""

    assert a_row.shape[0] % 4 == 0, "a_row should be a 4-vector"
    _, inv_boost_fn = create_boost_fn(a_row[:4])
    n_particles = len(a_row) // 4
    results = [inv_boost_fn(a_row[4 * x : 4 * (x + 1)]) for x in range(n_particles)]
    return list(itertools.chain(*[x.tolist() for x in results]))


# <TODO> Use different scaler methods
class InputScaler:
    def __init__(self, feature_range=(-0.99999, 0.99999)):
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def transform(self, df, outname=None):
        out_df = self.scaler.fit_transform(df)
        if outname is not None:
            self.save(outname)
        return out_df

    def save(self, outname):
        pickle.dump(self.scaler, open(outname, "wb"))
        return self

    def load(self, outname):
        self.scaler = pickle.load(open(outname, "rb"))
        return self

    def dump(self):
        print(
            "Min and Max for inputs: {",
            ", ".join(["{:.6f}".format(x) for x in self.scaler.data_min_]),
            ", ".join(["{:.6f}".format(x) for x in self.scaler.data_max_]),
            "}",
        )


def read(filename, max_evts=None, testing_frac=0.1) -> GAN_INPUT_DATA_TYPE:
    """
    Read the input data from a file and return a data for training a GAN
    """
    if type(filename) == list:
        if len(filename) > 1:
            print(len(filename), "too many files!")
        filename = filename[0]

    arrays = np.load(filename)
    truth_in = arrays["out_truth"].astype(np.float32)
    cond_info = arrays["cond_info"].astype(np.float32)

    shuffle(truth_in)
    shuffle(cond_info)
    print(truth_in.shape, cond_info.shape)

    # Split the data into training and testing
    # <HACK, FIXME, NOTE>
    # <HACK, For now a maximum of 10,000 events are used for testing, xju>

    if max_evts:
        print(f"Using {max_evts:,} events out of {truth_in.shape[0]:,} events")
    else:
        max_evts = truth_in.shape[0]

    num_test_evts = int(max_evts * testing_frac)
    print(f"{max_evts - num_test_evts} for training and {num_test_evts} for testing")
    if num_test_evts < 10_000:
        print("WARNING: num_test_evts < 10_000")

    test_in, train_in = cond_info[:num_test_evts], cond_info[num_test_evts:max_evts]
    test_truth, train_truth = truth_in[:num_test_evts], truth_in[num_test_evts:max_evts]
    xlabels = ["phi", "theta"]

    return (train_in, train_truth, test_in, test_truth, xlabels)


def create_dataloader(
    filename, batch_size, num_workers, max_evts=None, testing_frac=0.1
):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    train_cond, train_truth, test_cond, test_truth, xlabels = read(
        filename, max_evts, testing_frac
    )
    train_cond = torch.from_numpy(train_cond)
    train_truth = torch.from_numpy(train_truth)
    test_cond = torch.from_numpy(test_cond)
    test_truth = torch.from_numpy(test_truth)

    train_dataset = TensorDataset(train_cond, train_truth)
    test_dataset = TensorDataset(test_cond, test_truth)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader, test_loader, xlabels


def process_data_split(
    examples_used: Optional[int],
    frac_data_used: Optional[float],
    train_val_test_split: Tuple[float, float, float],
):
    split_by_count = reduce(
        lambda prev, x: isinstance(x, int) and prev, train_val_test_split, True
    )

    if split_by_count:
        if examples_used is not None or frac_data_used is not None:
            raise ValueError(
                f"`examples_used` and `frac_data_used` shouldn't be specified when `train_val_test_split` explicitly states the size of each set"
            )
        examples_used = sum(train_val_test_split)
    else:
        if not np.isclose(sum(train_val_test_split), 1.0):
            raise ValueError(
                f"`train_val_test_split` must sum up to 1.0 when fractions are used"
            )
        if frac_data_used is not None and examples_used is not None:
            raise ValueError(
                f"Specify either `frac_data_used` or `examples_used` but not both!"
            )
        if frac_data_used is not None and not (0 < frac_data_used <= 1.0):
            raise ValueError(
                f"Fraction of data used must be in range (0, 1], but found {frac_data_used}"
            )
    return examples_used


def get_num_asked_events(
    examples_used: Optional[int], frac_data_used: Optional[float], dataset_size: int
) -> int:
    if frac_data_used is not None:
        num_asked_events = int(dataset_size * frac_data_used)
    elif dataset_size < examples_used:
        raise ValueError(f"Asking {examples_used} > {dataset_size} available events")
    else:
        num_asked_events = examples_used

    print(f"Number of events: {dataset_size}, asking for {num_asked_events}")
    return num_asked_events
