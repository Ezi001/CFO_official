import numpy as np

import utils.readers as readers


def test_load_lorenz_normalization_and_split(monkeypatch):
    traj = np.linspace(-3.0, 7.0, num=10 * 6 * 3, dtype=np.float32).reshape(10, 6, 3)

    monkeypatch.setattr(readers.np, "load", lambda _: {"traj": traj})
    train, test = readers.load_Lorenz(file_path="dummy.npz", train_ratio=0.8)

    assert train.shape == (8, 6, 3)
    assert test.shape == (2, 6, 3)
    assert np.isclose(train.min(), 0.0)
    assert np.isclose(train.max(), 1.0)


def test_load_burgers_split(monkeypatch):
    output = np.random.RandomState(0).randn(12, 101, 101).astype(np.float32)

    monkeypatch.setattr(readers, "loadmat", lambda _: {"output": output})
    train, test = readers.load_burgers(file_path="dummy.mat", seed=0)

    assert train.shape[0] == 10
    assert test.shape[0] == 2
    assert train.shape[1:] == (101, 100)


def test_load_dataset_splits_eval_test_halves(monkeypatch):
    train = np.zeros((9, 4, 3), dtype=np.float32)
    heldout = np.zeros((6, 4, 3), dtype=np.float32)

    monkeypatch.setattr(readers, "load_Lorenz", lambda *args, **kwargs: (train, heldout))
    splits = readers.load_dataset_splits("lorenz")

    assert splits["train"].shape[0] == 9
    assert splits["eval"].shape[0] == 3
    assert splits["test"].shape[0] == 3


def test_load_dataset_aliases(monkeypatch):
    train = np.zeros((8, 4, 2), dtype=np.float32)
    heldout = np.zeros((4, 4, 2), dtype=np.float32)

    monkeypatch.setattr(readers, "load_diffusion_reaction", lambda *args, **kwargs: (train, heldout))
    monkeypatch.setattr(readers, "load_shallow_water_pdebench", lambda *args, **kwargs: (train, heldout))
    monkeypatch.setattr(readers, "load_nordic", lambda *args, **kwargs: (train, heldout))

    splits_dr = readers.load_dataset_splits("dr")
    splits_swe = readers.load_dataset_splits("swe")
    splits_nordic = readers.load_dataset_splits("nordic")

    assert splits_dr["train"].shape[0] == 8
    assert splits_swe["train"].shape[0] == 8
    assert splits_nordic["train"].shape[0] == 8


def test_load_nordic_ignores_static_groups_and_sorts_trajectories(monkeypatch):
    class Group(dict):
        pass

    class H5File:
        def __init__(self):
            self.groups = {
                "0002": Group(data=np.full((3, 2, 2, 1), 2, dtype=np.float32)),
                #"land_mask": Group(data=np.zeros((1, 2, 2, 1), dtype=np.float32)),
                "0001": Group(data=np.full((3, 2, 2, 1), 1, dtype=np.float32)),
                #"log_bathymetry": Group(data=np.zeros((1, 2, 2, 1), dtype=np.float32)),
                "0004": Group(data=np.full((3, 2, 2, 1), 4, dtype=np.float32)),
                "0003": Group(data=np.full((3, 2, 2, 1), 3, dtype=np.float32)),
            }

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def keys(self):
            return self.groups.keys()

        def __getitem__(self, key):
            return self.groups[key]

    monkeypatch.setattr(readers.h5py, "File", lambda *_args, **_kwargs: H5File())
    train, test = readers.load_nordic(
        file_path="dummy.h5",
        training_ratio=0.5,
        random_seed=0,
        start_ind=0,
        end_ind=3,
    )

    assert train.shape == (2, 3, 2, 2, 1)
    assert test.shape == (2, 3, 2, 2, 1)
    assert set(np.unique(train)) == {3.0, 4.0}
    assert set(np.unique(test)) == {1.0, 2.0}
