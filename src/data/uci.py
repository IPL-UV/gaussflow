from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pyprojroot import here
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


# spyder up to find the root
root = here(project_files=[".here"])


def download_uci_data(target_dir: str = None):
    # GET TARGET DIRECTORY
    # create target directory
    if target_dir is None:
        target_dir = Path(root).joinpath("datasets/uci")

    # create directory if it doesn't exist
    try:
        Path(target_dir).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Folder '{target_dir}' Is Already There.")
    else:
        print(f"Folder '{target_dir}' is created.")

    # DOWNLOAD FROM URL
    import urllib.request

    url = "https://zenodo.org/record/1161203/files/data.tar.gz"

    ds_dir = str(Path(target_dir).joinpath(str(Path(url).name)))
    urllib.request.urlretrieve(url, ds_dir)

    # UNZIP FILE
    import tarfile

    ds_dir = str(Path(target_dir).joinpath(str(Path(url).name)))
    tf = tarfile.open(ds_dir)
    tf.extractall(target_dir)

    return None


class GasDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size: int = 128, dataset_dir: str = None, standardize: bool = True
    ):
        if dataset_dir is None:
            dataset_dir = Path(root).joinpath(
                "datasets/uci/data/gas/ethylene_CO.pickle"
            )
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.standardize = standardize

    def prepare_data(self):
        # load data
        data = self.load_data()
        self.data = data

    def setup(self):
        # clean data
        self.clean_data()

        # split data to train and test
        self.Xtrain, self.Xval, self.Xtest = self.split_data()

        if self.standardize:
            scaler = StandardScaler().fit(
                np.concatenate([self.Xtrain, self.Xval], axis=0)
            )
            self.Xtrain = scaler.transform(self.Xtrain)
            self.Xval = scaler.transform(self.Xval)
            self.Xtest = scaler.transform(self.Xtest)

        self.ds_train = GenericDataset(self.Xtrain)
        self.ds_val = GenericDataset(self.Xval)
        self.ds_test = GenericDataset(self.Xtest)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def valid_dataloader(self):
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def load_data(self):
        data = pd.read_pickle(self.dataset_dir)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data

    def clean_data(self):

        data = self.data

        B = self._get_correlation_numbers(data)

        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = self._get_correlation_numbers(data)

        # normalize
        data = (data - data.mean()) / data.std()

        self.data = data

    def _get_correlation_numbers(self, data):
        correlation = data.corr()
        A = correlation > 0.98
        data = A.values.sum(axis=1)
        return data

    def split_data(self):
        data = self.data
        n_test = int(0.1 * data.shape[0])
        data_test = data[-n_test:]
        data_train = data[0:-n_test]
        n_validate = int(0.1 * data_train.shape[0])
        data_validate = data_train[-n_validate:]
        data_train = data_train[0:-n_validate]
        return data_train, data_validate, data_test

