from typing import Any, Dict, Optional, Tuple, Protocol
import torch, os, numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data.dataset import Dataset as GeometricDataset
from hadml.datamodules.components.utils import process_data_split, get_num_asked_events
from hadml.datamodules.components.herwig_multihadron_parser import HerwigMultiHadronEventParser
from hadml.datamodules.components.herwig import HerwigMultiHadronEventDataset
import matplotlib.pyplot as plt
from dataclasses import dataclass


class GANDataProtocol(Protocol):
    """Define a protocol for GAN data modules."""

    def prepare_data(self) -> None:
        """Prepare data for training and validation.
        Before the create_dataset function is called."""

    def create_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create dataset from core dataset.
        Returns:
            torch.Tensor: conditioinal information
            torch.Tensor: particle kinematics
            torch.Tensor: particle types
        """


class ParticleGANDataModule(LightningDataModule):
    def __init__(
        self,
        core_dataset: GANDataProtocol,
        batch_size: int = 5000,
        num_workers: int = 12,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["core_dataset"])
        self.core_dataset = core_dataset

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        # read the original file, determine the number of particle types
        # and create a map.
        self.core_dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables:
            `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning
        with both `trainer.fit()` and `trainer.test()`
        so be careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = TensorDataset(*self.core_dataset.create_dataset())

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.core_dataset.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            print(f"Number of training examples: {len(self.data_train)}")
            print(f"Number of validation examples: {len(self.data_val)}")
            print(f"Number of test examples: {len(self.data_test)}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=self.hparams.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=self.hparams.drop_last,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class EventGANDataModule(LightningDataModule):
    def __init__(
        self,
        cond_dataset: GeometricDataset,
        obs_dataset: GeometricDataset,
        batch_size: int = 500,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        frac_data_used: Optional[float] = 1.0,
        examples_used: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["cond_dataset", "obs_dataset"])

        self.hparams.examples_used = process_data_split(
            examples_used, frac_data_used, train_val_test_split
        )

        num_asked_events_obs = get_num_asked_events(
            self.hparams.examples_used, frac_data_used, len(obs_dataset)
        )
        num_asked_events_cond = get_num_asked_events(
            self.hparams.examples_used, frac_data_used, len(cond_dataset)
        )
        num_asked_events = min(num_asked_events_obs, num_asked_events_cond)
        print(f"Asking for smaller number of events ({num_asked_events})...")
        self.cond_dataset = cond_dataset[:num_asked_events]
        self.obs_dataset = obs_dataset[:num_asked_events]

        self.cond_data_train: Optional[Dataset] = None
        self.cond_data_val: Optional[Dataset] = None
        self.cond_data_test: Optional[Dataset] = None

        self.obs_data_train: Optional[Dataset] = None
        self.obs_data_val: Optional[Dataset] = None
        self.obs_data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables:
            `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning
        with both `trainer.fit()` and `trainer.test()`
        so be careful not to execute things like random split twice!
        """
        if (
            not self.cond_data_train
            and not self.cond_data_val
            and not self.cond_data_test
        ):

            (
                self.cond_data_train,
                self.cond_data_val,
                self.cond_data_test,
            ) = random_split(
                dataset=self.cond_dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

        if not self.obs_data_train and not self.obs_data_val and not self.obs_data_test:

            (
                self.obs_data_train,
                self.obs_data_val,
                self.obs_data_test,
            ) = random_split(
                dataset=self.obs_dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            print(f"Number of training examples: {len(self.obs_data_train)}")
            print(f"Number of validation examples: {len(self.obs_data_val)}")
            print(f"Number of test examples: {len(self.obs_data_test)}")

    def train_dataloader(self):
        return {
            "cond_data": GeometricDataLoader(
                dataset=self.cond_data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
            ),
            "obs_data": GeometricDataLoader(
                dataset=self.obs_data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
            ),
        }

    def val_dataloader(self):
        loaders = {
            "cond_data": GeometricDataLoader(
                dataset=self.cond_data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
            "obs_data": GeometricDataLoader(
                dataset=self.obs_data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def test_dataloader(self):
        loaders = {
            "cond_data": GeometricDataLoader(
                dataset=self.cond_data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
            "obs_data": GeometricDataLoader(
                dataset=self.obs_data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")


class MultiHadronEventGANDataModule(LightningDataModule):
    def __init__(
            self, 
            data_dir="data/Herwig",
            raw_file_list=["AllClusters_10K.dat"],
            processed_filename="herwig_multihadron_events_10K.npy",
            dist_plots_filename="distribution_plots_10K.pdf",
            pid_map_file="pid_to_idx.pkl",
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            batch_size=32,
            num_workers=8,
            initialise_data_preparation=False,
            debug=True
        ):
        super().__init__()

        self.data_dir = data_dir
        processed_path = os.path.join(os.path.normpath(data_dir), "processed")
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        self.processed_filename = os.path.join(processed_path, processed_filename)

        dist_plots_path = os.path.join(os.path.normpath(self.data_dir), "plots")
        if not os.path.exists(dist_plots_path):
            os.makedirs(dist_plots_path)
        self.dist_plots_path = os.path.join(dist_plots_path, dist_plots_filename)

        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        parser = HerwigMultiHadronEventParser(
            data_dir=data_dir,
            raw_file_list=raw_file_list,
            processed_filename=processed_filename,
            pid_map_file=pid_map_file,
            debug=debug
        )
        parser.parse_data()

        print('\n', ' '*20, self.__class__.__name__,'\n', '-'*70, sep='')

        if initialise_data_preparation:
            self.prepare_data()
            self.setup()
            one_train_batch = next(iter(self.train_dataloader()))
            gen_input, disc_input = one_train_batch
            print(f"\nGenerator input batch: {len(gen_input)} ([{gen_input.size()}])")
            print(f"Discriminator input batch: {len(disc_input)} ([{disc_input.size()}])")
            print("\nGenerator input sample:\n", gen_input[0])
            print("\nDiscriminator input sample:\n", disc_input[0])

    def prepare_data(self):
        # Load data prepared by the parser
        with open(self.processed_filename, "rb") as f:
            data = np.load(f, allow_pickle=True)
        
        cluster_kin = data.item()["cluster_kin"]
        hadron_kin = data.item()["had_kin"]
        had_type_indices = data.item()["had_type_indices"]
        cluster_labels = data.item()["cluster_labels"]
        n_events = len(cluster_kin)
        self.n_had_types = data.item()["n_had_type_indices"] + 1 # 1 extra type for a stop/padding token
        hadron_kin_rest_frame = data.item()["had_kin_rest_frame"]

        # Assigning hadrons to clusters 
        self.clusters, self.hadrons_with_types, n = self._get_hadrons_and_clusters__(
            n_events, cluster_kin, cluster_labels, hadron_kin_rest_frame, had_type_indices)
        n_clusters_extracted_from_events = n
        n_hadrons_per_cluster = [len(hadron_seq.types) for hadron_seq in self.hadrons_with_types]
        self.max_n_hadrons = max(n_hadrons_per_cluster)
        n_hadrons_per_event = [len(d) for d in hadron_kin]

        # Preparing distribution plots
        if not os.path.exists(self.dist_plots_path):
            hadron_energy = np.concatenate([d for d in hadron_kin])[:, 0]
            cluster_energy = np.concatenate([d for d in cluster_kin])[:, 0]
            self._plot_dist(
                filepath=self.dist_plots_path, 
                data=[[n_hadrons_per_event, n_hadrons_per_cluster], [hadron_energy, cluster_energy]],
                xlabels=[["Number of hadrons", "Number of hadrons"], 
                         ["Energy [GeV]", "Energy [GeV]"]],
                ylabels=[["Events", "Clusters"], 
                         ["Hadrons", "Clusters"]],
                legend_labels=[["Hadron Multiplicity\nDistribution per Event", 
                               "Hadron Multiplicity\nDistribution per Cluster"],
                               ["Hadron Energy Distribution", "Cluster Energy Distribution"]])

        # Printing general statistics about events, clusters and hadrons
        print("Initial number of events:", n_events)
        print("Total number of clusters:", n_clusters_extracted_from_events)
        print("Total number of hadron types (with a stop token):", self.n_had_types)
        print("Largest number of hadrons per cluster:", self.max_n_hadrons)
        print("Largest number of hadrons per event:", max(n_hadrons_per_event))

    def _get_hadrons_and_clusters__(self, n_events, cluster_kin, cluster_labels, hadron_kin, 
                                     had_type_indices):
        clusters, hadrons_with_types = [], []
        n_clusters_extracted_from_events = 0

        for i in range(n_events):
            cluster_in_event = []  # [cluster ... cluster]
            hadrons_in_event = []  # [HadronsWithTypes ... HadronsWithTypes]

            for j, cluster in enumerate(cluster_kin[i]):
                # Extracting all clusters from a single event
                cluster_idx_mask = [cl == j for cl in cluster_labels[i]]
                cluster = torch.from_numpy(cluster)
                cluster_in_event.append(cluster)
                n_clusters_extracted_from_events += 1
                # Extracting all hadrons from a single event
                hadrons = torch.tensor(hadron_kin[i][cluster_idx_mask])
                # Shifting the hadron type for an additional type assigned to a stop token (= 0) 
                hadron_types = torch.tensor(had_type_indices[i][cluster_idx_mask]).squeeze(1) + 1 
                hadrons_in_event.append(HadronsWithTypes(hadrons, hadron_types))

            clusters += cluster_in_event
            hadrons_with_types += hadrons_in_event

        return clusters, hadrons_with_types, n_clusters_extracted_from_events

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            # Passing clusters and hadrons to a dataset reponsible for tokenisation
            dataset = HerwigMultiHadronEventDataset(
                self.n_had_types,
                self.clusters,
                self.hadrons_with_types,
                self.max_n_hadrons)
            
            # Creating the training, validation and test datasets
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            print(f"Number of training examples: {len(self.data_train)}")
            print(f"Number of validation examples: {len(self.data_val)}")
            print(f"Number of test examples: {len(self.data_test)}")
            print('-'*70)
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(42)
        )

    def test_dataloader(self):
         return DataLoader(
            dataset=self.data_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(42)
        )
    
    def _get_n_bins(self, data):
        """ Compute Scott's normal reference rule and return
        the number of bins for a histogram plot """
        data = np.array(data)
        std, max_value, min_value = data.std(), data.max(), data.min()
        data_len = len(data)

        bin_width = 3.49 * std / np.power(2 * data_len, 1/3)
        n_bins = np.rint((max_value - min_value) / bin_width)
        
        return int(n_bins)

    def _plot_dist(self, filepath, data, xlabels, ylabels=None, legend_labels=None, labels=None):
        """ Draw distribution diagrams for a list of three data sets """
        plt.clf()
        _, ax = plt.subplots(2, 2, figsize=(13, 8))
        colors = [["black", "black"], ["orange", "orange"]]

        for r in range(len(data)):
            for c in range(len(data)):
                samples = data[r][c]
                # Setting the appropriate bin range
                if legend_labels[r][c].startswith("Hadron Multiplicity"):
                    sample_range = [1, max(samples)]
                    bins = np.linspace(
                        start=sample_range[0] - 0.5, 
                        stop=sample_range[1] + 0.5, 
                        num=sample_range[1] - sample_range[0] + 2, 
                        retstep=0.5)[0]
                else:
                    bins = self._get_n_bins(samples)

                # Preparing a chart
                ax[r][c].hist(samples, bins=bins, color=colors[r][c], alpha=0.7,
                        label=legend_labels[r][c])
                ax[r][c].set_xlabel(xlabels[r][c])
                if ylabels[r][c] is not None:
                    ax[r][c].set_ylabel(ylabels[r][c])
                ax[r][c].legend(loc='upper right')

                # Setting the ticks along OX if needed
                if legend_labels[r][c].startswith("Hadron Multiplicity"):
                    density = 2
                    xticks = np.arange(start=sample_range[0] - 1, stop=sample_range[1] + 1, 
                                    step=density)[1:]
                    ax[r][c].set_xticks(xticks)

        plt.tight_layout()
        plt.savefig(filepath)
        print("Distribution diagrams have been saved in\n   ", filepath)


@dataclass
class HadronsWithTypes:
    """Class holding hadrons and their types/indices, 
    assuming all originate from the same heavy cluster."""
    kinematics: torch.Tensor # [h_kin ... h_kin]
    types: torch.Tensor      # [h_id ... h_id]

    def get_kinematics_dims(self) -> int:
        return len(self.kinematics[0])