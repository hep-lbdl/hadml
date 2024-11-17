""" 

You may test the Parser class by adding the code below to your main script:

****************************************************************************************************
from hadml.datamodules.parsers.herwig_multihadron_parser import HerwigMultiHadronEventParser

parser = HerwigMultiHadronEventParser(
    data_dir="data/Herwig",
    raw_file_list=["AllClusters_1K.dat"],
    processed_filename="herwig_multihadron_events_1K.npy",
    pid_map_file="pid_to_idx.pkl",
    debug=True)

parser.parse_data()
****************************************************************************************************

"""

from collections import Counter
import numpy as np
import pandas as pd
import pickle, os
from hadml.datamodules.components.utils import (boost, get_angles)


class HerwigMultiHadronEventParser():  
    """ Parser for reading and processing raw data generated by Herwig:
    quarks -> heaviest cluster -> ... [light clusters] ... -> multiple hadrons. """

    def __init__(self,
            data_dir,                # Data directory path.
            raw_file_list,           # List of raw data filenames.
            processed_filename,      # Processed data filename.
            pid_map_file,            # PID-to-MostCommonID map filename.
            debug=False              # Number of events being processed is 
                                     # printed when True.
    ):
        processed_dir = os.path.join(data_dir, "processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        self.raw_file_list = [os.path.join(data_dir, "raw", f) for f in raw_file_list]
        self.processed_filepath = os.path.join(data_dir, "processed", processed_filename) 
        self.pids_to_idx_path = os.path.join(data_dir, "processed", pid_map_file) 
        self.debug = debug


    def parse_data(self):
        """ Parse data from raw files, prepare the PID-to-MostCommonID map
        and saving processed files as self.processed_filename. """

        # Looking for an existing processed data file
        print('\n', ' '*21, self.__class__.__name__, '\n', '-'*70, sep='')
        if os.path.exists(self.processed_filepath):
            print("Found processed data in:\n ", self.processed_filepath)
            print('-'*70)
            return

        # Loading/creating the PID-to-MostCommonID map
        if os.path.exists(self.pids_to_idx_path):
            print("Loading the existing PID-to-MostCommonID map:", self.pids_to_idx_path)
            with open(self.pids_to_idx_path, "rb") as f:
                self.pids_to_idx = pickle.load(f)
        else:
            self.pids_to_idx = self._create_pids_to_idx_dict(self.debug)

        # Parsing events
        all_data = []
        print("Parsing all events...")
        for raw_path in self.raw_file_list:
            with open(raw_path) as f:
                print(f"  --> Parsing data from file {raw_path}")
                data_list = []
                for event_index, event_line in enumerate(f): 
                    if event_line.strip():
                        if self.debug:
                            print(f"  processing event #{event_index}", end='\r')
                        data_list.append(self._parse_raw_event(event_line))
                if self.debug:
                    print()
                all_data += data_list
        
        # Saving the processed data as NumPy binary files   
        processed_data = {
            "cluster_kin" : [],
            "had_kin" : [],
            "had_kin_rest_frame" : [],
            "had_type_indices" : [],
            "cluster_labels" : [],
            "n_had_type_indices": len(self.pids_to_idx)
        }
        
        for d in all_data:
            processed_data["cluster_kin"].append(d["cluster_kin"])
            processed_data["had_kin"].append(d["had_kin"])
            processed_data["had_kin_rest_frame"].append(d["had_kin_rest_frame"])
            processed_data["had_type_indices"].append(d["had_type_indices"])
            processed_data["cluster_labels"].append(d["cluster_labels"])
        
        with open(self.processed_filepath, "wb") as f:
            np.save(f, processed_data)
        print("Processed data saved in:\n   ", self.processed_filepath, '\n', '-'*70, sep='')    


    def _parse_particles(self, event_line):
        """ Parse all the particles (including quarks, heavy/light cluster and hadrons from
        a single event provided as a raw event line). """
        cluster_decays = event_line.split("|")[:-1]
        particles = [pd.Series(c.split(";")[:-1]).str.split(',', expand=True) 
                        for c in cluster_decays]
        return particles


    def _parse_raw_event(self, event_line):
        """ Parse data presented as an event prepared in a specific format. """

        # Parsing all particles   
        particles = self._parse_particles(event_line)
        q1s, q2s, cs, hadrons, cluster_labels, cs_for_hadrons = [], [], [], [], [], []
        
        # Processing quarks, clusters and hadrons
        for i, particle in enumerate(particles):

            # Selecting the two quarks and the cluster/heavy cluster from the cluster
            q1 = particle.iloc[0].to_numpy()[[1,3,4,5,6]].astype(float).reshape(1, -1)
            q2 = particle.iloc[1].to_numpy()[[1,3,4,5,6]].astype(float).reshape(1, -1)
            c = particle.iloc[2].to_numpy()[[1,3,4,5,6]].astype(float).reshape(1, -1)
            q1s.append(q1)
            q2s.append(q2)
            cs.append(c)
            
            # Selecting the final states from the cluster
            hadron = particle[particle[2] == '[ ]'].to_numpy()[:, [1,3,4,5,6]].astype(float)
            hadrons.append(hadron)
            
            # Assigning cluster labels to hadrons
            cluster_labels += [i] * len(hadron)
            c_for_hadrons = c.repeat(len(hadron), axis=0)
            cs_for_hadrons.append(c_for_hadrons)

        # Concatenating clusters
        q1s = np.concatenate(q1s)
        q2s = np.concatenate(q2s)
        cs = np.concatenate(cs)

        # Hadrons [PID, E, px, py, pz]
        hadrons = np.concatenate(hadrons)  

        # Heaviest clusters [Cluster ID, E, px, py, pz]                 
        cs_for_hadrons = np.concatenate(cs_for_hadrons)    

        # Heaviest clusters + 2 quarks 
        # [c_E, c_px, c_py, c_pz, q1_E, q1_px, q1_py, q1_pz, q2_E, q2_px, q2_py, q2_pz]
        cond_kin = np.concatenate([cs[:, [1,2,3,4]], q1s[:, [1,2,3,4]], q2s[:, [1,2,3,4]]], axis=1)

        # Heaviest cluster + hadron
        # [c_E, c_px, c_py, c_pz, h_E, h_px, h_py, h_pz]
        had_kin = np.concatenate([cs_for_hadrons[:, [1,2,3,4]], hadrons[:, [1,2,3,4]]], axis=1)

        # Heaviest cluster + 2 quarks in the rest frame (rf) of the former
        # [c_E, c_px, c_py, c_pz, q1_Erf, q1_pxrf, q1_pyrf, q1_pzrf, 
        # q2_Erf, q2_pxrf, q2_pyrf, q2_pzrf]
        cond_kin_rest_frame = boost(cond_kin)
        
        # Hadrons in the rest frame (rf) of the heaviest cluster
        # [Erf, pxrf, pyrf, pzrf]
        had_kin_rest_frame = boost(had_kin)[:, 4:]

        # Hadrons [E, px, py, pz]
        had_kin = had_kin[:, 4:]

        # Computing angles for the two quarks via cond_kin_rest_frame, 
        # i.e. [cluster + q1/q2 in the cluster rest frame]
        q_phi, q_theta = get_angles(cond_kin_rest_frame[:, 4:8])
        
        # Computing quark momenta based on the angles
        q_momenta = np.stack([q_phi, q_theta], axis=1)

        # Preparing X (cluster + quark types + quark momenta in the cluster rest frame, crf)
        # [c_E, c_px, c_py, c_pz, q1_type, q2_type, q1_crf_momentum, q2_crf_momentum]
        cond_info = np.concatenate([cs[:, [1,2,3,4]], q1s[:, :1], q2s[:, :1], q_momenta], axis=1)
        cond_info = cond_info.astype(np.float32)

        # Mapping particle IDs to indices using the prepared PID-to-ID dictionary
        try:
            had_type_indices = np.vectorize(self.pids_to_idx.get)(hadrons[:, [0]].astype(np.int32))
        except Exception as X:
            # Debugging what event makes the parser stop working
            print("Line = ", event_line)
            print("Exception: ", X)

        return {
            "cluster_kin" : cond_info,
            "had_kin" : had_kin,
            "had_kin_rest_frame": had_kin_rest_frame,
            "had_type_indices" : had_type_indices,
            "cluster_labels" : cluster_labels
        }
        

    def _create_pids_to_idx_dict(self, debug=False):
        """ Create a new PID-to-MostCommonID map/dictionary """
        
        print("Creating a new PID-to-MostCommonID map/dictionary...")
        hadron_types = []

        for raw_path in self.raw_file_list:
            with open(raw_path) as f:
                print(f"  --> Analysing file {raw_path}")
                for event_index, event_line in enumerate(f): 
                    if event_line.strip():
                        if debug:
                            print(f"    processing event #{event_index}", end='\r')
                        particles = self._parse_particles(event_line)
                        for particle in particles:
                            PIDs = particle[particle[2] == '[ ]'].to_numpy()[:, [1]].astype(float)
                            for id in PIDs: 
                                hadron_types.append(id[0])
                if debug:
                    print()

        count = Counter(hadron_types)
        hadron_pids = list(map(lambda x: x[0], count.most_common()))
        pids_to_idx = {pids: i for i, pids in enumerate(hadron_pids)}

        with open(self.pids_to_idx_path, "wb") as f:
            pickle.dump(pids_to_idx, f)
        print("The PID-to-MostCommonID map has been successfully saved in:",
              "\n  ", self.pids_to_idx_path)

        return pids_to_idx