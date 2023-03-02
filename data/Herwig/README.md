### Structure of the data trainig file
[Example of the file](/data/Herwig/raw/ClusterTo2Pi0_new.dat) 
1. Each line of the data file contains information from 1 event about the objects (clusters and particles, which are shown in the red rectangle).
![alt text](/data/Herwig/event.png)
3. Each object is described by 5 numbers: particles id, energy E, momentum components px, py, pz. 
   - Particle id is an integer telling us what particle it is, see: https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf 
   - In that data set we are interested in so called Pions with id +/-111, +/-211 ("-" means anti-particle)  
   - The clusters have id 81 and heavy clusters 88.
4. One event (1 line in the file) can have few clusters (see fig above); information about each cluster decay is separated by "|".
i.e.: 
```
quark + quark -> cluster-> hadron + hadron |   quark + quark -> cluster-> hadron + hadron | quark + quark -> cluster-> hadron + hadron | ... |
```
4. Some clusters are to heavy and they decay to smaller clusters 
```
quark1 + quark2 -> cluster > small_cluster + small_cluster -> and each small_cluster -> hadron + hadron 
```
we are not working on the heavy cluster now so in that case currently the data files has place holder  
```
88,0.0000,0.0000,0.0000,0.0000;88,0.0000,0.0000,0.0000,0.0000 
```
