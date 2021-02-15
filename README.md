# ToF Mass Calibration Classification

This repository contains code for cleaning ToF Data and using it to train models
for classifying ToF spectra as well calibrated or poorly calibrated.

I work mostly in jupyter notebooks but whenever I write a function, I try to generalize
it and add it to a python script to save time in the future.

### Contents

+ Data: Raw cas, csv, and txt data for project. 
    + To save space base only individual file datasets and processed_cas.csv are included. Other files are rarely used and are created in the jupyter notebooks
+ Models: Folder for completed candidate models or models which take so long to train that saving them is worthwhile
+ Notebooks: Folder for all jupyter notebooks
+ SRC: scripts and classes used in project