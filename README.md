# Song Genre Classification 
for JHU EP Machine Learning for Signal Processing
by: Andrew Nguyen, Cameron Greene, Irene O'Hara, Jack Mccarty
We are utilizing MATLAB
## How to get this Repo
`git clone git@github.com:nguyea4/MLSP-genre.git`
## Running stuff
Be in the 'MLSP-genre' directory
## Using Virtual environment (if we use python)
So we all have same packages
Following [this](https://realpython.com/python-virtual-environments-a-primer/) link. 
To Create it
* `sudo apt-get install python3-venv`
* (If not already made:)`python3 -m venv mlvenv`
* To start environment: `source mlvenv/bin/activate`
* (requirements.txt is a WIP) `pip3 install-r requirements.txt`

## Git refresher
`git clone git@github.com:nguyea4/learngit.git`
Things I use alot:  
1. **`git status`**
2. **`git branch --all`**
3. **`git checkout branchname`**
4. **`git pull`**
5. **`git add whatevername.py`**
6. **`git commit -m "Title" -m "Description .........."`**
7. **`git push`**

## Description of Files
bayes_classifier.m</br>
comp_guass_dens_val.m</br>
compute_error.m</br>
divergence.m</br>
euclidean_classifier.m</br>
feature30sec.mat</br>
k_nn_classifier.m</br>
mccarty_classification.m</br>
readAllSongs.m: Stores all the '.wav' data file into audioDataMatrix</br>
readfiles.m</br>
spectral_feature_assessment.m: Explore separability of different spectrogram-based features. Including spectrogram hyperparameters, mean and variance of each frequency, PCA of PSD, NMF of PSD  </br>
