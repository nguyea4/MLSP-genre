% Andrew's main
clear all
close all

% Get the songs, clean variables
readAllSongs
clearvars -except audioMatrix audio labels fs

% Extract features.
spectral_featureextraction

