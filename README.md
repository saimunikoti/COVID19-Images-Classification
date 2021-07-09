# COVID19-Images-Classification
Identification of COVID-19 from chest radiography images

## data file ##
data/covid-19/four_classes

## scripts file ##
1. src/data/config.py: directory path
2. src/models/Baseline_ClassModel.py: Classification model for resnet-50 and Xception based models
3. src/models/CNN_ClassModel.py: Classification model with shallow CNN based architectures.
4. src/visualization/visualize.py: utility functions for visualization

5. data folder: processed images

## Run
1. change directory of your local data directory in config.py file

2. Run src/models/CNN_ClassModel.py for CNN model results

3. Run  src/models/Baseline_ClassModel.py for baseline implemenetation

Note: There are other files that are developed for trial and error but not needed for the output.
