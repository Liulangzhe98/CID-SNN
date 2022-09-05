# Siamese neural network for camera identification through sensor pattern noise

This repository consists of code to reproduce the results reported in our bachelors projects. Experiments were conducted on the Dresden data set. 

## Requirements
- Python 3.8 or higher
- Python libraries

All the needed libraries can be installed using: 
* `pip install -r requirements.txt`.

## Running the program

The program is ran via the command:
* `python main.py [options]` 

```
Train or load a siamese neural network for camera identification

options:
 == Required
    --size [{small,medium,large,color}]
        Runs with different transformations on the images and different layers in the ConvNets
    --mode [{create,test,both}]
        Create/test model or do both (default: create)
    --ID ID           
        Used for storing the created files with an unique experiment ID

 == Optional
    --dev                 
        Runs in development mode
    --save            
        Stores the model in the Models folder and keeps track of the parameters used
     --load [ID]     
        Loads the model with ID from the Models folder

```