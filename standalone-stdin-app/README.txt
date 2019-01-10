
PURPOSE: These files allow the user to input a VIN number and receive a prediction against trained model(s).


------------------------------------------------------------------------------------------

How to run directy from command line:

Install Python (https://www.anaconda.com/download/)

To run:
1. open cmd/powershell and navigate to folder location
	ex: "cd C:/Users/some_user/Desktop/vin_project/"

2. Use run command for targeted file (below)

------------------------------------------------------------------------------------------
vin-decoder-three-trees-multiclass.py

	This file contains THREE multiclass decision tree classifiers. One for each of the features: year, make, and model.
	
Accaurcy:
	Year: 100%
	Make: 97.4398%
	Model: 95.0739%

Run Command: "python vin-decoder-three-trees-multiclass.py"

------------------------------------------------------------------------------------------
vin-decoder-one-tree-multilabel.py

	This file contains ONE multilabel decision tree classifier and uses it for all three features: year, make, and model.

Accaurcy:
	Year: 100%
	Make: 96.9788%
	Model: 62.5004%

Run Command: "python vin-decoder-one-tree-multilabel.py"