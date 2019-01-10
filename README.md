# SKLearn VIN Classifier

Several different types of classifiers are used to predict the make, model, and year of a vehicle from its vehicle identification number (VIN).

## Getting Started

The project consists of four parts: a data scraper targeting Cars.com, a utility script for cleaning scraped data, the classifiers themselves, and a standalone script allowing the user to enter a VIN and receive a prediction.

### Dataset

A small dataset of approximately 3300 training examples is included, and can be found in the data-cleaner directory. The user may scrape additional examples from Cars.com using the included script.

### Directory Structure

* **data-cleaner**: Contains the script for cleaning raw scraped data. Raw and clean data are contained within respective subfolders.
* **web-scaper**: Contains configuration file for Chrome web scraper plugin. Has its own readme with further instructions.
* **sklearn-text-classifier**: Contains scripts for testing and evaluating numerous classifiers against different aspects of our problem. Also contains scripts that will train various types of decision trees and output decision tree charts in  PDF format. Decision tree PDFs are also contained here.
* **standalone-stdin-app**: Contains scripts that allow the user to input VIN numbers and receive a prediction from trained model(s).

## Authors

* Brendan McSweeney
* Seth Percy
* Nathan Rich

This project was submitted as coursework for COS 475 (Machine Learning) at the University of Southern Maine.

