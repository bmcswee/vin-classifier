
# coding: utf-8

# In[1]:


# Imports
import os
import re
import sys
import pandas as pd

# Path for all raw data files
inputFolder = "raw-data\\"
outputFolder = "clean-data\\"

firstMerge = 1

# Get all files in the directory
files = os.listdir(inputFolder)

# Loop through the files in the directory
for file in files:
    
    # Only accept files that are csv
    if file[len(file)-4:] == '.csv':
        
        # Read in file
        table = pd.read_csv(inputFolder + file)
        
        # Drop the unneeded columns
        table = table.drop(columns=['web-scraper-order', 'web-scraper-start-url'])
        
        # Drop all of the 'NaN' rows
        table = table.dropna()
        
        # Make the VIN column the index column
        table = table.set_index('vinregex')
        
        # Create a row for the year and rearrange columns to make more sense
        table['year'] = table.apply(lambda _: '', axis=1)
        table['vinchar1'] = table.apply(lambda _: '', axis=1)
        table['vinchar2'] = table.apply(lambda _: '', axis=1)
        table['vinchar3'] = table.apply(lambda _: '', axis=1)
        table['vinchar4'] = table.apply(lambda _: '', axis=1)
        table['vinchar5'] = table.apply(lambda _: '', axis=1)
        table['vinchar6'] = table.apply(lambda _: '', axis=1)
        table['vinchar7'] = table.apply(lambda _: '', axis=1)
        table['vinchar8'] = table.apply(lambda _: '', axis=1)
        table['vinchar9'] = table.apply(lambda _: '', axis=1)
        table['vinchar10'] = table.apply(lambda _: '', axis=1)
        table['vinchar11'] = table.apply(lambda _: '', axis=1)
        table['vinchar12'] = table.apply(lambda _: '', axis=1)
        table['vinchar13'] = table.apply(lambda _: '', axis=1)
        table['vinchar14'] = table.apply(lambda _: '', axis=1)
        table['vinchar15'] = table.apply(lambda _: '', axis=1)
        table['vinchar16'] = table.apply(lambda _: '', axis=1)
        table['vinchar17'] = table.apply(lambda _: '', axis=1)
        table['make'] = table.apply(lambda _: '', axis=1)
        table['model'] = table.apply(lambda _: '', axis=1)
        table = table[['year', 'makemodel', 'vinchar1', 'vinchar2', 
                       'vinchar3', 'vinchar4', 'vinchar5', 'vinchar6', 'vinchar7', 'vinchar8', 
                       'vinchar9', 'vinchar10', 'vinchar11', 'vinchar12', 'vinchar13', 'vinchar14', 
                       'vinchar15', 'vinchar16', 'vinchar17', 'make', 'model' ]]
        
        # Loop through the rows
        for index, row in table.iterrows():
            
            #vin_year = index[9:10]
            
            # Remove 'New', 'Used', and 'Certified'
            rawName = row[1]
            if rawName.find('New') != -1 and rawName.find('New') < 5:
                name = rawName[4:]
            elif rawName.find('Used') != -1 and rawName.find('Used') < 5:
                name = rawName[5:]
            elif rawName.find('Certified') != -1 and rawName.find('Certified') < 5:
                name = rawName[10:]
            else:
                sys.exit('Unknown prefex: ' + rawName)
            # End of if statement
            
            # Seperate year and madeModel
            year = name[:4]
            name = name[5:]
            
            # Force to all uppercase
            name = name.upper()
            
            # Remove all non alphanumeric chars from vehicle name
            name = re.sub('[^a-zA-Z\d\s:]', "", name)
            
            # Address special cases where vehicle maker is two words
            name = name.replace("LAND ROVER", "LANDROVER")
            name = name.replace("ALFA ROMEO", "ALFAROMEO")
            
            # Split vehicle maker into its own column
            # Vehicle maker name should now be the first word
            make = name.split(' ', 1)[0]
            
            # Split vehicle model into its own column
            # This also drops extra terms after model name like "Extended Cab", "4x4", etc
            model = name.split(' ', 1)[1].split(' ', 1)[0]
                    
            
            # Split each VIN character into its own column
            #print("this VIN is", index)
            for i, c in enumerate(index, start=1):
                #print("vin char", i, " is ", c)
                # Convert VIN character to number for ML use
                try:
                    vinchar_int = int(c)
                except ValueError:
                    vinchar_int = ord(c) - ord('A') + 10
                row[1+i] = vinchar_int
            

            # Set year and name
            row[0] = year
            row[1] = name
            row['make'] = make
            row['model'] = model
        
        # Check to see if its the first dataframe to be merged
        if firstMerge == 1:
            df = table
            firstMerge = 0
        else:            
            df = pd.concat([df, table])
            
        # End of if statement
    # End of if statement
# End of for loop

# Drop rows with duplicate VIN
df = df[~df.index.duplicated(keep='first')]

# Drop samples before year 2009. Currently, we have too few samples for those years.
df['year'] = pd.to_numeric(df['year'])
df = df.drop(df[df.year < 2009].index)

# Print out data frame to console
print(df)

# Print data frame to a csv file
df.to_csv(outputFolder + "dataset-fullsplit-make-model.csv")

