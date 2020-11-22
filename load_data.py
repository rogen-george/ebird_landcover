import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn import svm
from sklearn.svm import LinearSVC

bird_readings = pd.read_excel("readings_normalized_landcover.xlsx", usecols = "C:IR")

land_urban = pd.read_excel("readings_normalized_landcover.xlsx", usecols = ['pland_13_urban'])
land_wetland = pd.read_excel("readings_normalized_landcover.xlsx", usecols = ['pland_11_wetland'])
mixed_forest = pd.read_excel("readings_normalized_landcover.xlsx", usecols = ['pland_05_mixed_forest'])
grassland = pd.read_excel("readings_normalized_landcover.xlsx", usecols = ['pland_10_grassland'])
needle_leaf = pd.read_excel("readings_normalized_landcover.xlsx", usecols = ['pland_01_evergreen_needleleaf'])

# Convert values less than threshold to 0 and greater than threshold to 1 for classification
def convert_to_classes ( data, threshold = 0.1 ):
    data[ data > threshold ] = 1
    data[ data <= threshold ] = 0
    return np.array(data)

threshold = 0.5
land_urban = convert_to_classes( land_urban, threshold )
land_wetland = convert_to_classes( land_wetland, threshold )
mixed_forest = convert_to_classes( mixed_forest, threshold )
grassland = convert_to_classes( grassland, threshold )
needle_leaf = convert_to_classes( needle_leaf, threshold )

data = np.array( bird_readings )

#print ("Marsh Wren", np.sum(marsh_wren) )
#print ( "Yellow Throat", np.sum(yellow_throat))
#print ( "Red Winged ", np.sum(red_winged))
label = needle_leaf

# Delete columns where all readings are zero
#idx = np.argwhere(np.all(data[..., :] == 0, axis=0))
#data = np.delete(data, idx, axis=1)

train_data, test_data, train_label, test_label = train_test_split( data, label, train_size = 0.5, test_size = 0.5, shuffle = True, stratify = label)

svclassifier = svm.SVC(kernel='linear')
svclassifier.fit(train_data, train_label)
y_pred = svclassifier.predict(test_data)

f1 = f1_score(test_label,y_pred)
roc = roc_auc_score(test_label,y_pred)

print ( "Print without anything ", f1, roc )

positive_train = []
negative_train = []
# Split train data into positive and negative to solve the problem of class imbalances
for i in range( len( train_data ) ):
    if train_label[i] == 1:
        positive_train.append( train_data[i] )
    else:
        negative_train.append( train_data[i] )

positive_train = np.array( positive_train )
negative_train = np.array( negative_train )
