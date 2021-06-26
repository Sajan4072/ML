import csv 
import random 

from sklearn.linear_model import Perceptron 
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier


model =Perceptron()
# model=svm.SVC()
# model=KNeighborsClassifier(n_neighbors=1)


#read data in from  file 
with open("banknote.csv") as f:
	reader=csv.reader(f)
	next(reader)  # Since the first row of our csv file contains the headers (or field names), we skip them 


	data=[]
	for row in reader:
		data.append({
			"evidence":[float(cell) for cell in row[:4]],  
			"label":"authentic" if row[4]=="0" else "counterfit"	
			})


# print(data)
# [{'evidence': [3.6216, 8.6661, -2.8073, -0.44699], 'label': 'authentic'}, {'evidence': [4.5459, 8.1674, -2.4586, -1.4621], 'label': 'authentic'},......all data]

# Separate data into training and testing groups
holdout =int(0.40*len(data))
random.shuffle(data)

testing = data[:holdout]
training = data[holdout:]



# Train model on training set
x_training =[row["evidence"] for row in training ]
y_training= [row["label"] for row in training]
model.fit(x_training,y_training)


# Make predictions on the testing set
X_testing = [row["evidence"] for row in testing]
y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing)


# Compute how well we performed
correct = 0
incorrect = 0
total = 0

for actual,predicted in zip(y_testing,predictions):
	total +=1
	if actual == predicted:
		correct +=1
	else:
		incorrect +=1

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")












