import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Determine the desired length (e.g., the length of the first element)
desired_length = len(data[0])  # Assuming all items should be the same length

# Ensure that each sequence is padded or truncated to the desired length
data_processed = []

for item in data:
    item_length = len(item)
    
    # If the item is shorter, pad it
    if item_length < desired_length:
        padded_item = np.pad(item, (0, desired_length - item_length), 'constant')  # Padding with zeros
    # If the item is longer, truncate it
    elif item_length > desired_length:
        padded_item = item[:desired_length]  # Truncate the extra part
    else:
        padded_item = item  # No change if the lengths are already equal
    
    data_processed.append(padded_item)

# Convert the processed data into a numpy array
data = np.asarray(data_processed)

# Check the shape of the data
print(f"Shape of data: {data.shape}")
print(f"Shape of labels: {labels.shape}")

# Split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
