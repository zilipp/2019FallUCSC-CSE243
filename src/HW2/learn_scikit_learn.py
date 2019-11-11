from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

a_file_path = '../HW2_Data/a.arff'
a_data = arff.loadarff(a_file_path)
a_df = pd.DataFrame(a_data[0])
a_df.rename(columns={'class':'label'}, inplace=True)
print(a_df.head(10))

# split data
# X
a_features = ['x', 'y']
a_X = a_df[a_features]
# y
a_y = a_df.label
# split
train_a_X, val_a_X, train_a_y, val_a_y = train_test_split(a_X, a_y, random_state = 1)


# a: DT model
# model
a_DT_model = DecisionTreeRegressor(random_state=1)
# train
a_DT_model.fit(train_a_X, train_a_y)




#
# b_file_path = '../HW2_Data/b.arff'
# b_data = arff.loadarff(b_file_path)
# b_df = pd.DataFrame(b_data[0])