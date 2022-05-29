from my_data import MusicData
from tabulate import tabulate

data = MusicData()
df = data.get_dataframe()
X_train, y_train = data.get_x_y_training()
X_test, y_test = data.get_x_y_test()

#print(tabulate(y_train.sample(10, random_state=1), tablefmt='psql'))
print(data.get_target_as_category())