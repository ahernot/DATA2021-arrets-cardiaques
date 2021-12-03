from data_import import read_data


TRAIN_PROPORTION = 0.1  # train on 10% of data


data_dict = read_data()

def create_train_test_sets (data_dict):
    

    for data_type in data_dict.keys():



        data = data_dict[data_type]

        
