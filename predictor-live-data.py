import os

os.chdir('/Users/yigitcelik/Desktop/github/lead_time_muneccimi/src/data/')
os.system('python make_dataset_live.py')
os.chdir('/Users/yigitcelik/Desktop/github/lead_time_muneccimi/src/features/')
os.system('python build_features_for_live.py')
os.chdir('/Users/yigitcelik/Desktop/github/lead_time_muneccimi/src/models/')
os.system('python predict_neural_network_model.py')
print('tamam')
