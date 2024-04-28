import json

# # 指定要打开的JSON文件的路径
# file_path = '/data/heyinong/spatialVLM_Output/processed_dataset.json'

# # 打开并读取JSON文件
# try:
#     with open(file_path, 'r') as file:
#         # 加载JSON数据
#         data = json.load(file)
#         # 打印读取的数据
#         print(data[1])
# except FileNotFoundError:
#     print(f"The file {file_path} does not exist.")
# except json.JSONDecodeError:
#     print("Error decoding JSON from the file.")
# except Exception as e:
#     print(f"An error occurred: {e}")
import pickle
import pandas as pd
# Replace 'filename.pkl' with the path to your .pkl file
filename = '/data/heyinong/spatialVLM_Output4/chunk_2.pkl'

# Open the file in binary read mode
with open(filename, 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the objects loaded from the .pkl file
df=pd.DataFrame(data)
print(df.columns)

