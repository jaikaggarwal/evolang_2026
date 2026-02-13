import json

def read_json_file(data_path):
	with open(data_path, mode='r') as file_handle:
		data = json.loads(file_handle.read())
	return data
