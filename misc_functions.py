import pickle
import numpy as np



def transform_to_unif(params,bounds):
	log_params = {}
	for param_name, param_val in params.items():
		if param_name in bounds.keys():
			bound_a,bound_b = bounds[param_name]
			log_params[param_name] = (param_val - bound_a)/(bound_b - bound_a)
		else:
			log_params[param_name] = param_val
	return log_params

def transform_from_log_unif(log_params,bounds):
	params = {}
	for param_name, param_val in log_params.items():
		if param_name in bounds.keys():
			bound_a, bound_b = bounds[param_name]
			param_trans = (bound_b - bound_a)*param_val + bound_a 
			params[param_name] = 10**param_trans
		else:
			params[param_name] = param_val
	return params

def transform_to_log_unif(params,bounds):
	log_params = {}
	for param_name, param_val in params.items():
		if param_name in bounds.keys():
			bound_a,bound_b = bounds[param_name]
			log_params[param_name] = (np.log10(param_val) - bound_a)/(bound_b - bound_a)
		else:
			log_params[param_name] = param_val
	return log_params

def transform_from_std_norm(std_params,dist_params):
	params = []
	for std_param, dist_param in zip(std_params,dist_params.values()):
		params.append(std_param*dist_param[1]+dist_param[0])
	return np.array(params)

def transform_to_std_norm(params,dist_params):
	std_params = []
	for param, dist_param in zip(params,dist_params.values()):
		std_params.append((param-dist_param[0])/dist_param[1])
	return np.array(std_params)

def load_obj(name):
    """
    Load a pickle file. Taken from
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
    :param name: Name of file
    :return: the file inside the pickle
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    """
    Save a pickle file. Taken from
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file

    :param  obj: object to save
            name: Name of file
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


