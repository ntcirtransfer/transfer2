import numpy as np
import torch
from torchvision import transforms, io
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Dataset
import dpr_models as dpr_models

device = torch.device('cpu')

def load_categorical_data(df, configs):
	for col in df[configs['sensor_feats']].select_dtypes(['object']):
		df[col] = df[col].fillna('Unknown')
		df[col] = df[col].map(lambda x: configs[col][x] if x in configs[col] else 0.)
    
	return df

# Create a dataset class for image and sensor
class ImageSensorDataset(Dataset):
    def __init__(self, image_data, sensor_data):
        self.image_data = image_data
        self.sensor_data = sensor_data

    def __len__(self):
        return len(self.sensor_data)

    def __getitem__(self, idx):
        images = self.image_data[idx]
        sensors = self.sensor_data[idx]

        return images, sensors

def predict(configs, image_data, sensor_data):
	image_encoder = dpr_models.ImageEncoder(output_dim=configs['embedding_dim']).to(device)
	sensor_encoder = dpr_models.SensorEncoder(avg=configs['sensor_avg'].to(device),
										   	  std=configs['sensor_std'].to(device),
											  input_dim=len(configs['sensor_feats']),
										   	  hidden_dim=configs['hidden_dim'],
											  output_dim=configs['embedding_dim'],
											  normalization=configs['normalization']).to(device)
	dpr_model = dpr_models.DPRModel(image_encoder, sensor_encoder).to(device)

	dpr_model.load_state_dict(torch.load(configs['model_path'], weights_only=True, map_location=device))
	dpr_model.eval()

	dataset = ImageSensorDataset(image_data, sensor_data)
	data_loader = DataLoader(dataset, batch_size=configs['batch_size'])

	image_embeddings, sensor_embeddings = [], []
	for images, sensors in data_loader:
        # Forward pass
		image_emb, sensor_emb = dpr_model(images.to(device), sensors.to(device))
		
		image_embeddings.append(image_emb)
		sensor_embeddings.append(sensor_emb)

	image_embeddings_np = torch.cat(image_embeddings).detach().cpu().numpy()
	sensor_embeddings_np = torch.cat(sensor_embeddings).detach().cpu().numpy()

	similarities = 1 - pairwise_distances(image_embeddings_np, sensor_embeddings_np, metric='cosine')

	return similarities

if __name__ == '__main__':
	import argparse, os, pickle
	import pandas as pd
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', help='Input file [csv]')
	parser.add_argument('--output_file', help='Output file [csv]')
	parser.add_argument('--config_file', help='Config file [pkl]')
	args = parser.parse_args()

	with open(args.config_file, mode='rb') as f:
		configs = pickle.load(f)

	sensor_feats = configs['sensor_feats']

	# Load input file
	#df_input = pd.read_pickle(args.input_file)
	df_input = pd.read_json(args.input_file)
	df_input = load_categorical_data(df_input, configs)
	for col in sensor_feats:
		df_input[col] = df_input[col].fillna(0.)
	df_input['image'] = df_input['image'].map(lambda x: transforms.functional.resize(img=torch.from_numpy(np.array(x)).to(torch.float32), size=(224,224)))
	df_input['image'] = df_input['image'].map(lambda x: transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
	
	# Evaluate each topic
	df_output = pd.DataFrame()
	for topic_id in df_input['topic_id'].unique():
		df = df_input[df_input['topic_id']==topic_id].copy()
		N = len(df)
		print(f'topic: {topic_id}, Number of data: {N}')
		minute_ids = df['dmr_minute_id'].tolist()

		sensor_data = torch.from_numpy(df[sensor_feats].values).to(torch.float32)
		image_data = torch.from_numpy(np.stack(df['image'].values)).to(torch.float32)

		similarities = predict(configs, image_data, sensor_data)
		if 'img2sen' in topic_id:
			similarities = similarities[0]
		else:
			similarities = similarities.T[0]

		d = {'group_id':'ORG', 'run_id':'baseline', 'topic_id':topic_id, 'dmr_minute_id':minute_ids, 'score':similarities}
		df_output = pd.concat([df_output, pd.DataFrame(d)])

		df_output.to_csv(args.output_file, index=False)