import numpy as np
from tensorflow.keras.models import load_model

class Wrapper():
	def __init__(self, model_path) -> None:
		self.model = load_model(model_path)
		self.model.compile(
			optimizer='adam',
			loss='bce',
		)
		print(self.model.summary())

	def predict(self, img):
		pred = self.model.predict(np.expand_dims(img, axis=0))[0]
		return {
		'Atelectasis':pred[0],
		'Cardiomegaly':pred[1],
		'Consolidation':pred[2],
		'Edema':pred[3],
		'Enlarged Cardiomediastinum':pred[4],
		'Fracture':pred[5],
		'Lung Lesion':pred[6],
		'Lung Opacity':pred[7],
		'No Finding':pred[8],
		'Pleural Effusion':pred[9],
		'Pleural Other':pred[10],
		'Pneumonia':pred[11],
		'Pneumothorax':pred[12],
		'Support Devices':pred[13],
		}