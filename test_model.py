from u_net import UNet
import numpy as np
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from PIL import Image
import pdb
import os
from run_len import rlencode
import csv


def test_images(path, write=False, mask_path=None):

	IMAGE_SIZE = 224
	ORIGINAL_SIZE = (1918, 1280)
	net_depth = 2

	model = UNet((IMAGE_SIZE,IMAGE_SIZE,3), depth=net_depth)
	model.load_weights('weight2.hdf5')

	with open("test.csv", "w", newline='') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(["img", "rle_mask"])

		dice_coefs = []
		for subdir, dirs, files in os.walk(path):
			for file in files:
				name = os.path.join(subdir, file)
				im = load_img(name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
				#Normalize
				im = img_to_array(im)/255.0
				prediction = model.predict(np.array([im]))[0]
				# prediction = np.rint(prediction) #TODO: reint before resize?
				im = array_to_img(prediction*255)
				im = im.resize(ORIGINAL_SIZE)
				result = img_to_array(im).flatten()/255.0
				result = np.rint(result) #TODO: rint is the best?
				# print(result.shape)
				# print(np.unique(result))
				# im = array_to_img(result*255)
				# im.show()
				if (write):
					write_result(result, file, writer)
				if (mask_path):
					dice_coefs.append(calculate_dice_coef(result, mask_path, file))
	if (len(dice_coefs) > 0):
		print("dice coef:", np.sum(np.array(dice_coefs)) / len(dice_coefs))




def write_result(result, file, writer):
	starts, lengths, values = rlencode(result)
	result = []
	for start, lenght, value in zip(starts, lengths, values):
		if value:
			result.extend([start, lenght])
	writer.writerow([file, ' '.join(map(str, result))])
	# pdb.set_trace()


def calculate_dice_coef(result, mask_path, file):
	file = file.split('.')[0] + '_mask.gif'
	label = load_img(mask_path + file, grayscale=True)
	# label.show()
	label = img_to_array(label)/255
	label = label.flatten()
	dice_coef = 2 * np.sum(np.multiply(result,label)) / (np.sum(result) + np.sum(label))

	# print(np.sum(np.abs(result-label)))
	# print(dice_coef)

	return dice_coef



# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--test", required=True,
# 	help="Test folder")
# ap.add_argument("-m", "--mask", type=str, default="",
# 	help="Masks for test folder")
# args = vars(ap.parse_args())

# test_path = "D:/Data/carvana/train/validation/folder/"
# mask_path = "D:/Data/carvana/train_masks/validation_masks/folder/"
# test_images(test_path, write=False, mask_path=mask_path)

test_path = "D:/Data/carvana/test/"
test_images(test_path, write=True)
