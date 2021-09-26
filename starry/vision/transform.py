
import numpy as np
import cv2



def half_down(feature, label):
	h,w,c = feature.shape[-3:]
	h = h // 2
	w = w // 2
	b = feature.shape[0]
	down = np.zeros((b, h, w, c), dtype=feature.dtype)
	for i in range(b):
		#down[i] = cv2.resize(feature[i], (w, h))
		down[i] = cv2.pyrDown(feature[i]).reshape((h, w, c))

	return down, label

def tar_half_down(feature, label):
	h,w,c = label.shape[-3:]
	h = h // 2
	w = w // 2
	b = label.shape[0]
	down = np.zeros((b, h, w, c), dtype=label.dtype)
	for i in range(b):
		#resized = cv2.resize(label[i], (w, h))
		resized = cv2.pyrDown(label[i])
		down[i] = np.reshape(resized, (h, w, c))	# reshape to workaround single channel missing issue in OpenCV

	return feature, down

def mono(feature, label):
	monos = []
	for temp in feature:
		gray = temp
		if len(temp.shape) == 3:
			if temp.shape[2] == 3:
				gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
				gray = np.expand_dims(gray, -1)
				gray = (gray/255.0).astype(np.float32)
			elif gray.dtype == np.uint8:
				gray = (gray/255.0).astype(np.float32)
		monos.append(gray)
	monos = np.stack(monos)
	return monos, label

def normalize(feature, label):
	result = []
	for temp in feature:
		layer = (temp / 255.0).astype(np.float32)
		result.append(layer)
	result = np.stack(result)

	return result, label

def invert (feature, label):
	return 1 - feature, label

def tar_stdgray(feature, label):
	label = (label / 255.0).astype(np.float32)
	return feature, label

def img_std_nor(feature, label):
	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	feature = feature.astype(np.float32)
	feature = (feature / 255.0 - mean) / std
	return feature,label

def hwc2chw(feature, label):
	feature = np.moveaxis(feature, -1, 1)
	return feature, label

def target_hwc2chw(feature, label):
	label = np.moveaxis(label, -1, 1)
	return feature, label

def to_float32(feature, label):
	return feature.astype(np.float32), label

def nan_check(feature, label):
	if(np.isnan(feature).any()):
		raise ValueError('Input feature han nan value.')
	if(np.isnan(label).any()):
		raise ValueError('Input label han nan value.')
	return feature, label

def jitter_y(feature, label):
	y = round(np.random.normal(scale = 1))

	if y > 0:
		feature[:, :, y:, :] = feature[:, :, :-y, :]
		label[:, :, y:, :] = label[:, :, :-y, :]
	elif y < 0:
		feature[:, :, :y, :] = feature[:, :, -y:, :]
		label[:, :, :y, :] = label[:, :, -y:, :]

	return feature, label


def transposeBlur (image, kernel):	# (channel, height, width)
	image = np.transpose(image, (1, 2, 0))
	layer = cv2.GaussianBlur(image, kernel, 0)
	if len(image.shape) < 3:
		image = np.expand_dims(image, -1)
	image = np.transpose(image, (2, 0, 1))

	return image

# transform on label[:, 1, :, :], K[y] = X[y] - X[y-1]
def gradient_y(feature, label):
	layer = label[:, 1, :, :]

	layer_1 = layer[:, :-1, :]
	layer_0 = layer[:, 1:, :]
	layer[:, 1:, :] = layer_0 - layer_1
	layer[:, 0, :] = layer[:, 1, :]

	#layer = transposeBlur(layer, (13, 13))

	label[:, 1, :, :] = layer

	return feature, label


class TransWrapper:
	def __init__(self, fn):
		self.trans_fn = fn

	def __call__(self, feature, label):
		feature, label = self.trans_fn(feature, label)
		return feature, label


class Half_Down(TransWrapper):
	def __init__(self):
		super().__init__(half_down)

class Tar_Half_Down(TransWrapper):
	def __init__(self):
		super().__init__(tar_half_down)

class Img_std_N(TransWrapper):
	def __init__(self):
		super().__init__(img_std_nor)

class Mono(TransWrapper):
	def __init__(self):
		super().__init__(mono)

class Normalize(TransWrapper):
	def __init__(self):
		super().__init__(normalize)

class Invert(TransWrapper):
	def __init__(self):
		super().__init__(invert)

class Tar_STDgray(TransWrapper):
	def __init__(self):
		super().__init__(tar_stdgray)

class HWC2CHW(TransWrapper):
	def __init__(self):
		super().__init__(hwc2chw)

class Tar_HWC2CHW(TransWrapper):
	def __init__(self):
		super().__init__(target_hwc2chw)

class To_Float32(TransWrapper):
	def __init__(self):
		super().__init__(to_float32)

class Nan_Check(TransWrapper):
	def __init__(self):
		super().__init__(nan_check)

class JitterY(TransWrapper):
	def __init__(self):
		super().__init__(jitter_y)

class Tar_GradientY(TransWrapper):
	def __init__(self):
		super().__init__(gradient_y)


class Composer():
	def __init__(self, trans):
		self.trans_name = trans

	def __call__(self, feature, target):
		newf, newl = feature, target
		for name in self.trans_name:
			trans = transform_factory.get_transfn(name)
			newf, newl = trans(newf, newl)

		return newf, newl


class TransformFactory():
	def __init__(self):
		trans_classes = [
			Half_Down, Img_std_N, HWC2CHW,
			Tar_HWC2CHW, Tar_Half_Down, Tar_STDgray,
			To_Float32, Nan_Check,
			Mono, Normalize, Invert,
			JitterY, Tar_GradientY,
		]
		self.trans_dict = dict([(c.__name__, c()) for c in trans_classes])

	def get_transfn(self, name):
		return self.trans_dict[name]


transform_factory = TransformFactory()
