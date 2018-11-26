# -*- coding: utf-8 -*-

from pandas import DataFrame
from pandas import concat
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from PIL import Image

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def SerieToImage(timeseries, image_size = 24, test_partition=0.33, validation_size=0.2, step=1):
    timeseries =  preprocessing.minmax_scale(timeseries, feature_range=(0, 1))
    
    data_train, data_test = train_test_split(timeseries, test_size=test_partition, shuffle=False)
    
    TrainImageData = imageDataBank(image_size, data_train, step)
    TestImageData = imageDataBank(image_size, data_test, step)
    
    X_imageTrain, y_imageTrain = makeDataSets(TrainImageData)
    X_imageTest, y_imageTest = makeDataSets(TestImageData)
    
    return X_imageTrain, y_imageTrain, X_imageTest, y_imageTest
    
def imageDataBank(image_size, data, step):
    arrayInputTarget = []
    
    for i in range (0, (len(data) - image_size - step) + 1 ):

        input = data[ i: i + image_size]

        target = data[ i + image_size + step - 1]

        input = map(lambda x : math.acos(x), input)
        aux = list(input)
        imageData = SerieToImageData(aux)

        target = np.array(target)

        arrayInputTarget.append({'image' : imageData, 'target' : target})

    return arrayInputTarget

def SerieToImageData(original):
    imageArray = []

    for i in original:
        for j in original:

            ac = math.cos(i + j)

            imageArray.append(ac)
            
    imageArray = preprocessing.minmax_scale(imageArray, feature_range=(0, 255))
    
    dataImage = np.array(imageArray).reshape(len(original), len(original))

    return dataImage

def makeDataSets(image_data):
    imageData = []
    targetData = []
    for value in image_data:
        aux1 = value['image']
        aux2 = value['target']
        imageData.append(aux1)
        targetData.append(aux2)
    return imageData, targetData

def imageDataToJPG(imageBank, target, folderPath):
    image_number = 0
    relation = []
    for datos in imageBank:
        image_heigh = datos.shape[0]
        image_width = datos.shape[1]
        image = Image.new("L", (image_heigh, image_width))
        image.putdata(datos.flatten())
        image_name = str(image_number+1) + '.jpeg'
        image.save(folderPath+"/"+image_name,"JPEG")
        relation.append([image_name, target[image_number][0]])
        image_number+=1
    df = pd.DataFrame(relation)
    df.to_csv(folderPath+"/relacion.csv")