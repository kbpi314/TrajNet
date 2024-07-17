### import libraries
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# disable warnings, use w caution
import warnings
warnings.filterwarnings('ignore')

# project specific libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping


### set (local) path
# path = '/Users/KevinBu/Desktop/clemente_lab/Projects/iclust_paper/'
path = '/sc/arion/projects/clemej05a/kevin/cnn/'

from PIL import Image

# Function to convert image to pixel matrix
def image_to_pixel_matrix(image_path):
    # Open an image file
    img = Image.open(image_path)
    
    # Convert the image to grayscale (optional)
    img_gray = img.convert('L')
    
    # Get the pixel values as a list of tuples
    pixel_values = list(img_gray.getdata())
    
    # Convert the pixel values list to a matrix
    width, height = img_gray.size
    pixel_matrix = [pixel_values[i * width:(i + 1) * width] for i in range(height)]
    
    return pixel_matrix


## import glob

rows = []
ids = []
files = []
#for i in glob.glob(path + 'inputs/sim_0.04_0_train/trajs/*.jpg'):
# for i, image in enumerate(glob.glob(path + 'inputs/jobs87_pca_std/results/sim_0.04*/trajs/*.jpg')):
from pathlib import Path

directory = Path(path + 'inputs/jobs87_pca_std/results/')
for subdir in directory.glob('sim_0.04*'):
    #print(str(subdir))
    subdir = Path(str(subdir) + '/trajs/')
    #print(str(subdir))
    #.glob('*'))
    #print(list(subdir.glob('*.jpg')))
    files = files + list(subdir.glob('*.jpg'))

# print(files)

for i, image in enumerate(files):
    image = str(image)
    pixel_matrix = image_to_pixel_matrix(image)
    
    # flatten pixel matrix
    flat_pm = flattened_list = [item for sublist in pixel_matrix for item in sublist]

    # make row
    row = pd.Series(data=np.array(flat_pm), index=['pixel' + str(j) for j in range(len(flat_pm))])

    # grab id
    id = 'batch' + str(i) + '_' + image.split('sims_')[1].split('_')[0]

    # make cols
    ids.append(id)
    rows.append(row)

df = pd.DataFrame(dict(zip(ids,rows)))
df = df.T
    
df['label'] = df.index.map(lambda x: int((int(x.split('_')[-1])-1)/4))

df.head()


# try new 
train = df.iloc[:-100,:]
test = df.iloc[-100:,:-1]

# X is the pixels and Y is the image labels
X = train.iloc[:,:-1]
Y = train.iloc[:,-1]

#splitting dataframe using train_test_split
seed = 0
x_train , x_test , y_train , y_test = train_test_split(X, Y , test_size=0.1, random_state=seed)

#first param in reshape is number of examples. We can pass -1 here as we want numpy to figure that out by itself

#reshape(examples, height, width, channels)
x_train = x_train.values.reshape(-1, 285, 281, 1)
x_test = x_test.values.reshape(-1, 285, 281, 1)
# df_test=df_test.values.reshape(-1,28,28,1)

# create new data to make training more robust
datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
# df_test = df_test.astype("float32")/255

#fitting the ImageDataGenerator
datagen.fit(x_train)

#notice num_classes is set to 10 as we have 10 different labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

test.head()


### specify layers of CNN
#Conv2d data_format parameter we use 'channel_last' for imgs
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last',
                 input_shape=(285,281,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#Optimizer
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999 )

#Compiling the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

#Define LearningRateScheduler
reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

print('Done')



### model training
#defining these prior to model to increase readability and debugging
batch_size = 32 # 64
epochs = 5 # 59

# Fit the Model
history = model.fit(datagen.flow(x_train, y_train, batch_size = batch_size), epochs = epochs, 
                              validation_data = (x_test, y_test), verbose=1, 
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks = [reduce_lr]) #left out early_stopping parameter as it gets better accuracy


plt.figure(figsize=(13, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'])
plt.grid()
# plt.show()
plt.savefig(path + 'outputs/inhouse.pdf')

print('cnn done')
