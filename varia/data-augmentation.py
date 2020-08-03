"""
Run this code in the folder where images are located

__author__      = "Neelanshi Varia"
"""


from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os


#list of data generators to be applied
datagen_list = [ImageDataGenerator(rotation_range=90), 
                ImageDataGenerator(brightness_range=[0.2,1.0]), 
                ImageDataGenerator(zoom_range=[0.5,1.0]), 
                ImageDataGenerator(rotation_range=90),
                ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2),
                ImageDataGenerator(horizontal_flip=True, vertical_flip=True)]

#folder to save augmented images
os.makedirs('aug_images')

#apply operation on all images in a folder
# imagePath = '/Users/neelanshi/Documents/Research_Projects/SAGE/data_augmentation/images'

for filename in os.listdir('.'):
    if filename.endswith(".png"):
        # load the image
        img = load_img(filename)

        # convert to numpy array
        data = img_to_array(img)

        #expand dimension to one sample
        samples = expand_dims(data, 0)

        # create image data augmentation generator
        for datagen in datagen_list:
            # prepare iterator
            it = datagen.flow(samples, batch_size=1, save_to_dir='aug_images', save_prefix='aug', save_format='png')
            # generate samples and plot
            for i in range(4):
                # define subplot
                # pyplot.subplot(220 + 1 + i)
                # generate batch of images
                batch = it.next()
                # convert to unsigned integers for viewing
                image = batch[0].astype('uint8')
                # plot raw pixel data
                # pyplot.imshow(image)
            # show the figure
            # pyplot.show()




######### not to use
# datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# datagen = ImageDataGenerator(zca_whitening=True)
# datagen = ImageDataGenerator(horizontal_flip=True) #combined with vertical flip
# datagen = ImageDataGenerator(width_shift_range=[-200,200]) #combined with height shift
# datagen = ImageDataGenerator(height_shift_range=0.5) #combined with width shift
#########

#### following are included in the datagen list
# datagen = ImageDataGenerator(rotation_range=90)
# datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
# datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
# datagen = ImageDataGenerator(rotation_range=90)
# datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
# datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)