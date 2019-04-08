# Human & Non Human Classification
#### By Vaibhav Pelne

## The problem Statements

Given a set of labeled images of Human and Non Human, a machine learning model is to be learnt and later it is to be used to classify a set of new images as Human or Non Human.

## You can Download the DataSet from the link below:
http://pascal.inrialpes.fr/data/human/

### folder structure
##humandata<br>
--humantrain<br>
------Human<br>
------NonHuman<br>
--humanvalidation<br>
------Human<br>
------NonHuman<br>

## Dependencies
<ul>
    <li>Jupyter notebook</li> 
    <li>keras</li>  
    <li>Python 3.6</li>  
    <li>Numpy</li>  
</ul>

### Note: Install dependencies using conda : https://conda.io/en/latest/

## train and validation Split
Image training set contain 497 images for each category. I split those into 80% train and 20% means train Split each class images into 497 for train and 126 for validation.

## Network Parameter:

<ul>
    <li>Rectifier Linear Unit</li> 
    <li>Adam optimizer</li>  
    <li>Sigmoid on Final output</li>  
    <li>Binary CrossEntropy loss</li>  
</ul>

## Conclusion
The Architecture and parameter used in this network are capable to classified the given image human or non human on Validation Data which is pretty good. It is possible to Achieve more accuracy for that increase the size of data set .

on good system use following parameter
steps_per_erpoch = 8000,
epochs = 25,
validation_data=validation_generator,
std validation steps 2000)

for specific width and height of image:
img_width = 256
img_height = 256

classifier.add(Conv2D( 32,(3,3), input_shape=( img_width, img_height, 3 ), activation = 'relu'))


train_generator = train_datagen.flow_from_directory(
        'humandata/humantrain',
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'humandata/humanvalidation',
        #target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')