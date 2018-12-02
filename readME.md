
This module is classification of images into 2 classes. I have encoded the class label as -

label 0 for YE358311_defects
label 1 for YE358311_Healthy

Structure of Module -
    --> src
    	contain all the source code.
    --> model
    	contain the trained model and graph from scratch
	--> inceptionV3_retrain
		contains retrained imagenet model and graph on the dataset provided

Details of each source code -
1. readImg.py - read all the images assign label to each image as per there class, resize it to 224 X 224 image and convert it to RGB. After converting images, it will write these data to data folder in tfrecords file. there is separate file for train and test set.

	>> python3 readImg.py

2. trainModel.py - this code trains the network. read the data form data folder, feed data to network and evaluate the model using test data.

	>> python3 trainModel.py

3. predict.py - predict the label of image. load the save model, read the image apply pre-processing, pass to network and return the predicted label. this code is called by runServer.py file. you can also use this code for prediction without using runServer file. Read the image in this file and pass to preProcess_img function, it returns the label of the image.

4. runService.py - this code deploys this module as a service, on 5000 port using POST method. it will run on /getLabel address on 5000 port.

	>> python3 runService.py

	after running this service, request on http://0.0.0.0:5000/getLabel variable name is img which accept the image and return the predicted label.

5. imgAug.py – for augmentation of image data, I have applied some image processing function to increase the number of images.

	>> python3 imgAug.py

6. img_preProcess.py - called by other codes, this code applies some image processing functions on image

7. tf_freezeGraph.py - for freezing the trained model and graph for production env.

	>> python3 tf_freezeGraph.py

8. retrain.py - for transfer learning, I have also used pertained inception ImageNet model and trained on the dataset provided. I just downloaded this script from tensorflow git repo and use it as guided in tensorflow guide. below is the command I have used for retraining -

	>> python3 retrain.py \
		--image_dir=dataSet_dir \
		--saved_model_dir=model_path \
		--bottleneck_dir=bottleneck_path \
		--output_graph=output_graph_path

	

