# Project Write-Up
People counter app is a project which is to be done as a part of Udacity Intel Edge AI for IoT Developers Nanodegree to fulfill its requirements.

People counter app is an application which is used to perform inference over a camera or video or image and detect number of people in the frame.

## Explaining Custom Layers
The list of supporting layers can be found out in the below link.

https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html

Any layer not in the list of above link is classified as a custom layer by the Model Optimizer.

The process behind converting custom layers is dependent on the original model framework. The process is different for different model frameworks.

In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.

For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. We will need Caffe on our system to do this option.

For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.

Custom layers need to be handled in the program because, if a model consists of any custom layers, we need to add extensions to the model optimizer as well as the inference engine while performing inference with our model.

There are a lot of pre-trained models available in frameworks like Tensorflow, Caffe, PyTorch, MXNet etc. I have used the Intel OpenVINO pre-trained model 'person-detection-retail-0013' model to do this project as the models which I tried were unable to handle false positive cases i.e., when a subject is idle continuously the models failed to detect the presence of a subject.

The steps to execute the project are as follows.

Selected Intel pre-trained Model: "person-detection-retail-0013"

Additional model details can be found in the link below.
https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html

1)	Step 1: Go to the downloader directory and download the model	
Use the following command to change the directory.
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
Download the model by using the command below.
sudo ./downloader.py --name person-detection-retail-0013 -o /home/workspace
cd /home/workspace

2)	Step 2:
Finally, use the below command to see the people counter app in action.
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm



## Comparing Model Performance

The difference between model accuracy pre- and post-conversion has a drastic change when comes to the Intel pre-trained model whereas the other models failed to detect a subject which is idle continuously and had many false positives.

The size of the .bin file for the Intel pre-trained model was just 2.8 MB. While the size of .bin files of the other models were way higher than this. Below are the sizes of the models.

SSD MobileNet V2 COCO         64.1 MB
SSDLite MobileNet V2 COCO     17.1 MB
SSD Inception V2 COCO         95.4 MB

The inference time of the Intel pre-trained model was very less when compared with the other 3 models considering the accuracy. The inference times for the person-detection-retail-0013, SSD MobileNetV2, SSDLite MobileNet V2 and SSD Inception V2 are 45 ms, 69 ms, 32 ms, 159 ms respectively.

Though the SSDLite MobileNet V2 model had a less inference time, the results for not reliable enough as they were not accurate.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are 
1) A librarian in schools/universities/colleges can track students sitting in the library for long hours and may ask them to leave when they sit for certain number of hours.
2) Number of votes casted by the people in the polling booth can be calculated for a specific area and voting percentages by area can be declared.

Each of these use cases would be useful because, librarian can find students who are spending long hours in the library and further the faculties of the students can know which students are not attending their classes. Also, the manual counting verification done at the polling booths in India to calculate voting percentages can eliminate the need of physical presence of a person.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows.

1) Lighting: Usually good lighting conditions will fetch better accuracy in people detection.
2) Model accuracy: If we want the model accuracy to be high then we might need high computation power, however if the user budget is low we might have to have some trade-off with accuracy of the model. The models with higher accuracy fetch better results.
3) Camera focal length/Image size: Having a high focal length fetches focussed images and thus yield high accuracy of the model. Image size effects the processing time. Higher accuracy will need more processing time and thus yield good results.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD MobileNet V2 COCO
  - [Model Source: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments.
      1. wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
      2. tar  -xvf  ssd_mobilenet_v2_coco_2018_03_29.tar.gz
      3. python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  - The model was insufficient for the app because it was having many false positives and was unable to detect a person when in frame continuously.
  - I tried to improve the model for the app by checking the detected person to be in the center of the frame and by decreasing the probility threshold.
  
- Model 2: SSDLite MobileNet V2 COCO
  - [Model Source: http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments.
      1. wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
      2. tar -xvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
      3. python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  - The model was insufficient for the app because it was having many false positives and was unable to detect a person when in frame continuously.
  - I tried to improve the model for the app by checking the detected person to be in the center of the frame and by decreasing the probility threshold.

- Model 3: SSD Inception V2 COCO
  - [Model Source: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments.
      1. wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
      2. tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz
      3. python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssd_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  - The model was insufficient for the app because it was having many false positives and was unable to detect a person when in frame continuously.
  - I tried to improve the model for the app by checking the detected person to be in the center of the frame and by decreasing the probility threshold

## References
Custom Layers, https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html