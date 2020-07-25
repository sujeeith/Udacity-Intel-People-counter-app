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

There are a lot of pre-trained models available in frameworks like Tensorflow, Caffe, PyTorch, MXNet etc. I chose the Tensorflow's 'SSD MobileNet V2 COCO' model to do this project.
The steps to execute the project are documented in the following file.
https://github.com/sujeeith/Udacity-Intel-People-counter-app/blob/master/People_counter_app_execution_steps.docxs


## Comparing Model Performance

I have gone through the MobileNetV2 research paper (https://arxiv.org/pdf/1801.04381.pdf) to compare models before and after conversion to Intermediate Representations.

The difference between model accuracy pre- and post-conversion was negligible and almost same accuracy was obtained after conversion. Although, this model detects less number of people with more accuracy but might fail to detect a person who is idle continuously.

The size of the model pre- and post-conversion was almost over 2 MB lesser. The size of the model's .pb file was 66.4 MB while the size of the generated .bin file after conversion was 64.1 MB.

The inference time of the model pre- and post-conversion was 75 ms and 69 ms respectively.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are 
1) A librarian in schools/universities/colleges can track students sitting in the library for long hours and may ask them to leave when they sit for certain number of hours.
2) Number of votes casted by the people in the polling booth can be calculated for a specific area and voting percentages by area can be declared.

Each of these use cases would be useful because, librarian can find students who are spending long hours in the library and further the faculties of the students can know which students are not attending their classes. Also, the manual counting verification done at the polling booths in India to calculate voting percentages can eliminate the need of physical presence of a person.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

## References
1) Custom Layers, https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html
2) MobileNet V2, https://arxiv.org/pdf/1801.04381.pdf   
