"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
#global person_detected, dur
#person_detected = False
global dur
dur = 0

def draw_boxes(frame, result, args, width, height):
    """
    Draw bounding boxes onto the frame
    """
    global dur, person_in_frame
    person_in_frame = False
    counter = 0
    leave_room = ""
    
    #Frame Center co-ordinates
    f_center_x = int(width/2)
    f_center_y = int(height/2)
    
    for box in result[0][0]: #Output shape is 1x1xNx7
        person = box[1]
        conf = box[2]
        tm = time.time()
        time_diff = tm - dur
        if conf >= prob_threshold and person == 1:
            dur = time.time()
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            
            if (xmin < f_center_x and xmax > f_center_y and ymin < f_center_x and ymax > f_center_y):
                person_in_frame = True
                
            if person_in_frame and (0 < time_diff < 5):
                leave_room = "Please leave the room -->"
            cv2.putText(frame, leave_room, (480, 30), cv2.FONT_HERSHEY_COMPLEX,  0.5, (255, 0, 0), 1) 
                
            if person_in_frame and time_diff > 5:
                counter+=1
            
    return frame, counter



def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    plugin = Network()
    # Set Probability threshold for detections
    global prob_threshold
    prob_threshold = args.prob_threshold
    
    ### Variables used for inference
    single_image_mode = False  # Flag for single images
    req_id = 0
    start_time = 0
    last_count = 0
    total_count = 0
    time_lag = 0
    duration = 0    

    ### Check for input type: webcam, image
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True


    ### TODO: Load the model through `infer_network` ###
    plugin.load_model(args.model,args.device,args.cpu_extension,req_id)
    net_input_shape = plugin.get_input_shape() #Input size is [1x3x300x300]

    ### TODO: Handle the input stream ###
    ##Open video capture
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    ##Shape of input
    
    w = int(cap.get(3))
    h = int(cap.get(4))
    
    out = cv2.VideoWriter('inference_output.mp4', 0x00000021, 30, (w,h))
    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read() #Frame size is 432x768x3
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame,(net_input_shape[3],net_input_shape[2])) #Frame size is (300,300,3)
        p_frame = p_frame.transpose((2,0,1)) #Frame size: (3, 300, 300)
        p_frame = p_frame.reshape(1,*p_frame.shape) #Frame size: (1, 3, 300, 300)
   
        ### TODO: Start asynchronous inference for specified request ###
        start_infer = time.time()
        plugin.exec_net(req_id,p_frame)
       

        ### TODO: Wait for the result ###
        if plugin.wait(req_id) == 0:
            det_time = time.time() - start_infer
            result = plugin.get_output(req_id)
            ## Update the frame to include detected bounding boxes
            frame, current_count = draw_boxes(frame,result,args,w,h)
                
            inf_time_msg = "Inference time: {:.3f}ms".format(det_time*1000)
            ttl_cnt = "Total count: {:}".format(total_count)
            cv2.putText(frame, inf_time_msg, (15, 15), cv2.FONT_HERSHEY_SIMPLEX,  0.6, (0, 255, 0), 1)   
            cv2.putText(frame, ttl_cnt, (15, 31), cv2.FONT_HERSHEY_SIMPLEX,  0.6, (0, 255, 0), 1)
            
            ## Write out the frame
            out.write(frame)

            ### TODO: Get the results of the inference request ###

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###

            if current_count <= last_count:
                duration = int(time.time() - start_time)
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###             
            else:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person",json.dumps({"total": total_count}))
                client.publish(ttl_cnt)
            
            if total_count >= current_count and current_count != 0:
                client.publish("person/duration",json.dumps({"duration": duration }))
            if person_in_frame == True:
                current_count = 1
            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count

            

            ## Break if escape key is pressed
            if key_pressed == 27:
                break
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_img.jpg',frame)
            
        
    ### Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
