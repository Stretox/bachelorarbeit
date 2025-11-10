import numpy as np
import json
import re
from typing import NamedTuple

class Camera(NamedTuple):
    name:str
    extrinsic:np.ndarray
    fx:float
    fy:float
    cx:float
    cy:float
    u_id:int = -1
    height:int = 480
    width:int = 720

def get_json_cameras(json_file_path: str):

    '''
    Get Cameras as a List from calibration_dome.json or a similar file
    
    '''

    data:dict = {}

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    cams:list = []

    camera_centers = []
    for camera in data['cameras']:
        cam_center = np.array(camera['extrinsics']['view_matrix']).reshape(4,4)[:3, 3] # [x,y,z]
        camera_centers.append(cam_center)

    scene_center = np.mean(np.stack(camera_centers), axis=0)
    # print(f"Camera centers before: {scene_center}")

    new_camera_centers=[]

    for camera in data['cameras']:
        name = camera['camera_id']
        uid = int(re.findall('\d+', name)[0])
        extr = np.array(camera['extrinsics']['view_matrix']).reshape(4,4)
        fx = np.array(camera['intrinsics']['camera_matrix'][0])
        fy = np.array(camera['intrinsics']['camera_matrix'][4])
        cx = np.array(camera['intrinsics']['camera_matrix'][2])
        cy = np.array(camera['intrinsics']['camera_matrix'][5])
        height = int(camera['intrinsics']['resolution'][1])
        width = int(camera['intrinsics']['resolution'][0])

        extr[:3,3] = extr[:3,3] - scene_center
        new_center = np.linalg.inv(extr)[:3, 3]
        new_camera_centers.append(new_center)

        camera = Camera(name=name, u_id=uid, fx = fx, fy = fy, cx = cx, cy = cy, width = width, height = height, extrinsic = extr)

        cams.append(camera)

    return cams