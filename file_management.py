# this class is responsible for managing all the os file things
import math
import random

import matplotlib.pyplot as plt
import os

import skimage.measure
import trimesh

import binvox_rw
import numpy as np

PATH = os.getcwd()
# Set the directory path
dir_3d_path = str(PATH + "\\Data\\3d\\")
dir_img_path = str(PATH + "/Data/imgs/")
dir_gen_3d_path = str(PATH + "/Data/gen/")


# to visualize the 3d objects
def visualize_3d(obj):
    obj.show()


# Rename all files in the directory
def renameFiles(path):
    path = (PATH + path)

    # iterate over all files in the directory
    l = 0
    for filename in os.listdir(path):
        # get the full path of the file
        full_path = os.path.join(path, filename)

        # add a prefix to the filename
        new_filename = str(l) + ".binvox"

        # rename the file
        os.rename(full_path, os.path.join(path, new_filename))
        l += 1


# This method is heavy to and takes time to plot
def visualize3DInPlotter(obj):
    voxels = obj.data.astype(np.bool_)
    # plot the voxels using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # set limits so that the aspect ratio is equal
    ax.voxels(voxels)
    plt.savefig(dir_img_path + 'image.png', transparent=True, dpi=500)
    # display the plot
    plt.show()


def load3DFile(name, Generated):
    if Generated:
        path = dir_gen_3d_path
    else:
        path = dir_3d_path
    # outputs a 3d formatted file
    with open(path + name + ".binvox", 'rb') as file:
        return binvox_rw.read_as_3d_array(file)


def load3DModel(path):
    # outputs a 3d formatted file
    with open(path, 'rb') as file:
        return binvox_rw.read_as_3d_array(file)


# take in array and visualize it
def visualizeTrimeshLoad(voxel_model):
    # Convert coordinates to a numpy array
    # Create trimesh object
    # extract the surface mesh using marching cubes
    vertices, faces, _, _ = skimage.measure.marching_cubes(voxel_model, level=0.5, spacing=(1, 1, 1))
    # create a trimesh object from the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # visualize the mesh
    mesh.show()


# retrieve a random screenshot from 5 positions of an object
def randomizedRotation(i, center):
    try:
        arr = [[45, 20, 4, 45, center[0] / 2, -center[1], center[2]],
               [45, 20, -2, 135, center[0] / 2, -center[1], center[2] / 2],
               [45, 20, -2, 70, center[0] / 2, -center[1], center[2]],
               [45, 30, -20, 270, center[0] / 2, center[1] / 2, center[2]],
               [-45, 30, 20, 320, center[0], center[1] / 2, center[2]],
               [-85, -70, -20, 50, center[0], center[1], center[2]],
               [-85, 200, 120, 50, center[0], center[1], center[2]],
               [80 * 4, 80 * 4, -20, 90, center[0], center[1], center[2]],
               [-80 * 4, 0, 0, 90, center[0], center[1], center[2]]]
        rotate = trimesh.transformations.rotation_matrix(
            angle=np.radians(arr[i][3]),
            direction=[arr[i][0], arr[i][1], arr[i][2]],  # [x,y,z]
            point=np.array([arr[i][4], arr[i][5], arr[i][6]])
        )
        return rotate
    except:
        raise Exception


# fn to return an image given the object name
def get_screenShot(name):
    voxel_model = load3DFile(str(name), False)

    vertices, faces, _, _ = skimage.measure.marching_cubes(voxel_model.data, level=0.5, spacing=(1, 1, 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    center = scene.centroid

    rotate = randomizedRotation(random.randint(0, 8), center)

    camera_old, _geometry = scene.graph[scene.camera.name]
    camera_new = np.dot(rotate, camera_old)

    # apply the new transform
    scene.graph[scene.camera.name] = camera_new

    img = scene.save_image(resolution=(640, 480))
    scene.show()

    return img


# fn to save image
def saveImg(img, name):
    with open(dir_img_path + str(name) + ".png", "wb") as f:
        f.write(img)
        f.close()


def saveModel(model, name):
    # save the binvox file
    with open(dir_gen_3d_path + name + '.binvox', 'wb') as f:
        binvox_rw.write(model, f)


# simple array as an input of false and trues
def saveVoxel(data, name):
    dims = (32, 32, 32)
    translate = (0.0, 0.0, 0.0)
    scale = 1.0

    # Create the Voxels object
    voxels = binvox_rw.Voxels(data, dims, translate, scale, 'xyz')

    with open(dir_gen_3d_path + name + '.binvox', 'wb') as f:
        # Write the header
        # Write the binary data
        voxels.write(f)


visualizeTrimeshLoad(load3DModel(dir_gen_3d_path + "lol.binvox").data)

# the fn is responsible for copying the 3d models that have a screenshot to another repository for dataset perpose
# import shutil
#
# source_dir = dir_img_path
# destination_dir = dir_3d_path
# third_dir = str(PATH + "/Data/temp/")
#
# # Iterate through files in source folder
# for filename in os.listdir(source_dir):
#     source_path = os.path.join(source_dir, filename)
#     name = filename[:len(filename) - 4]
#     name = name + ".binvox"
#     # Check if file exists in destination folder
#     if os.path.exists(os.path.join(destination_dir, name)):
#         print(name)
#         # If it does, copy it to third folder
#         shutil.copy(os.path.join(destination_dir, name),third_dir)
