import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label, stuff_cat_ids, get_frame_panoptic_instances,\
    get_panoptic_instances_stats
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap
import threading

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("nuScenes dev-kit only supports Python version 3.")

from nuscenes.nuscenes import NuScenes, NuScenesExplorer

class My_NuScenes(NuScenes):

    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        super().__init__(version=version, dataroot=dataroot, verbose=verbose)
        self.explorer = My_NuScenesExplorer(self)

    def render_pointcloud_in_scene(self, sample_token: str, 
                                   dot_size: int = 5, 
                                   pointsensor_channel: list = None,
                                   camera_channel: list = None, 
                                   out_path: str = None,
                                   render_intensity: bool = False,
                                   show_lidarseg: bool = False,
                                   filter_lidarseg_labels: List = None,
                                   show_lidarseg_legend: bool = False,
                                   verbose: bool = True,
                                   lidarseg_preds_bin_path: str = None,
                                   show_panoptic: bool = False,
                                   show_both_modality: bool = False) -> None:
        self.explorer.render_pointcloud_in_scene(sample_token, dot_size, pointsensor_channel=pointsensor_channel,
                                                 camera_channel=camera_channel, out_path=out_path,
                                                 render_intensity=render_intensity,
                                                 show_lidarseg=show_lidarseg,
                                                 filter_lidarseg_labels=filter_lidarseg_labels,
                                                 show_lidarseg_legend=show_lidarseg_legend,
                                                 verbose=verbose,
                                                 lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                 show_panoptic=show_panoptic,
                                                 show_both_modality=show_both_modality)
        
    def render_sample_data_new(self, sample_data_token: str, with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY, axes_limit: float = 40, ax: Axes = None,
                           nsweeps: int = 1, out_path: str = None, underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           show_lidarseg: bool = False,
                           show_lidarseg_legend: bool = False,
                           filter_lidarseg_labels: List = None,
                           lidarseg_preds_bin_path: str = None, verbose: bool = True,
                           show_panoptic: bool = False) -> None:
        self.explorer.render_sample_data_new(sample_data_token, with_anns, box_vis_level, axes_limit, ax, nsweeps=nsweeps,
                                         out_path=out_path,
                                         underlay_map=underlay_map,
                                         use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                         show_lidarseg=show_lidarseg,
                                         show_lidarseg_legend=show_lidarseg_legend,
                                         filter_lidarseg_labels=filter_lidarseg_labels,
                                         lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                         verbose=verbose,
                                         show_panoptic=show_panoptic)
    def render_sample_new(self, sample_token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      sample_path: str = None,
                      show_lidarseg: bool = False,
                      filter_lidarseg_labels: List = None,
                      lidarseg_preds_bin_path: str = None,
                      verbose: bool = True,
                      show_panoptic: bool = False) -> None:
        self.explorer.render_sample_new(sample_token, box_vis_level, nsweeps=nsweeps, out_path=out_path, sample_path=sample_path,
                                    show_lidarseg=show_lidarseg, filter_lidarseg_labels=filter_lidarseg_labels,
                                    lidarseg_preds_bin_path=lidarseg_preds_bin_path, verbose=verbose,
                                    show_panoptic=show_panoptic)
        
class My_NuScenesExplorer(NuScenesExplorer):

    def __init__(self, nusc: My_NuScenes):
        self.nusc = nusc
        self.radar = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 
                                                                'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        self.camera = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 
                                                           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    def render_pointcloud_in_scene(self,
                                   sample_token: str,
                                   dot_size: int = 5,
                                   pointsensor_channel: list = None,
                                   camera_channel: list = None, 
                                   out_path: str = None,
                                   render_intensity: bool = False,
                                   show_lidarseg: bool = False,
                                   filter_lidarseg_labels: List = None,
                                   ax: Axes = None,
                                   show_lidarseg_legend: bool = False,
                                   verbose: bool = True,
                                   lidarseg_preds_bin_path: str = None,
                                   show_panoptic: bool = False,
                                   show_both_modality: bool = False):
        """
        Scatter-plots a pointcloud on top of image.
        :param sample_token: Sample token.
        :param dot_size: Scatter plot dot size.
        :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param out_path: Optional path to save the rendered figure to disk.
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
        :param ax: Axes onto which to render.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param verbose: Whether to display the image in a window.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        if show_lidarseg:
            show_panoptic = False
        sample_record = self.nusc.get('sample', sample_token)

        if pointsensor_channel is None:
            pointsensor_channel = self.radar
        if camera_channel is None:
            camera_channel = self.camera

        # Here we just grab the front camera and the point sensor.
        pointsensor_token = [sample_record['data'][pointsensor_channel[i]] for i in range(len(pointsensor_channel))]
        camera_token = [sample_record['data'][camera_channel[i]] for i in range(len(camera_channel))]

        # mapping pointcloud to images (able to map RADAR and LiDAR together)
        if show_both_modality == True:
            points_result = {}
            coloring_result = {}
            pointsensor_channel_L = ['LIDAR_TOP']
            pointsensor_token_L = [sample_record['data'][pointsensor_channel_L[i]] for i in range(len(pointsensor_channel_L))]
            points_result_L, coloring_result_L, im_result = self.map_multiview_pointcloud_to_image(pointsensor_token_L,
                                                                                           pointsensor_channel_L,
                                                                                           camera_token,
                                                            camera_channel=camera_channel,
                                                            render_intensity=render_intensity,
                                                            show_lidarseg=show_lidarseg,
                                                            filter_lidarseg_labels=filter_lidarseg_labels,
                                                            lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                            show_panoptic=show_panoptic)
            points_result_R, coloring_result_R, im_result = self.map_multiview_pointcloud_to_image(pointsensor_token,
                                                                                           pointsensor_channel,
                                                                                           camera_token,
                                                            camera_channel=camera_channel,
                                                            render_intensity=render_intensity,
                                                            show_lidarseg=show_lidarseg,
                                                            filter_lidarseg_labels=filter_lidarseg_labels,
                                                            lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                            show_panoptic=show_panoptic)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(2, 3, figsize=(24, 9))
            
            # plot axes
            index_ax = [5,0,1,4,3,2]
            flip_list = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
            for i, ax in enumerate(ax.flat):
                # image
                channel = camera_channel[index_ax[i]]
                img = im_result[channel]
                ax.imshow(img)
                # LiDAR point cloud
                x_L = points_result_L[channel][0, :]
                y_L = points_result_L[channel][1, :]
                c_L = coloring_result_L[channel]
                ax.scatter(x_L,y_L,c=c_L,cmap='hot', s=3, edgecolor='black', linewidth=0.1)
                # RADAR point cloud
                # This print shows points per each camera frame
                # print(len(points_result_R[channel][0]))
                if points_result_R[channel][0] != []:
                    x_R = points_result_R[channel][0, :]
                    y_R = points_result_R[channel][1, :]
                    c_R = coloring_result_R[channel]
                    # flip
                    if channel in flip_list:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        x_R = img.size[0]-x_R
                    ax.scatter(x_R,y_R,c=c_R,cmap='hot', s=15, marker='*', edgecolor='black', linewidth=0.1)
                else:
                    print('no radar point cloud in this (sample token, sample timestamp, image): ', 
                          '(', sample_record['token'], sample_record['timestamp'], ',', channel, ')')
                    print('--------------------------------------------------------------------')
                ax.axis('off')
            
            plt.subplots_adjust(wspace=-0.048, hspace=-0.027)

            if out_path is not None:
                plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
            if verbose:
                plt.show()
            plt.close()

        else:
            points_result, coloring_result, im_result = self.map_multiview_pointcloud_to_image(pointsensor_token,
                                                                                           pointsensor_channel,
                                                                                           camera_token,
                                                            camera_channel=camera_channel,
                                                            render_intensity=render_intensity,
                                                            show_lidarseg=show_lidarseg,
                                                            filter_lidarseg_labels=filter_lidarseg_labels,
                                                            lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                            show_panoptic=show_panoptic)
        
        # print('points_result: ', points_result)
        # print('coloring_result: ', coloring_result)
        # print('im_result: ', im_result)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(2, 3, figsize=(24, 9))
                
            #     if lidarseg_preds_bin_path:
            #         fig.canvas.set_window_title(sample_token + '(predictions)')
            #     else:
            #         fig.canvas.set_window_title(sample_token)
            # else:  # Set title on if rendering as part of render_sample.
            #     ax.set_title("render_pointcloud_in_scene")
            
            # plot axes
            index_ax = [5,0,1,4,3,2]
            flip_list = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
            for i, ax in enumerate(ax.flat):
                # image
                channel = camera_channel[index_ax[i]]
                img = im_result[channel]
                # point cloud
                x = points_result[channel][0, :]
                y = points_result[channel][1, :]
                c = coloring_result[channel]
                # flip
                if channel in flip_list:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    x = img.size[0]-x
                ax.imshow(img)
                if 'LIDAR_TOP' not in pointsensor_channel:
                    ax.scatter(x,y,c=c,cmap='hot', s=dot_size, marker='*', edgecolor='black', linewidth=0.2)
                else:
                    ax.scatter(x,y,c=c,cmap='hot', s=dot_size, edgecolor='black', linewidth=0.2)
                ax.axis('off')
            
            plt.subplots_adjust(wspace=-0.048, hspace=-0.027)

            # # Produce a legend with the unique colors from the scatter.
            # if pointsensor_channel == 'LIDAR_TOP' and (show_lidarseg or show_panoptic) and show_lidarseg_legend:
            #     # If user does not specify a filter, then set the filter to contain the classes present in the pointcloud
            #     # after it has been projected onto the image; this will allow displaying the legend only for classes which
            #     # are present in the image (instead of all the classes).
            #     if filter_lidarseg_labels is None:
            #         if show_lidarseg:
            #             # Since the labels are stored as class indices, we get the RGB colors from the
            #             # colormap in an array where the position of the RGB color corresponds to the index
            #             # of the class it represents.
            #             color_legend = colormap_to_colors(self.nusc.colormap, self.nusc.lidarseg_name2idx_mapping)
            #             filter_lidarseg_labels = get_labels_in_coloring(color_legend, coloring_result)
            #         else:
            #             # Only show legends for all stuff categories for panoptic.
            #             filter_lidarseg_labels = stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping))

            #     if filter_lidarseg_labels and show_panoptic:
            #         # Only show legends for filtered stuff categories for panoptic.
            #         stuff_labels = set(stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping)))
            #         filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

            #     create_lidarseg_legend(filter_lidarseg_labels, self.nusc.lidarseg_idx2name_mapping, self.nusc.colormap,
            #                         loc='upper left', ncol=1, bbox_to_anchor=(1.05, 1.0))

            if out_path is not None:
                plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
            if verbose:
                plt.show()
            plt.close()

    def map_multiview_pointcloud_to_image(self,
                                pointsensor_token: list,
                                pointsensor_channel: list,
                                camera_token: list,
                                camera_channel: list,
                                min_dist: float = 1.0,
                                render_intensity: bool = False,
                                show_lidarseg: bool = False,
                                filter_lidarseg_labels: List = None,
                                lidarseg_preds_bin_path: str = None,
                                show_panoptic: bool = False) -> Tuple:
        '''
        Mapping 5 radars' pointcloud to 6 images - aggregate all radar, camera informations per one frame
        '''

        """
        Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
        plane.
        :param pointsensor_token: Lidar/radar sample_data token.
        :param camera_token: Camera sample_data token.
        :param min_dist: Distance from the camera below which points are discarded.
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :param show_lidarseg: Whether to render lidar intensity instead of point depth.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """
        # 6 image infos
        cam_list = [self.nusc.get('sample_data', camera_token[i]) for i in range(len(camera_token))]

        # create empty list for points, coloring, pc
        points_result = {}
        coloring_result = {}
        im_result= {}

        # iteration for all img
        for j in range(len(cam_list)):
            points_list = [[],[],[]]
            coloring_list = []
            im = Image.open(osp.join(self.nusc.dataroot, cam_list[j]['filename']))
            # iteration for all point cloud
            for k in range(len(pointsensor_token)):
                pointsensor = self.nusc.get('sample_data', pointsensor_token[k])
                pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
                if 'LIDAR_TOP' in pointsensor_channel:
                    pc = LidarPointCloud.from_file(pcl_path)
                else:
                    pc = RadarPointCloud.from_file(pcl_path)
                # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
                # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
                cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
                pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
                pc.translate(np.array(cs_record['translation']))

                # Second step: transform from ego to the global frame.
                poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
                pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
                pc.translate(np.array(poserecord['translation']))

                # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
                poserecord = self.nusc.get('ego_pose', cam_list[j]['ego_pose_token'])
                pc.translate(-np.array(poserecord['translation']))
                pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

                # Fourth step: transform from ego into the camera.
                cs_record = self.nusc.get('calibrated_sensor', cam_list[j]['calibrated_sensor_token'])
                pc.translate(-np.array(cs_record['translation']))
                pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

                # Fifth step: actually take a "picture" of the point cloud.
                # Grab the depths (camera frame z axis points away from the camera).
                depths = pc.points[2, :]

                if render_intensity:
                    assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                                    'not %s!' % pointsensor['sensor_modality']
                    # Retrieve the color from the intensities.
                    # Performs arbitary scaling to achieve more visually pleasing results.
                    intensities = pc.points[3, :]
                    intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
                    intensities = intensities ** 0.1
                    intensities = np.maximum(0, intensities - 0.5)
                    coloring = intensities
                elif show_lidarseg or show_panoptic:
                    assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
                                                                    'not %s!' % pointsensor['sensor_modality']

                    gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                    semantic_table = getattr(self.nusc, gt_from)

                    if lidarseg_preds_bin_path:
                        sample_token = self.nusc.get('sample_data', pointsensor_token)['sample_token']
                        lidarseg_labels_filename = lidarseg_preds_bin_path
                        assert os.path.exists(lidarseg_labels_filename), \
                            'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                            'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
                    else:
                        if len(semantic_table) > 0:  # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                            lidarseg_labels_filename = osp.join(self.nusc.dataroot,
                                                                self.nusc.get(gt_from, pointsensor_token)['filename'])
                        else:
                            lidarseg_labels_filename = None

                    if lidarseg_labels_filename:
                        # Paint each label in the pointcloud with a RGBA value.
                        if show_lidarseg:
                            coloring = paint_points_label(lidarseg_labels_filename,
                                                        filter_lidarseg_labels,
                                                        self.nusc.lidarseg_name2idx_mapping,
                                                        self.nusc.colormap)
                        else:
                            coloring = paint_panop_points_label(lidarseg_labels_filename,
                                                                filter_lidarseg_labels,
                                                                self.nusc.lidarseg_name2idx_mapping,
                                                                self.nusc.colormap)

                    else:
                        coloring = depths
                        print(f'Warning: There are no lidarseg labels in {self.nusc.version}. Points will be colored according '
                            f'to distance from the ego vehicle instead.')
                else:
                    # Retrieve the color from the depth.
                    coloring = depths

                # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

                # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
                # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
                # casing for non-keyframes which are slightly out of sync.
                mask = np.ones(depths.shape[0], dtype=bool)
                mask = np.logical_and(mask, depths > min_dist)
                mask = np.logical_and(mask, points[0, :] > 1)
                mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
                mask = np.logical_and(mask, points[1, :] > 1)
                mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
                points = points[:, mask]
                coloring = coloring[mask]

                # print("k: ", k)
                # print("points: ", points)
                # print("len points: ", len(points[0]))

                # print('coloring: ', coloring)
            
                # print(points)
                if len(points[0]) != 0:
                    points_list = np.hstack((points_list, points)) # after k iteration, all radar points for one camera
                if len(coloring) != 0:
                    coloring_list = np.hstack((coloring_list, coloring))
            im_result[camera_channel[j]] = im
            points_result[camera_channel[j]] = points_list 
            coloring_result[camera_channel[j]] = coloring_list
        # contains 6 results each, there are 5 datas per each result(except im_result)
        return points_result, coloring_result, im_result

    def render_sample_data_new(self,
                           sample_data_token: str,
                           with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 150,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           show_lidarseg: bool = False,
                           show_lidarseg_legend: bool = False,
                           filter_lidarseg_labels: List = None,
                           lidarseg_preds_bin_path: str = None,
                           verbose: bool = True,
                           show_panoptic: bool = False) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to display the image after it is rendered.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        if show_lidarseg:
            show_panoptic = False
        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':
                if show_lidarseg or show_panoptic:
                    gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                    assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

                    # Ensure that lidar pointcloud is from a keyframe.
                    assert sd_record['is_key_frame'], \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                    assert nsweeps == 1, \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                        'be set to 1.'

                    # Load a single lidar point cloud.
                    pcl_path = osp.join(self.nusc.dataroot, ref_sd_record['filename'])
                    pc = LidarPointCloud.from_file(pcl_path)
                else:
                    # Get aggregated lidar point cloud in lidar frame.
                    pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan,
                                                                     nsweeps=nsweeps)
                velocities = None
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if underlay_map:
                assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            if sensor_modality == 'lidar' and (show_lidarseg or show_panoptic):
                gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                semantic_table = getattr(self.nusc, gt_from)
                # Load labels for pointcloud.
                if lidarseg_preds_bin_path:
                    sample_token = self.nusc.get('sample_data', sample_data_token)['sample_token']
                    lidarseg_labels_filename = lidarseg_preds_bin_path
                    assert os.path.exists(lidarseg_labels_filename), \
                        'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                        'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, sample_data_token)
                else:
                    if len(semantic_table) > 0:
                        # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                        lidarseg_labels_filename = osp.join(self.nusc.dataroot,
                                                            self.nusc.get(gt_from, sample_data_token)['filename'])
                    else:
                        lidarseg_labels_filename = None

                if lidarseg_labels_filename:
                    # Paint each label in the pointcloud with a RGBA value.
                    if show_lidarseg or show_panoptic:
                        if show_lidarseg:
                            colors = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                        self.nusc.lidarseg_name2idx_mapping, self.nusc.colormap)
                        else:
                            colors = paint_panop_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                              self.nusc.lidarseg_name2idx_mapping, self.nusc.colormap)

                        if show_lidarseg_legend:

                            # If user does not specify a filter, then set the filter to contain the classes present in
                            # the pointcloud after it has been projected onto the image; this will allow displaying the
                            # legend only for classes which are present in the image (instead of all the classes).
                            if filter_lidarseg_labels is None:
                                if show_lidarseg:
                                    # Since the labels are stored as class indices, we get the RGB colors from the
                                    # colormap in an array where the position of the RGB color corresponds to the index
                                    # of the class it represents.
                                    color_legend = colormap_to_colors(self.nusc.colormap,
                                                                      self.nusc.lidarseg_name2idx_mapping)
                                    filter_lidarseg_labels = get_labels_in_coloring(color_legend, colors)
                                else:
                                    # Only show legends for stuff categories for panoptic.
                                    filter_lidarseg_labels = stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping))

                            if filter_lidarseg_labels and show_panoptic:
                                # Only show legends for filtered stuff categories for panoptic.
                                stuff_labels = set(stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping)))
                                filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

                            create_lidarseg_legend(filter_lidarseg_labels,
                                                   self.nusc.lidarseg_idx2name_mapping,
                                                   self.nusc.colormap,
                                                   loc='upper left',
                                                   ncol=1,
                                                   bbox_to_anchor=(1.05, 1.0))
                else:
                    print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                          'from the ego vehicle instead.'.format(self.nusc.version))

            point_scale = 0.07 if sensor_modality == 'lidar' else 1.0
            ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale, cmap = 'hot')
            # scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

            # # Show velocities.
            # if sensor_modality == 'radar':
            #     points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
            #     deltas_vel = points_vel - points
            #     deltas_vel = 6 * deltas_vel  # Arbitrary scaling
            #     max_delta = 20
            #     deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
            #     colors_rgba = scatter.to_rgba(colors)
            #     for i in range(points.shape[1]):
            #         ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')

            # Get boxes in lidar frame.
            _, boxes, _ = self.nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                    use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)

        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,
                                                                           box_vis_level=box_vis_level)
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis('off')
        # ax.set_title('{} {labels_type}'.format(
        #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

        if verbose:
            plt.show()

    def render_sample_new(self,
                      token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      sample_path: str = None,
                      show_lidarseg: bool = False,
                      filter_lidarseg_labels: List = None,
                      lidarseg_preds_bin_path: str = None,
                      verbose: bool = True,
                      show_panoptic: bool = False) -> None:
        """
        Render all LIDAR and camera sample_data in sample along with annotations.
        :param token: Sample token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param show_lidarseg: Whether to show lidar segmentations labels or not.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to show the rendered sample in a window or not.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        record = self.nusc.get('sample', token)

        # Separate RADAR from LIDAR and vision.
        radar_data = {}
        camera_data = {}
        lidar_data = {}
        for channel, token in record['data'].items():
            sd_record = self.nusc.get('sample_data', token)
            sensor_modality = sd_record['sensor_modality']

            if sensor_modality == 'camera':
                camera_data[channel] = token
            elif sensor_modality == 'lidar':
                lidar_data[channel] = token
            else:
                radar_data[channel] = token

        # Create plots.
        num_radar_plots = 1 if len(radar_data) > 0 else 0
        num_lidar_plots = 1 if len(lidar_data) > 0 else 0
        n = num_radar_plots + num_lidar_plots
        cols = 2
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(18, 9))

        # Plot radars into a single subplot.
        if len(radar_data) > 0:
            ax = axes[0]
            for i, (_, sd_token) in enumerate(radar_data.items()):
                self.render_sample_data_new(sd_token, with_anns=i == 0, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,
                                        verbose=False)
            # ax.set_title('Fused RADARs')

        # Plot lidar into a single subplot.
        if len(lidar_data) > 0:
            for (_, sd_token), ax in zip(lidar_data.items(), axes.flatten()[num_radar_plots:]):
                self.render_sample_data_new(sd_token, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,
                                        show_lidarseg=show_lidarseg, filter_lidarseg_labels=filter_lidarseg_labels,
                                        lidarseg_preds_bin_path=lidarseg_preds_bin_path, verbose=False,
                                        show_panoptic=show_panoptic)

        # Plot cameras in separate subplots.
        for (_, sd_token), ax in zip(camera_data.items(), axes.flatten()[num_radar_plots + num_lidar_plots:]):
            if show_lidarseg or show_panoptic:
                sd_record = self.nusc.get('sample_data', sd_token)
                sensor_channel = sd_record['channel']
                valid_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                  'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
                assert sensor_channel in valid_channels, 'Input camera channel {} not valid.'.format(sensor_channel)

                self.render_pointcloud_in_image(record['token'],
                                                pointsensor_channel='LIDAR_TOP',
                                                camera_channel=sensor_channel,
                                                show_lidarseg=show_lidarseg,
                                                filter_lidarseg_labels=filter_lidarseg_labels,
                                                ax=ax, verbose=False,
                                                lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                show_panoptic=show_panoptic)
            else:
                self.render_sample_data_new(sd_token, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,
                                        show_lidarseg=False, verbose=False)
        img_out_path='/home/byounghun/workspace/temp_11.jpg'
        self.render_pointcloud_in_scene(record['token'],
                                   dot_size=5,
                                   pointsensor_channel=self.radar,
                                   camera_channel=self.camera, 
                                   out_path=img_out_path,
                                   render_intensity= False,
                                   show_lidarseg= False,
                                   filter_lidarseg_labels= None,
                                   ax = None,
                                   show_lidarseg_legend= False,
                                   verbose= False,
                                   lidarseg_preds_bin_path= None,
                                   show_panoptic= False,
                                   show_both_modality = True)

        # Change plot settings and write to disk.
        axes.flatten()[-1].axis('off')
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(out_path)

        imgs = plt.imread(img_out_path)
        pts = plt.imread(out_path)
        h, w, _ = imgs.shape
        h_p, w_p, _ = pts.shape
        ratio = h_p/w_p

        new_w = w
        new_h = int(new_w*ratio)

        resized_pts = cv2.resize(pts, (new_w, new_h))
        concat_image = np.vstack((imgs,resized_pts))

        if sample_path is not None:
            plt.imsave(sample_path, concat_image)

        sample = plt.imread(sample_path)

        if verbose:
            plt.close()
            plt.figure()
            plt.imshow(sample)
            plt.axis('off')
            # plt.show()
        plt.close('all')