from .new_methods import My_NuScenes, My_NuScenesExplorer
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os
import os.path as osp
import sys
from typing import Tuple, List, Iterable

import cv2
import matplotlib.pyplot as plt
import math
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
from nuscenes.utils.data_classes import LidarPointCloud, Box
from new_methods.MyRadarPointCloud import MyRadarPointCloud
from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.color_map import get_colormap
from matplotlib.patches import Circle

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("nuScenes dev-kit only supports Python version 3.")

class Extended_Nusenes(My_NuScenes):

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
        self.explorer = Extended_NuScenesExplorer(self)

    def render_radar_pcd_in_scene(self,
                                sample_token: str,
                                dot_size: int = 5,
                                filt_dot_size: int=20,
                                nsweeps: int = 1,
                                pointsensor_channel: list = None,
                                camera_channel: list = None, 
                                out_path: str = None,
                                render_intensity: bool = False,
                                ax: Axes = None,
                                verbose: bool = True,
                                show_panoptic: bool = False):
        self.explorer.render_radar_pcd_in_scene(sample_token=sample_token,
                                                dot_size=dot_size,
                                                filt_dot_size=filt_dot_size,
                                                nsweeps=nsweeps,
                                                pointsensor_channel=pointsensor_channel,
                                                camera_channel=camera_channel, 
                                                out_path=out_path,
                                                render_intensity=render_intensity,
                                                ax=ax,
                                                verbose=verbose,
                                                show_panoptic=show_panoptic)
            
    def render_radar_pts(self, sample_token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      sample_path: str = None,
                      show_lidarseg: bool = False,
                      filter_lidarseg_labels: List = None,
                      lidarseg_preds_bin_path: str = None,
                      verbose: bool = True,
                      show_panoptic: bool = False) -> None:
        self.explorer.render_radar_pts(sample_token, box_vis_level, nsweeps=nsweeps, out_path=out_path, sample_path=sample_path,
                                    show_lidarseg=show_lidarseg, filter_lidarseg_labels=filter_lidarseg_labels,
                                    lidarseg_preds_bin_path=lidarseg_preds_bin_path, verbose=verbose,
                                    show_panoptic=show_panoptic)
    
    def check_depth_diff(self,
                    sample_token: str,
                    boundary: int,
                    dot_size: int = 5,
                    pointsensor_channel: list = None,
                    camera_channel: list = None, 
                    out_path: str = None,
                    render_intensity: bool = False,
                    show_lidarseg: bool = False,
                    filter_lidarseg_labels: List = None,
                    ax: Axes = None,
                    verbose: bool = True,
                    lidarseg_preds_bin_path: str = None,
                    show_panoptic: bool = False):
        return self.explorer.check_depth_diff(
                    sample_token,
                    boundary,
                    dot_size,
                    pointsensor_channel = pointsensor_channel,
                    camera_channel = camera_channel, 
                    out_path = out_path,
                    render_intensity = render_intensity,
                    show_lidarseg = show_lidarseg,
                    filter_lidarseg_labels = filter_lidarseg_labels,
                    ax = ax,
                    verbose = verbose,
                    lidarseg_preds_bin_path = lidarseg_preds_bin_path,
                    show_panoptic = show_panoptic)
        
class Extended_NuScenesExplorer(My_NuScenesExplorer):

    def __init__(self, nusc: Extended_Nusenes):
        self.nusc = nusc
        self.radar = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 
                                                                'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        self.camera = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 
                                                           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    def render_pointcloud_in_scene(self,
                                   sample_token: str,
                                   dot_size: int = 5,
                                   nsweeps: int = 1,
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
            # points_result = {}
            # coloring_result = {}
            pointsensor_channel_L = ['LIDAR_TOP']
            pointsensor_token_L = [sample_record['data'][pointsensor_channel_L[i]] for i in range(len(pointsensor_channel_L))]
            points_result_L, coloring_result_L, im_result = self.map_multiview_pointcloud_to_image(pointsensor_token_L,
                                                                                           pointsensor_channel_L,
                                                                                           camera_token,
                                                            camera_channel=camera_channel,
                                                            nsweeps=nsweeps,
                                                            render_intensity=render_intensity,
                                                            show_lidarseg=show_lidarseg,
                                                            filter_lidarseg_labels=filter_lidarseg_labels,
                                                            lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                            show_panoptic=show_panoptic)
            points_result_R, coloring_result_R, im_result = self.map_multiview_pointcloud_to_image(pointsensor_token,
                                                                                           pointsensor_channel,
                                                                                           camera_token,
                                                            camera_channel=camera_channel,
                                                            nsweeps=nsweeps,
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
                                                            nsweeps=nsweeps,
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
                if len(points_result[channel][0]) > 0:
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
                else: ax.imshow(img)
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

    def check_depth_diff(self,
                    sample_token: str,
                    boundary: int = 20,
                    dot_size: int = 5,
                    pointsensor_channel: list = None,
                    camera_channel: list = None, 
                    out_path: str = None,
                    render_intensity: bool = False,
                    show_lidarseg: bool = False,
                    filter_lidarseg_labels: List = None,
                    ax: Axes = None,
                    verbose: bool = True,
                    lidarseg_preds_bin_path: str = None,
                    show_panoptic: bool = False):
        
        sample_record = self.nusc.get('sample', sample_token)

        if pointsensor_channel is None:
            pointsensor_channel = self.radar
        if camera_channel is None:
            camera_channel = self.camera

        # Here we just grab 6 camera and the 5 point sensor.
        pointsensor_token = [sample_record['data'][pointsensor_channel[i]] for i in range(len(pointsensor_channel))]
        camera_token = [sample_record['data'][camera_channel[i]] for i in range(len(camera_channel))]

        # mapping pointcloud to images (able to map RADAR and LiDAR together)
        pointsensor_channel_L = ['LIDAR_TOP']
        pointsensor_token_L = [sample_record['data'][pointsensor_channel_L[i]] for i in range(len(pointsensor_channel_L))]
        points_result_L, depth_result_L, _ = self.map_multiview_pointcloud_to_image(pointsensor_token_L,
                                                                                            pointsensor_channel_L,
                                                                                            camera_token,
                                                                                            camera_channel=camera_channel,
                                                                                            render_intensity=render_intensity,
                                                                                            show_lidarseg=show_lidarseg,
                                                                                            filter_lidarseg_labels=filter_lidarseg_labels,
                                                                                            lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                                                            show_panoptic=show_panoptic)
        points_result_R, depth_result_R, _ = self.map_multiview_pointcloud_to_image(pointsensor_token,
                                                                                            pointsensor_channel,
                                                                                            camera_token,
                                                                                            camera_channel=camera_channel,
                                                                                            render_intensity=render_intensity,
                                                                                            show_lidarseg=show_lidarseg,
                                                                                            filter_lidarseg_labels=filter_lidarseg_labels,
                                                                                            lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                                                            show_panoptic=show_panoptic)

        # depth difference storage
        depth_diff_array = []

        # distance between pts storage
        distance_array = []

        # no near lidar points
        no_near_pts = 0

        # iteration for total images (6 images)
        for k in range(6):
            # image
            index_ax = [5,0,1,4,3,2]
            channel = camera_channel[index_ax[k]]
            # LiDAR point cloud
            x_L = points_result_L[channel][0, :]
            y_L = points_result_L[channel][1, :]
            d_L = depth_result_L[channel]
            # RADAR point cloud
            if len(points_result_R[channel][0]) > 0:
                x_R = points_result_R[channel][0, :]
                y_R = points_result_R[channel][1, :]
                d_R = depth_result_R[channel]

                # iteration for radar points (1 image)
                for i in range(len(d_R)):
                    Lipo = np.array([x_L, y_L, d_L])
                    Rapo = np.array([x_R, y_R, d_R])

                    # masking LiDAR points (near Radar points survive)
                    mask = np.ones(d_L.shape[0], dtype=bool)
                    mask = np.logical_and(mask, x_L > Rapo[0][i] - boundary)
                    mask = np.logical_and(mask, x_L < Rapo[0][i] + boundary)
                    mask = np.logical_and(mask, y_L > Rapo[1][i] - boundary)
                    mask = np.logical_and(mask, y_L < Rapo[1][i] + boundary)

                    # Survived Lidar points per one radar point
                    if True in mask:
                        distance = 100 # initial distance
                        index = 0 # initial index of nearest Lidar point
                        Lipo = Lipo[:,mask]
                        for j in range(len(Lipo[0])):
                            x_diff = Rapo[0][i] - Lipo[0][j]
                            y_diff = Rapo[1][i] - Lipo[1][j]
                            new_distance = np.sqrt(np.square(x_diff) + np.square(y_diff))
                            if new_distance < distance:
                                distance = new_distance # update nearest distance
                                index = j # update index
                        depth_diff = Rapo[2][i] - Lipo[2][index]
                        depth_diff_array.append(depth_diff)
                        distance_array.append(distance)
                    else:
                        no_near_pts += 1
        return distance_array, depth_diff_array, no_near_pts
    
    def make_points_list(self, sensor: str, frame_idx: int, nsweeps: int) -> np.ndarray:
        # Initialize an empty numpy array to store points
        points_list = np.empty((0, 3))

        # Get sample token and relevant records
        token = self.nusc.sample[frame_idx]['token']
        record = self.nusc.get('sample', token)
        ref_chan = 'LIDAR_TOP'
        ref_sd_rec = self.nusc.get('sample_data', record['data'][ref_chan])
        ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])

        # Collect sensor data based on sensor type ('r' for radar, else lidar assumed)
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

        if sensor == 'r':
            pointsensor_channel = list(radar_data.keys())
            pointsensor_token = list(radar_data.values())
        else:
            pointsensor_channel = list(lidar_data.keys())
            pointsensor_token = list(lidar_data.values())
        
        # Iterate over each point sensor and accumulate points
        for k in range(len(pointsensor_token)):
            pointsensor = self.nusc.get('sample_data', pointsensor_token[k])
            
            if 'LIDAR_TOP' in pointsensor_channel:
                pc, times = LidarPointCloud.from_file_multisweep(self.nusc, record, pointsensor['channel'], ref_chan, nsweeps=nsweeps)
            else:
                pc, times = MyRadarPointCloud.from_file_multisweep(self.nusc, record, pointsensor['channel'], ref_chan, nsweeps=nsweeps)

            if pc is not None:
                # Transform the point cloud to the ego vehicle frame
                pc.rotate(Quaternion(ref_cs_rec['rotation']).rotation_matrix)
                pc.translate(np.array(ref_cs_rec['translation']))

                if len(pc.points[0]) != 0:
                    points = pc.points[:3, :].T  # Transpose points to match (N, 3) shape
                    points_list = np.concatenate((points_list, points), axis=0)  # Concatenate points

        return points_list

    def render_radar_pts(self,
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Ensure we always have 3 plots

        # Plot unfiltered radar.
        if len(radar_data) > 0:
            ax = axes[0]
            for i, (_, sd_token) in enumerate(radar_data.items()):
                MyRadarPointCloud.disable_filters()
                self.render_sample_data_new(sd_token, with_anns=i == 0, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,
                                            verbose=False)
            ax.set_title('Radar Unfiltered')

        # Plot filtered radar.
        if len(radar_data) > 0:
            ax = axes[1]
            for i, (_, sd_token) in enumerate(radar_data.items()):
                MyRadarPointCloud.filtered_pts()
                self.render_sample_data_new(sd_token, with_anns=i == 0, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,
                                            verbose=False)
            ax.set_title('Filtered Radar pts')

        # Plot lidar.
        if len(lidar_data) > 0:
            ax = axes[2]
            for _, sd_token in lidar_data.items():
                self.render_sample_data_new(sd_token, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,
                                            show_lidarseg=show_lidarseg, filter_lidarseg_labels=filter_lidarseg_labels,
                                            lidarseg_preds_bin_path=lidarseg_preds_bin_path, verbose=False,
                                            show_panoptic=show_panoptic)
            ax.set_title('Lidar')

        check_path = '/home/cwkim0214/workspace/Nuscenes_tutorial'
        img_out_path = osp.join(check_path, 'temp.jpg')
        if not osp.exists(check_path):
            os.makedirs(check_path)

        self.render_pointcloud_in_scene(record['token'],
                                        dot_size=5,
                                        nsweeps=nsweeps,
                                        pointsensor_channel=self.radar,
                                        camera_channel=self.camera,
                                        out_path=img_out_path,
                                        render_intensity=False,
                                        show_lidarseg=False,
                                        filter_lidarseg_labels=None,
                                        ax=None,
                                        show_lidarseg_legend=False,
                                        verbose=False,
                                        lidarseg_preds_bin_path=None,
                                        show_panoptic=False,
                                        show_both_modality=True)

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

        # Display the plots if verbose is True.
        if verbose:
            plt.close()
            plt.figure(figsize=(24, 12))
            plt.imshow(concat_image)  # Ensure the correct image is displayed
            plt.axis('off')
            plt.show()
        else:
            plt.close('all')

    def render_radar_pcd_in_scene(self,
                                sample_token: str,
                                dot_size: int=5,
                                filt_dot_size: int=20,
                                nsweeps: int=1,
                                pointsensor_channel: list=None,
                                camera_channel: list=None,
                                out_path: str=None,
                                render_intensity: bool=False,
                                ax: Axes=None,
                                verbose: bool=True,
                                show_panoptic: bool=False):
        """
        Scatter-plots a pointcloud on top of image.
        :param sample_token: Sample token.
        :param dot_size: Scatter plot dot size.
        :param filt_dot_size: Scatter plot dot size of removed points by Nuscenes filter.
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
                            to False, the colors of the lidar data represent the distance from the center of 
                            the ego vehicle. If show_lidarseg is True, show_panoptic will be set to False.
        """
        sample_record = self.nusc.get('sample', sample_token)

        if pointsensor_channel is None:
            pointsensor_channel = self.radar
        if camera_channel is None:
            camera_channel = self.camera

        # Here we just grab the front camera and the point sensor.
        pointsensor_token = [sample_record['data'][pointsensor_channel[i]] for i in range(len(pointsensor_channel))]
        camera_token = [sample_record['data'][camera_channel[i]] for i in range(len(camera_channel))]
        
        # Map point clouds to image
        MyRadarPointCloud.default_filters()
        points_result, coloring_result, im_result = self.map_multiview_pointcloud_to_image(pointsensor_token,
                                                                                        pointsensor_channel,
                                                                                        camera_token,
                                                                                        camera_channel=camera_channel,
                                                                                        nsweeps=nsweeps,
                                                                                        render_intensity=render_intensity,
                                                                                        show_lidarseg=False,
                                                                                        filter_lidarseg_labels=None,
                                                                                        lidarseg_preds_bin_path=None,
                                                                                        show_panoptic=show_panoptic)
        MyRadarPointCloud.removed_invalid_states()
        points_invalid, coloring_invalid, im_result_invalid = self.map_multiview_pointcloud_to_image(pointsensor_token,
                                                                                        pointsensor_channel,
                                                                                        camera_token,
                                                                                        camera_channel=camera_channel,
                                                                                        nsweeps=nsweeps,
                                                                                        render_intensity=render_intensity,
                                                                                        show_lidarseg=False,
                                                                                        filter_lidarseg_labels=None,
                                                                                        lidarseg_preds_bin_path=None,
                                                                                        show_panoptic=show_panoptic)
        
        MyRadarPointCloud.removed_ambig_states()
        points_ambig, coloring_ambig, im_result_ambig = self.map_multiview_pointcloud_to_image(pointsensor_token,
                                                                                        pointsensor_channel,
                                                                                        camera_token,
                                                                                        camera_channel=camera_channel,
                                                                                        nsweeps=nsweeps,
                                                                                        render_intensity=render_intensity,
                                                                                        show_lidarseg=False,
                                                                                        filter_lidarseg_labels=None,
                                                                                        lidarseg_preds_bin_path=None,
                                                                                        show_panoptic=show_panoptic)
        MyRadarPointCloud.removed_dynprop_states()
        points_dynprop, coloring_dynprop, im_result_dynprop = self.map_multiview_pointcloud_to_image(pointsensor_token,
                                                                                        pointsensor_channel,
                                                                                        camera_token,
                                                                                        camera_channel=camera_channel,
                                                                                        nsweeps=nsweeps,
                                                                                        render_intensity=render_intensity,
                                                                                        show_lidarseg=False,
                                                                                        filter_lidarseg_labels=None,
                                                                                        lidarseg_preds_bin_path=None,
                                                                                        show_panoptic=show_panoptic)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(2, 3, figsize=(24, 9))
            
        index_ax = [5, 0, 1, 4, 3, 2]
        flip_list = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        
        for i, axis in enumerate(ax.flat):
            # image
            channel = camera_channel[index_ax[i]]
            img = im_result[channel]
            axis.imshow(img)
            
            # Initialize x and y for this iteration
            x = np.array([])
            y = np.array([])
            
            # Render points from points_result
            if len(points_result[channel][0]) > 0:
                x = points_result[channel][0, :]
                y = points_result[channel][1, :]
                c = coloring_result[channel]
                # flip
                if channel in flip_list:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    x = img.size[0] - x
                axis.scatter(x, y, c=c, cmap='hot', s=dot_size, marker='o', edgecolor='black', linewidth=0.2)

            if len(points_invalid[channel][0]) > 0:
                x = points_invalid[channel][0, :]
                y = points_invalid[channel][1, :]
                c = coloring_invalid[channel]
                # flip
                if channel in flip_list:
                    x = img.size[0] - x
                axis.scatter(x, y, c=c, cmap='hot', s=dot_size, marker='x', linewidth=1.0)

            if len(points_ambig[channel][0]) > 0:
                x = points_ambig[channel][0, :]
                y = points_ambig[channel][1, :]
                c = coloring_ambig[channel]
                # flip
                if channel in flip_list:
                    x = img.size[0] - x
                axis.scatter(x, y, c=c, cmap='hot', s=filt_dot_size, marker='1', linewidth=0.8)

            if len(points_dynprop[channel][0]) > 0:
                x = points_dynprop[channel][0, :]
                y = points_dynprop[channel][1, :]
                c = coloring_dynprop[channel]
                # flip
                if channel in flip_list:
                    x = img.size[0] - x
                axis.scatter(x, y, c=c, cmap='hot', s=filt_dot_size, marker='*', edgecolor='black', linewidth=0.5)
            
            axis.axis('off')
        plt.subplots_adjust(wspace=-0.048, hspace=-0.027)

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
        if verbose:
            plt.show()
        plt.close()

    def render_ego_centric_map_modified(self,
                                        sample_data_token: str,
                                        rotation: float = 0,
                                        axes_limit: float = 40,
                                        ax: Axes = None) -> None:
        """
        Render map centered around the associated ego pose.
        :param sample_data_token: Sample_data token.
        :param rotation: angle in degrees to rotate the map in CCW direction
        :param axes_limit: Axes limit measured in meters.
        :param ax: Axes onto which to render.
        """

        def crop_image(image: np.array,
                       x_px: int,
                       y_px: int,
                       axes_limit_px: int) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        # Get data.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sample = self.nusc.get('sample', sd_record['sample_token'])
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        map_ = self.nusc.get('map', log['map_token'])
        map_mask = map_['mask']
        pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

        # Retrieve and crop mask.
        pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
        scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
        mask_raster = map_mask.mask()
        cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

        # Rotate image.
        ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0]) + rotation
        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

        # Crop image.
        ego_centric_map = crop_image(rotated_cropped,
                                     int(rotated_cropped.shape[1] / 2),
                                     int(rotated_cropped.shape[0] / 2),
                                     scaled_limit_px)

        # Init axes and show image.
        # Set background to white and foreground (semantic prior) to gray.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))
        ego_centric_map[ego_centric_map == map_mask.foreground] = 125
        ego_centric_map[ego_centric_map == map_mask.background] = 255
        ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit],
                  cmap='gray', vmin=0, vmax=255)