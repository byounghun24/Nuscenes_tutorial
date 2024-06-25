from nuscenes.utils.data_classes import RadarPointCloud

class MyRadarPointCloud(RadarPointCloud):
    @classmethod
    def filtered_pts(cls) -> None:
        # Outdated, can only Visualize certain points that were filtered out by invalid_states
        cls.invalid_states = [1,6,9,14]
        cls.dynprop_states = list(range(8))
        cls.ambig_states = [3]

    @classmethod
    def removed_invalid_states(cls) -> None:
        """
        points that are removed due to 'invalid_states' parameter
        Note that this method affects the global settings.
        """
        cls.invalid_states = list(range(1,18))
        cls.dynprop_states = list(range(8))
        cls.ambig_states = list(range(5))

    @classmethod
    def removed_dynprop_states(cls) -> None:
        """
        points that are removed due to 'dynprop_states' parameter
        Note that this method affects the global settings.
        """
        cls.invalid_states = list(range(18))
        cls.dynprop_states = [7]
        cls.ambig_states = list(range(5))

    @classmethod
    def removed_ambig_states(cls) -> None:
        """
        Set the defaults for all radar filter settings.
        Note that this method affects the global settings.
        """
        cls.invalid_states = [0]
        cls.dynprop_states = range(7)
        cls.ambig_states = [0,1,2,4]

        """
        Loads RADAR data from a Point Cloud Data file. See details below.
        :param file_name: The path of the pointcloud file.
        :param invalid_states: Radar states to be kept. See details below.
        :param dynprop_states: Radar states to be kept. Use [0, 2, 6] for moving objects only. See details below.
        :param ambig_states: Radar states to be kept. See details below.
        To keep all radar returns, set each state filter to range(18).
        :return: <np.float: d, n>. Point cloud matrix with d dimensions and n points.

        Example of the header fields:
        # .PCD v0.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
        SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
        TYPE F F F I I F F F F F I I I I I I I I
        COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        WIDTH 125
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS 125
        DATA binary

        Below some of the fields are explained in more detail:

        x is front, y is left

        vx, vy are the velocities in m/s.
        vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
        We recommend using the compensated velocities.

        invalid_state: state of Cluster validity state.
        (Invalid states)
        0x01	invalid due to low RCS
        0x02	invalid due to near-field artefact
        0x03	invalid far range cluster because not confirmed in near range
        0x05	reserved
        0x06	invalid cluster due to high mirror probability
        0x07	Invalid cluster because outside sensor field of view
        0x0d	reserved
        0x0e	invalid cluster because it is a harmonics
        (Valid states)
        0x00	valid
        0x04	valid cluster with low RCS
        0x08	valid cluster with azimuth correction due to elevation
        0x09	valid cluster with high child probability
        0x0a	valid cluster with high probability of being a 50 deg artefact
        0x0b	valid cluster but no local maximum
        0x0c	valid cluster with high artefact probability
        0x0f	valid cluster with above 95m in near range
        0x10	valid cluster with high multi-target probability
        0x11	valid cluster with suspicious angle

        dynProp: Dynamic property of cluster to indicate if is moving or not.
        0: moving
        1: stationary
        2: oncoming
        3: stationary candidate
        4: unknown
        5: crossing stationary
        6: crossing moving
        7: stopped

        ambig_state: State of Doppler (radial velocity) ambiguity solution.
        0: invalid
        1: ambiguous
        2: staggered ramp
        3: unambiguous
        4: stationary candidates

        pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
        0: invalid
        1: <25%
        2: 50%
        3: 75%
        4: 90%
        5: 99%
        6: 99.9%
        7: <=100%
        """