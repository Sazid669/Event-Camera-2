import cv2
import numpy as np

from typing import Tuple


class GroundTruthSynchronizer():
    """
    Ground truth synchronizer for displacement-based optical flow. This synchronizer
    interpolates the optical flow between two desired timestamps. To run it refer to the
    'sync' function after initialization.

    Attributes
    ----------
    flows : np.ndarray
        Ground truth optical flows of shape ('num_frames', 2, H, W).
    timestamps : np.ndarray
        Timestamps of ground truth optical flow of shape ('num_frames').
    """
    def __init__(
        self,
        flows: np.ndarray,
        timestamps: np.ndarray
    ) -> None:
        # Ground truth data
        self._flows = flows
        self._timestamps = timestamps
        # Divide flow
        self._flows_x = self._flows[:, 0, :, :]
        self._flows_y = self._flows[:, 1, :, :]

    def _init_ids(self) -> None:
        """
        Initializes flow indices.
        """
        # Initialize flow displacement indices
        self._x_ids, self._y_ids = np.meshgrid(
            np.arange(self._flows.shape[3]), np.arange(self._flows.shape[2])
        )
        # Convert flow data type
        self._x_ids = self._x_ids.astype(np.float32)
        self._y_ids = self._y_ids.astype(np.float32)
        # Initial flow indices
        self._x_ids_init = np.copy(self._x_ids)
        self._y_ids_init = np.copy(self._y_ids)

    def _init_masks(self) -> None:
        """
        Initializes flow masks.
        """
        self._mask_x = np.ones(self._x_ids.shape, dtype=bool)
        self._mask_y = np.ones(self._y_ids.shape, dtype=bool)

    def _propagate_flow(
        self,
        flow_x: np.ndarray,
        flow_y: np.ndarray,
        delta_time: float = 1.0
    ) -> None:
        """
        Propagates the optical flow indices.
        """
        remapped_flow_x = cv2.remap(flow_x, self._x_ids, self._y_ids, cv2.INTER_NEAREST)
        remapped_flow_y = cv2.remap(flow_y, self._x_ids, self._y_ids, cv2.INTER_NEAREST)
        # Compute flow masks
        self._mask_x[remapped_flow_x == 0] = False
        self._mask_y[remapped_flow_y == 0] = False
        # Apply propagation
        self._x_ids += remapped_flow_x*delta_time
        self._y_ids += remapped_flow_y*delta_time

    def sync(
        self,
        start_time: float,
        end_time: float
    ) -> Tuple[np.ndarray]:
        """
        Synchronizes ground truth optical flow between two desired timestamps. This
        function must be used to synchronize the flow.

        Parameters
        ----------
        start_time : float
            Initial timestamp (in seconds).
        end_time : float
            Final timestamp (in seconds).

        Returns
        -------
        flow_shift_x : np.ndarray
            Synchronized optical flow in the x-direction of shape (H, W).
        flow_shift_y : np.ndarray
            Synchronized optical flow in the y-direction of shape (H, W).
        """
        # Get initial flow index
        gt_idx = np.searchsorted(self._timestamps, start_time, side="left")

        # Extract flow
        flow_x = self._flows_x[gt_idx, :, :]
        flow_y = self._flows_y[gt_idx, :, :]
        # Compute flow duration
        total_dt = end_time - start_time
        gt_dt = self._timestamps[gt_idx + 1] - self._timestamps[gt_idx]

        # Interpolate (low ground truth refresh rate)
        if total_dt <= gt_dt:
            return flow_x*total_dt/gt_dt, flow_y*total_dt/gt_dt

        # Initialize propagation (high ground truth refresh rate)
        self._init_ids()
        self._init_masks()
        total_dt = self._timestamps[gt_idx + 1] - start_time
        delta_time = total_dt/gt_dt
        self._propagate_flow(
            flow_x=flow_x,
            flow_y=flow_y,
            delta_time=delta_time
        )
        gt_idx += 1

        # Repeat process
        while self._timestamps[gt_idx + 1] < end_time:
            flow_x = self._flows_x[gt_idx, :, :]
            flow_y = self._flows_y[gt_idx, :, :]
            self._propagate_flow(
                flow_x=flow_x,
                flow_y=flow_y,
                delta_time=1.0
            )
            gt_idx += 1

        # Apply for final timestamp
        flow_x = self._flows_x[gt_idx, :, :]
        flow_y = self._flows_y[gt_idx, :, :]
        total_dt = end_time - self._timestamps[gt_idx]
        gt_dt = self._timestamps[gt_idx + 1] - self._timestamps[gt_idx]
        delta_time = total_dt/gt_dt
        self._propagate_flow(
            flow_x=flow_x,
            flow_y=flow_y,
            delta_time=delta_time
        )

        # Compute flow shift
        flow_shift_x = self._x_ids - self._x_ids_init
        flow_shift_y = self._y_ids - self._y_ids_init
        flow_shift_x[~self._mask_x] = 0
        flow_shift_y[~self._mask_y] = 0

        return flow_shift_x, flow_shift_y


# ===== Example Code ===== #
if __name__ == "__main__":
    # Import data reader
    from reader import DataReader

    # Directory variables
    data_dir = ""
    gt_dir = ""
    data_hdf5_groups = {
        "events": ["davis", "left", "events"],
        "nearest_events": ["davis", "left", "image_raw_event_inds"],
        "gray_images": ["davis", "left", "image_raw"],
        "gray_timestamps": ["davis", "left", "image_raw_ts"]
    }
    gt_hdf5_groups = {
        "flows": ["davis", "left", "flow_dist"],
        "timestamps": ["davis", "left", "flow_dist_ts"]
    }

    # Run data reader
    data_reader = DataReader(
        data_dir=data_dir,
        gt_dir=gt_dir,
        data_hdf5_groups=data_hdf5_groups,
        gt_hdf5_groups=gt_hdf5_groups,
        start_time=0.1,
        end_time=0.5,
        reset_time=True,
        num_slices=10
    )
    gt_flows = data_reader.gt_flows
    gt_timestamps = data_reader.gt_timestamps

    # Initialize synchronizer
    gt_synchronizer = GroundTruthSynchronizer(
        flows=gt_flows,
        timestamps=gt_timestamps
    )
    # Synchronize flow
    flow = gt_synchronizer.sync(start_time=10, end_time=10.2)
