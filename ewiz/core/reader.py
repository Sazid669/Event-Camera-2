import h5py
import numpy as np

from typing import Dict, List, Tuple, Union


# ========================================= #
# ----- Reader Functions, Version 1.0 ----- #
# ========================================= #
def read_data(
    data_file: h5py.Dataset,
    events_group: List[str],
    nearest_events_group: List[str] = None,
    grayscale_images_group: List[str] = None,
    grayscale_timestamps_group: List[str] = None,
    reset_time: bool = False
) -> Tuple[np.ndarray]:
    """
    Unpacks a '.hdf5' dataset file and inserts the data into Numpy arrays.
    NOTE: DEPRECATED FUNCTION.

    Parameters
    ----------
    data_file : h5py.Dataset
        The '.hdf5' dataset file.
    events_group : List[str]
        Group location of the 'events' data in the '.hdf5' dataset, all group
        directories and subdirectories need to be given in order. For example,
        ['data', 'events'].
    nearest_events_group : List[str]
        Group location of the 'nearest_events' data in the '.hdf5' dataset. The
        'nearest_events' are the indices of the events closest to every
        grayscale image (defaults to None).
    grayscale_images_group : List[str]
        Group location of the 'grayscale_images' data in the '.hdf5' dataset
        (defaults to None).
    grayscale_timestamps_group : List[str]
        Group location of the 'grayscale_timestamps' data in the '.hdf5' dataset
        (defaults to None).
    reset_time : bool
        Checks for the smallest timestamp in the data, and resets the value
        of the first timestamp to 0 (defaults to False).

    Returns
    -------
    events : np.ndarray
        Events data of size ('num_events', 4), where axis 1 corresponds to
        (x, y, t, p).
    nearest_events : np.ndarray
        Nearest events of size ('num_grayscale_images'). None is returned
        instead if no 'nearest_events_group' was given as input.
    grayscale_images : np.ndarray
        Grayscale images of size ('num_grayscale_images', H, W). None is returned
        instead if no 'grayscale_images_group' was given as input.
    grayscale_timestamps : np.ndarray
        Grayscale timestamps of size ('num_grayscale_images'). None is returned
        instead if no 'grayscale_timestamps_group' was given as input.
    """
    events = None
    nearest_events = None
    grayscale_images = None
    grayscale_timestamps = None

    events = data_file
    for dir in events_group:
        events = events[dir]
    events = np.array(events)

    # Save initial timestamp for time reset
    start_time = events[0, 2]

    if nearest_events_group:
        nearest_events = data_file
        for dir in nearest_events_group:
            nearest_events = nearest_events[dir]
        nearest_events = np.array(nearest_events)

    if grayscale_images_group:
        grayscale_images = data_file
        for dir in grayscale_images_group:
            grayscale_images = grayscale_images[dir]
        grayscale_images = np.array(grayscale_images)

    if grayscale_timestamps_group:
        grayscale_timestamps = data_file
        for dir in grayscale_timestamps_group:
            grayscale_timestamps = grayscale_timestamps[dir]
        grayscale_timestamps = np.array(grayscale_timestamps)

        # Compare grayscale and events timestamps
        if grayscale_timestamps[0] < start_time:
            start_time = grayscale_timestamps[0]

        # Reset first grayscale timestamp to 0
        if reset_time:
            grayscale_timestamps -= start_time

    # Reset first event timestamp to 0
    if reset_time:
        events[:, 2] -= start_time

    return events, nearest_events, grayscale_images, grayscale_timestamps

def read_gt(
    gt_file: h5py.Dataset,
    flow_group: List[str],
    timestamps_group: List[str]
) -> Tuple[np.ndarray]:
    """
    Extracts ground truth flow data from a '.hdf5' ground truth file.
    NOTE: DEPRECATED FUNCTION.

    Parameters
    ----------
    gt_file : h5py.Dataset
        The '.hdf5' ground truth file.
    flow_group : List[str]
        Group location of the 'flow' data in the '.hdf5' ground truth, all group
        directories and subdirectories need to be given in order. For example,
        ['davis', 'left', 'flow_dist'].
    timestamps_group : List[str]
        Group location of the ground truth 'timestamps' data in the '.hdf5' file. The flow
        and its corresponding timestamp have the same index in the list.

    Returns
    -------
    flow : np.ndarray
        The ground truth flow of shape (2, H, W), where axis 0 corresponds to x and y flows
        respectively.
    timestamps : np.ndarray
        The ground truth timestamps of shape ('num_frames').
    """
    flow = None
    timestamps = None

    flow = gt_file
    for dir in flow_group:
        flow = flow[dir]
    flow = np.array(flow)

    timestamps = gt_file
    for dir in timestamps_group:
        timestamps = timestamps[dir]
    timestamps = np.array(timestamps)

    return flow, timestamps

def clip_data(
    start_time: float,
    end_time: float,
    events: np.ndarray,
    nearest_events: np.ndarray = None,
    grayscale_images: np.ndarray = None,
    grayscale_timestamps: np.ndarray = None
) -> Tuple[np.ndarray]:
    """
    Clips a sequence of event-based data with an initial and final timestamp (in seconds).
    Does not correct 'nearest_events' values.
    NOTE: DEPRECATED FUNCTION.

    Parameters
    ----------
    start_time : float
        Initial timestamp (in seconds).
    end_time : float
        Final timestamp (in seconds).
    events : np.ndarray
        Events data of size ('num_events', 4), where axis 1 corresponds to
        (x, y, t, p).
    nearest_events : np.ndarray
        Nearest events of size ('num_grayscale_images') (defaults to None).
    grayscale_images : np.ndarray
        Grayscale images of size ('num_grayscale_images', H, W) (defaults to None).
    grayscale_timestamps : np.ndarray
        Grayscale timestamps of size ('num_grayscale_images') (defaults to None).

    Returns
    -------
    _events : np.ndarray
        Events data of size ('num_events', 4), where axis 1 corresponds to
        (x, y, t, p).
    _nearest_events : np.ndarray
        Nearest events of size ('num_grayscale_images'). None is returned
        instead if no 'nearest_events_group' was given as input.
    _grayscale_images : np.ndarray
        Grayscale images of size ('num_grayscale_images', H, W). None is returned
        instead if no 'grayscale_images_group' was given as input.
    _grayscale_timestamps : np.ndarray
        Grayscale timestamps of size ('num_grayscale_images'). None is returned
        instead if no 'grayscale_timestamps_group' was given as input.
    """
    _events = None
    _nearest_events = None
    _grayscale_images = None
    _grayscale_timestamps = None

    # Clip events
    idx0 = np.searchsorted(events[:, 2], start_time)
    idx1 = np.searchsorted(events[:, 2], end_time)
    _events = events[idx0:idx1]

    # Clip grayscale images
    if grayscale_timestamps is not None:
        idx0 = np.searchsorted(grayscale_timestamps, start_time)
        idx1 = np.searchsorted(grayscale_timestamps, end_time)

        # For small durations
        if idx0 == idx1:
            _nearest_events = nearest_events[idx0][None, ...]
            _grayscale_images = grayscale_images[idx0][None, ...]
            _grayscale_timestamps = grayscale_timestamps[idx0][None, ...]
        else:
            _nearest_events = nearest_events[idx0:idx1]
            _grayscale_images = grayscale_images[idx0:idx1]
            _grayscale_timestamps = grayscale_timestamps[idx0:idx1]

    return _events, _nearest_events, _grayscale_images, _grayscale_timestamps


# ========================================= #
# ----- Reader Functions, Version 2.0 ----- #
# ========================================= #
class DataReader():
    """
    The data reader class unpacks '.hdf5' files containing data and ground truth flows. The
    data is then saved into internal object variables which can be accessed from outside.

    Attributes
    ----------
    data_dir : str
        The directory of the dataset '.hdf5' file.
    gt_dir : str
        The directory of the ground truth '.hdf5' file (defaults to None). In case no
        ground truth file is provided the reader still works with only events data.
    data_hdf5_groups : Dict[str, List]
        The '.hdf5' group names of each data format at the end of the hierarchy. An example
        structure can be found below:
            data_hdf5_groups = {
                'events': ['dvs', 'left', 'events'],
                'nearest_events': ['dvs', 'left', 'nearest_events'],
                'gray_images': ['dvs', 'left', 'gray_images'],
                'gray_timestamps': ['dvs', 'left', 'gray_timestamps']
            }
    gt_hdf5_groups : Dict[str, List]
        The '.hdf5' group names of each ground truth format at the end of the hierarchy.
        An example structure can be found below:
            gt_hdf5_groups = {
                'flows': ['dvs', 'left', 'flows'],
                'timestamps': ['dvs', 'left', 'timestamps']
            }
    start_time : float
        In case you want to clip the sequence, provide a start time and end time. Leave
        empty if you do not want to clip (defaults to None).
    end_time : float
        In case you want to clip the sequence, provide a start time and end time. Leave
        empty if you do not want to clip (defaults to None).
    reset_time : bool
        In case the sequence does not start with a timestamp of 0, you can reset it (defaults
        to False).
    num_slices : int
        When clipping huge datasets, the reader loads data to memory in chunks. This variable
        dictates the number of chunks to use. The higher the value the better the memory
        management, the slower the algorithm (defaults to 10).

    Data Variables
    --------------
    events : Union[np.ndarray, h5py.Dataset]
        The events data which consists of one array of shape ('num_events', 4), where the
        second dimension corresponds to (x, y, t, p) respectively. The timestamps may appear
        as big values since they are recorded with ROS time. As for the polarities they are
        either -1 or +1, depending on the direction of intensity change relative to each event.
    nearest_events : Union[np.ndarray, h5py.Dataset]
        The term 'nearest events' alludes to the index of the event in the 'events' array
        that is closest to each grayscale frame. The data is a 1D array of shape ('num_frames'),
        it is always equal to the number of available grayscale frames.
    gray_images : Union[np.ndarray, h5py.Dataset]
        The extracted grayscale images which consist of one array of shape ('num_frames', H, W).
        The grayscale images are extracted in sequence, from the first frame till the last one
        in the dataset.
    gray_timestamps : Union[np.ndarray, h5py.Dataset]
        The timestamp of each grayscale image in the form of a 1D array of shape ('num_frames'),
        in which the index of each element relates to its corresponding frame in the 'gray_images'
        array.
    gt_flows : Union[np.ndarray, h5py.Dataset]
        The ground truth flows across the sequence. This dataset if of shape ('num_gt_flows', H, W).
        Note that the ground truth data may not be synchronized and might need a ground truth
        synchronizer script for it to work correctly.
    gt_timestamps : Union[np.ndarray, h5py.Dataset]
        The ground truth timestamps of shape ('num_gt_flows'). These timestamps can be used
        for correct synchronization.
    """
    def __init__(
        self,
        data_dir: str,
        gt_dir: str = None,
        data_hdf5_groups: Dict[str, List] = None,
        gt_hdf5_groups: Dict[str, List] = None,
        start_time: float = None,
        end_time: float = None,
        reset_time: bool = False,
        num_slices: int = 10
    ) -> None:
        # Data variables
        self.events = None
        self.nearest_events = None
        self.gray_images = None
        self.gray_timestamps = None
        self.gt_flows = None
        self.gt_timestamps = None

        # Reader variables
        self._data_dir = data_dir
        self._gt_dir = gt_dir
        self._data_hdf5_groups = data_hdf5_groups
        self._gt_hdf5_groups = gt_hdf5_groups
        self._start_time = start_time
        self._end_time = end_time
        self._reset_time = reset_time
        self._num_slices = num_slices

        # Read data
        self._data = self.read_data(
            data_dir=self._data_dir,
            hdf5_groups=self._data_hdf5_groups
        )
        self._gt = self.read_gt(
            gt_dir=self._gt_dir,
            hdf5_groups=self._gt_hdf5_groups
        )

        # Clip data
        if self._start_time is not None:
            self._min_time = self._get_min_timestamp()
            self._clip_data()

        # Update data
        self.events = self._data["events"]
        self.nearest_events = self._data["nearest_events"]
        self.gray_images = self._data["gray_images"]
        self.gray_timestamps = self._data["gray_timestamps"]
        self.gt_flows = self._gt["flows"]
        self.gt_timestamps = self._gt["timestamps"]

        # Clear data
        self._data = None
        self._gt = None

    @staticmethod
    def read_data(data_dir: str, hdf5_groups: Dict[str, List]) -> Dict[str, h5py.Dataset]:
        """
        Reads a '.hdf5' data file and divides its components. Can be used as a static
        method.

        Parameters
        ----------
        data_dir : str
            Directory of '.hdf5' file.
        hdf5_groups : Dict[str, List]
            The '.hdf5' group names of each data format at the end of the hierarchy.
            An example structure can be found below:
                {
                    'events': ['dvs', 'left', 'events'],
                    'nearest_events': ['dvs', 'left', 'nearest_events'],
                    'gray_images': ['dvs', 'left', 'gray_images'],
                    'gray_timestamps': ['dvs', 'left', 'gray_timestamps']
                }

        Returns
        -------
        data : Dict[str, h5py.Dataset]
            The data organized in a dictionary. An example can be found below:
                {
                    'events': DATA,
                    'nearest_events': DATA,
                    'gray_images': DATA,
                    'gray_timestamps': DATA
                }
        """
        data_file = h5py.File(data_dir, "r")
        data = {
            "events": None,
            "nearest_events": None,
            "gray_images": None,
            "gray_timestamps": None
        }

        events = data_file
        for dir in hdf5_groups["events"]:
            events = events[dir]
        data["events"] = events

        if hdf5_groups["nearest_events"] is not None:
            nearest_events = data_file
            for dir in hdf5_groups["nearest_events"]:
                nearest_events = nearest_events[dir]
            data["nearest_events"] = nearest_events

        if hdf5_groups["gray_images"] is not None:
            gray_images = data_file
            for dir in hdf5_groups["gray_images"]:
                gray_images = gray_images[dir]
            data["gray_images"] = gray_images

        if hdf5_groups["gray_timestamps"] is not None:
            gray_timestamps = data_file
            for dir in hdf5_groups["gray_timestamps"]:
                gray_timestamps = gray_timestamps[dir]
            data["gray_timestamps"] = gray_timestamps

        return data

    @staticmethod
    def read_gt(gt_dir: str, hdf5_groups: Dict[str, List]) -> Dict[str, h5py.Dataset]:
        """
        Reads a '.hdf5' ground truth file and divides its components. Can be used as a static
        method.

        Parameters
        ----------
        data_dir : str
            Directory of '.hdf5' file.
        hdf5_groups : Dict[str, List]
            The '.hdf5' group names of each data format at the end of the hierarchy.
            An example structure can be found below:
                {
                    'flows': ['dvs', 'left', 'flows'],
                    'timestamps': ['dvs', 'left', 'timestamps']
                }

        Returns
        -------
        gt : Dict[str, h5py.Dataset]
            The ground truth organized in a dictionary. An example can be found below:
                {
                    'flows': DATA,
                    'timestamps': DATA
                }
        """
        gt_file = h5py.File(gt_dir, "r")
        gt = {
            "flows": None,
            "timestamps": None
        }

        flows = gt_file
        for dir in hdf5_groups["flows"]:
            flows = flows[dir]
        gt["flows"] = flows

        timestamps = gt_file
        for dir in hdf5_groups["timestamps"]:
            timestamps = timestamps[dir]
        gt["timestamps"] = timestamps

        return gt

    def _clip_data(self) -> None:
        """
        Clips events and ground truth sequences between two timestamps of choice, this method
        requires class initialization of the 'DataReader' object.
        """
        # Clip events
        start_idx, end_idx = self._search_data(data=self._data["events"])
        self._data["nearest_events"] = self._data["nearest_events"] - start_idx
        self._data["events"] = self._extract_data(
            data=self._data["events"],
            start_idx=start_idx,
            end_idx=end_idx
        )

        # Clip grayscale images
        if self._data["gray_timestamps"] is not None:
            start_idx, end_idx = self._search_data(data=self._data["gray_timestamps"])
            self._data["nearest_events"] = self._data["nearest_events"][start_idx:end_idx]
            self._data["gray_images"] = self._data["gray_images"][start_idx:end_idx, ...]
            self._data["gray_timestamps"] = self._extract_data(
                data=self._data["gray_timestamps"],
                start_idx=start_idx,
                end_idx=end_idx
            )

        # Clip ground truth flow
        if self._gt["timestamps"] is not None:
            start_idx, end_idx = self._search_data(data=self._gt["timestamps"])
            self._gt["flows"] = self._gt["flows"][start_idx:end_idx, ...]
            self._gt["timestamps"] = self._extract_data(
                data=self._gt["timestamps"],
                start_idx=start_idx,
                end_idx=end_idx
            )

    def _search_data(self, data: h5py.Dataset) -> Tuple[int, int]:
        """
        Applies a sorted search algorithm similar to the one of Numpy without loading all
        data in memory.

        Parameters
        ----------
        data : h5py.Dataset
            Data to be clipped, can be 1-dimensional or 2-dimensional depending if the data
            contains events or not.

        Returns
        -------
        start_idx : int
            Initial index to clip the data.
        end_idx : int
            Final index to clip the data.
        """
        num_timestamps = data.shape[0]
        slice_size = num_timestamps//self._num_slices
        ids = np.arange(0, num_timestamps, slice_size)

        sum_idx = 0
        for i in ids:
            if len(data.shape) == 1:
                data_slice: np.ndarray = data[i:i + slice_size]
            else:
                data_slice = data[i:i + slice_size, 2]
            occur_idx = np.argmax(data_slice > (self._start_time + self._min_time))

            if occur_idx != 0:
                start_idx = sum_idx + occur_idx
                break

            sum_idx += data_slice.shape[0]

        sum_idx = 0
        for i in ids:
            if len(data.shape) == 1:
                data_slice: np.ndarray = data[i:i + slice_size]
            else:
                data_slice = data[i:i + slice_size, 2]
            occur_idx = np.argmax(data_slice > (self._end_time + self._min_time))

            if occur_idx != 0:
                end_idx = sum_idx + occur_idx
                break

            sum_idx += data_slice.shape[0]

        return start_idx, end_idx

    def _get_min_timestamp(self) -> float:
        """
        Returns the global minimum timestamp across all data.

        Returns
        -------
        min_time : float
            Minimum timestamp (in seconds).
        """
        min_timestamps = []
        min_timestamps.append(self._data["events"][0, 2])

        if self._data["gray_timestamps"] is not None:
            min_timestamps.append(self._data["gray_timestamps"][0])

        if self._gt["timestamps"] is not None:
            min_timestamps.append(self._gt["timestamps"][0])

        min_timestamps = np.array(min_timestamps)
        min_time = np.min(min_timestamps)

        return min_time

    def _extract_data(
        self,
        data: h5py.Dataset,
        start_idx: int,
        end_idx: int
    ) -> Union[np.ndarray, h5py.Dataset]:
        """
        Extract data in which timestamps are present, automatically resets the timestamps
        if required by the user.
        """
        data_slice = data[start_idx:end_idx, ...]

        # All data needs to be loaded into memory to modify the timestamps,
        # might not be very efficient if the data stream is too big
        if self._reset_time is True:
            data_slice = np.array(data_slice, dtype=np.float64)

            if len(data.shape) == 1:
                data_slice = data_slice - self._min_time
            else:
                data_slice[:, 2] = data_slice[:, 2] - self._min_time

            return data_slice

        return data_slice


# ===== Example Code ===== #
if __name__ == "__main__":
    # ----- Example Code for Version 1.0 ----- #
    # Directory variables
    data_dir = ""
    gt_dir = ""

    data_file = h5py.File(data_dir, "r")
    data = read_data(
        data_file=data_file,
        events_group=["davis", "left", "events"],
        nearest_events_group=["davis", "left", "image_raw_event_inds"],
        grayscale_images_group=["davis", "left", "image_raw"],
        grayscale_timestamps_group=["davis", "left", "image_raw_ts"],
        reset_time=True
    )

    # Data variables
    events = data[0]
    nearest_events = data[1]
    grayscale_images = data[2]
    grayscale_timestamps = data[3]

    data = clip_data(
        start_time=0.4,
        end_time=1.0,
        events=events,
        nearest_events=nearest_events,
        grayscale_images=grayscale_images,
        grayscale_timestamps=grayscale_timestamps
    )

    # Data variables
    events = data[0]
    nearest_events = data[1]
    grayscale_images = data[2]
    grayscale_timestamps = data[3]

    # Print data
    print("Events Timestamps:", events[:, 2])
    print("Grayscale Timestamps:", grayscale_timestamps)

    # ----- Example Code for Version 2.0 ----- #
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

    # Print data
    print("Events Timestamps:", data_reader.events[:, 2])
    print("Grayscale Timestamps:", data_reader.gray_timestamps[:])
