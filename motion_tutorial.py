"""
Tutorial on how to apply motion compensation.
"""

import cv2
import h5py
import tonic
import numpy as np
import matplotlib.pyplot as plt


def keep_windows_open() -> None:
    """
    Keeps visualization windows open till 'ESC' key press.
    """
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            break


# ===== Tutorial Code ===== #
if __name__ == "__main__":
    # Data reader imports
    from ewiz.core.reader import read_data, read_gt, clip_data
    from ewiz.core.sync import GroundTruthSynchronizer

    # ===== Replace variables here ===== #
    """
    Fill the empty string cells below. Choose your data directories, and '.hdf5' groups as
    stated in the lab's given.
    """
    # Directory and clip variables
    data_dir = "hop_events_session2/indoor_flying1_data.hdf5"
    gt_dir = "hop_events_session2/indoor_flying1_gt.hdf5"
    clip = [10, 10.2]

    # ----- Read dataset ----- #
    data_file = h5py.File(data_dir, "r")
    data = read_data(
        data_file=data_file,
        events_group=["davis", "left", "events"],
        nearest_events_group=["davis", "left", "image_raw_event_inds"],
        grayscale_images_group=["davis", "left", "image_raw"],
        grayscale_timestamps_group=["davis", "left", "image_raw_ts"],
        reset_time=False
    )
    data_file = None

    events = data[0]
    nearest_events = data[1]
    grayscale_images = data[2]
    grayscale_timestamps = data[3]

    # Save start time
    time_idx = events[0, 2]

    # Clip data
    data = clip_data(
        start_time=clip[0] + time_idx,
        end_time=clip[1] + time_idx,
        events=events,
        nearest_events=nearest_events,
        grayscale_images=grayscale_images,
        grayscale_timestamps=grayscale_timestamps
    )

    events = data[0]
    nearest_events = data[1]
    grayscale_images = data[2]
    grayscale_timestamps = data[3]
    # ------------------------ #

    # ----- Read Ground Truth Flows ----- #
    gt_file = h5py.File(gt_dir, "r")

    gt = read_gt(
        gt_file=gt_file,
        flow_group=["davis", "left", "flow_dist"],
        timestamps_group=["davis", "left", "flow_dist_ts"]
    )
    gt_flows = gt[0]
    gt_timestamps = gt[1]
    gt_synchronizer = GroundTruthSynchronizer(flows=gt_flows, timestamps=gt_timestamps)
    gt_flow = gt_synchronizer.sync(
        start_time=clip[0] + time_idx,
        end_time=clip[1] + time_idx
    )
    """
    NOTE: BELOW IS THE GROUND TRUTH FLOW YOU WILL BE USING FOR VALIDATION.
    THE FLOW HAS A SHAPE OF (2, H, W), THE FIRST INDEX OF DIMENSION 0 IS THE FLOW IN X,
    THE SECOND INDEX OF DIMENSION 0 IS THE FLOW IN Y.
    """
    gt_flow = gt_flow/(events[-1, 2] - events[0, 2])
    # -----------------------
    
    # ----------------------------------- #
    # ================================== #


    # ===== Apply motion compensation ===== #
    """
    NOTE: BELOW IS THE PREDICTED FLOW GIVEN BY MOTION COMPENSATION.
    THE FLOW HAS A SHAPE OF (2, H, W), THE FIRST INDEX OF DIMENSION 0 IS THE FLOW IN X,
    THE SECOND INDEX OF DIMENSION 0 IS THE FLOW IN Y.
    """
    comp_flow = None

    from ewiz.losses.loss import MotionCompensationLoss
    from ewiz.solvers.pyramidal import PyramidalPatchMotionCompensation

    # ----- Crop images ----- #
    """
    Before applying the algorithm we have to crop the images into square equivalents.
    We choose an output size of (256, 256).
    """
    
    from ewiz.transforms.events import EventsCenterCropUns
    from ewiz.transforms.flow import FlowCenterCrop

    events_transforms = tonic.transforms.Compose([
        EventsCenterCropUns(sensor_size=(346, 260), out_size=(256, 256))
    ])
    flow_transforms = tonic.transforms.Compose([
        FlowCenterCrop(out_size=(256, 256))
    ])

    # Transformed flow and events
    events = events_transforms(events)
    gt_flow = flow_transforms(gt_flow)
    # ----------------------- #

    # Loss function initialization
    loss_function = MotionCompensationLoss(
        image_size=(256, 256),
        losses=[
            "multifocal_normalized_image_variance",
            "multifocal_normalized_gradient_magnitude",
            "regularizer"
        ],
        weights=[1.0, 1.0, 0.01],
        gt_flow=gt_flow
    )

    # # Initialize optimizer
  
           
    # # pyramidal_optimizer = PyramidalPatchMotionCompensation(
    # #     image_size=(256, 256),
    # #     optimizer=['BFGS', 'CG', 'Newton-CG'],
    # #     init_method="random",
    # #     loss_function=loss_function,
    # #     scale_range=(1, 5)
    # # )

    #     # Run optimization
    #     patch_flows, optimizer_results = pyramidal_optimizer.optimize(events=events)
    #     # ===================================== #

    #     # ===== Metric Evaluation ===== #
    #     """
    #     IMPLEMENT THE AEE METRIC FUNCTION HERE.
    #     NOTE: REMEMBER BELOW IS THE PREDICTED FLOW.
    #     """
    #     comp_flow = pyramidal_optimizer.dense_flow.cpu().detach().numpy()

    def get_average_endpoint_error(
        gt_flow: np.ndarray = None,
        comp_flow: np.ndarray = None
    ) -> float:
        """
        Compute the Average Endpoint Error.

        Parameters
        ----------
        gt_flow : np.ndarray
            Ground truth flow of shape (2, H, W).
        comp_flow : np.ndarray
            Compensated flow of shape (2, H, W).
        """
        # Compute over valid points in ground truth data
        mask_flow = np.logical_and(
            np.logical_and(~np.isinf(gt_flow[[0], ...]), ~np.isinf(gt_flow[[1], ...])),
            np.logical_and(np.abs(gt_flow[[0], ...]) > 0, np.abs(gt_flow[[1], ...]) > 0)
        )
        gt_flow = gt_flow*mask_flow
        comp_flow = comp_flow*mask_flow
        # Calculate the Euclidean distance between the predicted and ground truth vectors
        distance = np.linalg.norm(gt_flow - comp_flow, axis=0)

        # Average these distances over all pixels
        aee = np.sum(distance) / np.count_nonzero(mask_flow)
        return aee
    
    def evaluate_optimizers(
        optimizers: list,
        events: np.ndarray,
        gt_flow: np.ndarray,
        loss_function,
        image_size: tuple = (256, 256),
        init_method: str = "random",
        scale_range: tuple = (1,5)
    ) -> dict:
        results = {}
        for optimizer in optimizers:
            print(f"Evaluating optimizer: {optimizer}")

            # Initialize the motion compensation optimizer with the current optimizer
            pyramidal_optimizer = PyramidalPatchMotionCompensation(
                image_size=image_size,
                optimizer=optimizer,
                init_method=init_method,
                loss_function=loss_function,
                scale_range=scale_range
            )

            # Run optimization
            patch_flows, optimizer_results = pyramidal_optimizer.optimize(events=events)

            # Assume pyramidal_optimizer now holds the compensated flow as a result of optimization
            comp_flow = pyramidal_optimizer.dense_flow.cpu().detach().numpy()

            # Compute AEE
            aee = get_average_endpoint_error(gt_flow=gt_flow, comp_flow=comp_flow)
            
            # Store the results
            results[optimizer] = {
                "AEE": aee,
                "Loss": loss_function.loss
            }
            title = f"Optimizer: {optimizer} - AEE: {results[optimizer]['AEE']:.4f}, Loss: {results[optimizer]['Loss']:.4f}"
            
        return results

 
   
    optimizers_to_evaluate = [
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP',
    'trust-constr',
    'dogleg',
    'trust-ncg',
    'trust-krylov',
    'trust-exact'
]

    evaluation_results = evaluate_optimizers(
        optimizers=optimizers_to_evaluate,
        events=events,
        gt_flow=gt_flow,
        loss_function=loss_function
    )
    min_loss = float('inf')  # Initialize with a very high value
    best_optimizer = None

    # Output the results for comparison
    for optimizer,  metrics in evaluation_results.items():
        print(f"Optimizer: {optimizer}, Loss: {metrics['Loss']}, AEE: {metrics['AEE']}")
 

    # Iterate through the results to find the optimizer with the minimum loss

        if metrics['Loss'] < min_loss:
            min_loss = metrics['Loss']
            best_optimizer = optimizer

    # Output the best optimizer based on the minimum loss
    print(f"The best optimizer based on minimum loss is: {best_optimizer} with a loss of {min_loss}")
         
         
       
         
    # Compute AEE here
    # aee=None
    # aee = get_average_endpoint_error(gt_flow=gt_flow, comp_flow=comp_flow)
    # loss = loss_function.loss
    # print("Metrics,", "LOSS:", loss, "AEE:", aee)

    # ============================= #
    keep_windows_open()



