import numpy as np
from skimage import measure
from typing import Sequence, Union





class Segmentation2DMetrics:
    """Wraps a segmentation map and implements algorithms that operate on it to compute metrics."""

    def __init__(
        self,
        segmentation: np.ndarray,
        struct_labels: Sequence[Union[int, Sequence[int]]],		
    ):
        """Initializes class instance.

        Args:
            segmentation: (H, W), 2D array where the value of each entry in the array is the label of the segmentation
                class for the pixel.
            struct_labels: Label(s) of the class(es) present in the segmentation for which to compute metrics.
        """
        
        self.segmentation = segmentation
        #self.binary_structs = {
            #struct_label: np.isin(segmentation, struct_label).astype(dtype=np.uint8) for struct_label in struct_labels
        #}
        self.binary_structs = {struct_labels: np.isin(segmentation, struct_labels).astype(dtype=np.uint8)}

        self.binary_structs_inverse = {
            struct_label: 1 - binary_struct for struct_label, binary_struct in self.binary_structs.items()
        }

										
    def count_holes(self, struct_label: Union[int, Sequence[int]]) -> int:
        """Counts the pixels that form holes in a supposedly contiguous segmented area.

        Args:
            struct_label: Label(s) of the class(es) making up the segmented area for which to compute the metric.

        Returns:
            Number of pixels that form holes in the segmented area.
        """
        # Mark the class of interest as 0 and everything else as 1
        # Merge the regions of 1 that are open by a side using padding (these are not holes)
        binary_struct = self.binary_structs_inverse[struct_label]
        binary_struct = np.pad(binary_struct, ((1, 1), (1, 1)), "constant", constant_values=1)

        # Extract properties of continuous regions of 1
        props = measure.regionprops(measure.label(binary_struct, connectivity=2))

        hole_pixel_count = 0
        for prop in props:
            # Skip the region open by the side (the one that includes padding)
            if prop.bbox[0] != 0:
                hole_pixel_count += prop.area

        return hole_pixel_count


    def count_disconnectivity(self, struct_label: Union[int, Sequence[int]]) -> int:
        """Counts the pixels that are disconnected from a supposedly contiguous segmented area.

        Args:
            struct_label: Label(s) of the class(es) making up the segmented area for which to compute the metric.

        Returns:
            Total number of disconnected pixels in the segmented area.
        """
        binary_struct = self.binary_structs[struct_label]

        # Extract properties for every disconnected region of the supposedly continuous segmented area
        labels_props = measure.regionprops(measure.label(binary_struct, connectivity=2))

        labels_props_by_desc_size = sorted(labels_props, key=lambda k: k.area, reverse=True)

        total_minus_biggest = 0
        for label_props in labels_props_by_desc_size:
            if label_props.area < self.segmentation.shape[0]*self.segmentation.shape[1]/1000:  
              # different threshold
              # print("Area", label_props.area, (self.segmentation.shape[0] * self.segmentation.shape[1]) / 1000)

              return label_props.area

        return 0


    def count_disconnectivity_big(self, struct_label: Union[int, Sequence[int]]) -> int:
        """ As for the bigger disconneted objects, we need to use VAE transform them

        Args:
            struct_label: Label(s) of the class(es) making up the segmented area for which to compute the metric.

        Returns:
            Total number of disconnected pixels in the segmented area.
        """
        binary_struct = self.binary_structs[struct_label]

        # Extract properties for every disconnected region of the supposedly continuous segmented area
        labels_props = measure.regionprops(measure.label(binary_struct, connectivity=2))

        labels_props_by_desc_size = sorted(labels_props, key=lambda k: k.area, reverse=True)

        total_minus_biggest = 0
        for label_props in labels_props_by_desc_size:
            if (self.segmentation.shape[0] * self.segmentation.shape[1]) / 1000 < label_props.area < \
                (self.segmentation.shape[0] * self.segmentation.shape[1]) / 300:

                # print("show", label_props.area,self.segmentation.shape[0]*self.segmentation.shape[1]/300)
                return label_props.area

        return 0






