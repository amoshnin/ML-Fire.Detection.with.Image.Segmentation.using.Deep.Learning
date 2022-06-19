"""
    File Name : VisualizeMask.py
    File Description : VisualizeMask class for putting the bounding box and mask over the image to see the results
"""
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Root directory of the project

# Import Mask RCNN
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import log

from .References import References


class VisualizeMask(References):

    def get_ax(self, rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

    def predict_and_display(self, dataset, config, model):
        """ Run Detection of Mask """

        # RUN DETECTION
        image_id = random.choice(dataset.image_ids)
        print(image_id)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))

        # Run object detection
        results = model.detect([image], verbose=1)

        # Display results
        ax = self.get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)

        # This is for predicting images which are not present in dataset
        image1 = mpimg.imread(self.ROOT_DIR + '/input/dataset/test/249.jpg')

        # Run object detection
        print(len([image1]))
        results1 = model.detect([image1], verbose=1)

        # Display results
        ax = self.get_ax(1)
        r1 = results1[0]
        visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],
                                    dataset.class_names, r1['scores'], ax=ax,
                                    title="Predictions1")
