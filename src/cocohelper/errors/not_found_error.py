class COCOImageNotFoundError(Exception):

    def __init__(
            self,
            image_id: int
    ):
        """Error raised when a certain image id does not exist in a dataset.

        Args:
            image_id: the id of the image.
        """
        super(COCOImageNotFoundError, self).__init__(f"Image id {image_id} does not exist in the dataset")
        self.image_id = image_id


class COCOAnnotationNotFoundError(Exception):

    def __init__(
            self,
            ann_id: int
    ):
        """Error raised when a certain annotation id does not exist in a dataset.

        Args:
            ann_id: the id of the annotation.
        """
        super(COCOAnnotationNotFoundError, self).__init__(f"Annotation id {ann_id} does not exist in the dataset")
        self.ann_id = ann_id
