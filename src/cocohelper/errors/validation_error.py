"""Exception raised when assertion of dataset validity fails.
"""

class COCOValidationError(Exception):
    def __init__(self):
        """Exceptions raised when the input COCO is not in a valid COCO format."""
        super(COCOValidationError, self).__init__("The dataset has an invalid COCO format")
