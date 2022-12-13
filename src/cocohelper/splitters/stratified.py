from typing import List
import random
from cocohelper.splitters.proportional import ProportionalDataSplitter
from cocohelper import COCOHelper

# TODO: clean code in this file:
#  - Functions are too long.
#  - Function nesting is too high (StratifiedDataSplitter has 5 nested sections)



class StratifiedDataSplitter(ProportionalDataSplitter):

    def _get_ids(
            self,
            ch: COCOHelper
    ) -> List:
        """Get the ids needed for the stratified dataset splitting.

        Args:
            ch: a COCOHelper with the source COCO dataset.

        Returns:
            A list of ids for each subset.
        """
        n_subsets = len(self.proportions)
        subset_ratios = self.proportions

        # normalise proportions in 0-1:
        if len(subset_ratios) != n_subsets:
            raise ValueError("The number of values for the ratios must be equal to the number of subsets 'n_subsets'")

        # initialise dictionary to track subset content
        dataset: List[dict] = list()
        for sset in range(n_subsets):
            dataset.append({
                "ids": [],
                "desired_size": -1,
                "desired_size_for_label": dict()
            })

        # ----
        # 0) get image ids grouped by label
        imgs_anns = ch.joins.imgs_anns.fillna(-1)
        ids_by_label = {k: list(set(v)) for k, v in imgs_anns.reset_index().groupby('category_id')['image_id']}

        # ----
        # 1) Compute the desired number of examples in each subset:
        n_samples = len(ch.imgs)
        for sset, r in enumerate(subset_ratios):
            dataset[sset]["desired_size"] = r * n_samples  # ok if it is float...

        # ----
        # 2) Compute the desired number of samples of each label at each subset:
        label_ratios = self._compute_label_ratios(ids_by_label)
        for sset in range(n_subsets):
            for lbl in label_ratios.keys():
                dataset[sset]["desired_size_for_label"][lbl] = dataset[sset]["desired_size"] * label_ratios[lbl]

        # ----
        # 3) Iterative assignment:
        while n_samples > 0:

            # select the label with the fewest remaining samples:
            n_min = min([len(v) for v in ids_by_label.values() if len(v) >= 1])
            min_labels = [lbl for lbl in ids_by_label.keys() if len(ids_by_label[lbl]) == n_min]
            selected_label = random.choice(min_labels)

            # for each example of this label: assign to the best subset:
            for img_id in ids_by_label[selected_label]:

                # find the subset with the largest number of desired examples for this label
                selected_subset = 0  # initialise with first subset (e.g. train)
                n_max = dataset[selected_subset]["desired_size_for_label"][selected_label]
                for sset in range(1, n_subsets):  # iterate over the remaining subsets

                    if dataset[sset]["desired_size_for_label"][selected_label] > n_max:
                        selected_subset = sset

                    # breaking ties:
                    elif dataset[sset]["desired_size_for_label"][selected_label] == n_max:
                        if dataset[sset]["desired_size"] > dataset[selected_subset]["desired_size"]:
                            selected_subset = sset
                        elif dataset[sset]["desired_size"] == dataset[selected_subset]["desired_size"]:
                            selected_subset = random.choice([selected_subset, sset])

                # assign the element to the selected subset:
                dataset[selected_subset]["ids"].append(img_id)

                # remove this image from the dataset (i.e. remove it from the remaining lists)
                for lbl in ids_by_label.keys():
                    ids_by_label[lbl] = list(filter(lambda _id: _id != img_id, ids_by_label[lbl]))

                # decrease the number of desired samples for each label of this sample (inside selected_subset)
                sample_labels = [lbl for lbl in ch.filtered_anns(img_ids=img_id)['category_id'].tolist()]
                for lbl in sample_labels:
                    dataset[selected_subset]["desired_size_for_label"][lbl] -= 1

                # decrease counter
                n_samples -= 1

        return [dataset[sset]["ids"] for sset in range(n_subsets)]

    @staticmethod
    def _compute_label_ratios(
            images_by_label: dict
    ) -> dict:
        """Computes the ratio of labels within the COCO dataset.

        Args:
            images_by_label: a dictionary grouping images by their labels.

        Returns:
            A dictionary with the ratios of labels in the COCO dataset.
        """
        ratios = dict()
        for k, v in zip(images_by_label.keys(), images_by_label.values()):
            ratios[k] = float(len(v))
        tot_number = sum(list(ratios.values()))
        if tot_number <= 0:
            raise ValueError("The sum of ratio values must be greater than zero.")
        for k, v in zip(ratios.keys(), ratios.values()):
            ratios[k] = float(v) / tot_number
        return ratios
