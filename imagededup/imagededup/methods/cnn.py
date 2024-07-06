def _get_cnn_features_batch(
        self,
        image_dir: PurePath,
        recursive: Optional[bool] = False,
        num_workers: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Generate CNN encodings for all images in a given directory of images.
        Args:
            image_dir: Path to the image directory.
            recursive: Optional, find images recursively in a nested image directory structure.
            num_workers: Optional, number of cpu cores to use for multiprocessing encoding generation (supported only on linux platform), set to 0 by default. 0 disables multiprocessing.

        Returns:
            A dictionary that contains a mapping of filenames and corresponding numpy array of CNN encodings.
        """
        self.logger.info("Start: Image encoding generation")
        self.dataloader = img_dataloader(
            image_dir=image_dir,
            batch_size=self.batch_size,
            basenet_preprocess=self.apply_preprocess,
            recursive=recursive,
            num_workers=num_workers,
        )

        feat_arr, all_filenames = [], []
        bad_im_count = 0

        with torch.no_grad():
            for ims, filenames, bad_images in self.dataloader:
                arr = self.model(ims.to(self.device))
                feat_arr.extend(arr)
                all_filenames.extend(filenames)
                if bad_images:
                    bad_im_count += 1

        if bad_im_count:
            self.logger.info(
                f"Found {bad_im_count} bad images, ignoring for encoding generation .."
            )

        feat_vec = torch.stack(feat_arr).squeeze()
        feat_vec = (
            feat_vec.detach().numpy()
            if self.device.type == "cpu"
            else feat_vec.detach().cpu().numpy()
        )
        valid_image_files = [filename for filename in all_filenames if filename]
        self.logger.info("End: Image encoding generation")

        filenames = generate_relative_names(image_dir, valid_image_files)
        if (
            len(feat_vec.shape) == 1
        ):  # can happen when encode_images is called on a directory containing a single image
            self.encoding_map = {filenames[0]: feat_vec}
        else:
            self.encoding_map = {j: feat_vec[i, :] for i, j in enumerate(filenames)}
        return self.encoding_map

def _check_threshold_bounds(thresh: float) -> None:
        """
        Check if provided threshold is valid. Raises TypeError if wrong threshold variable type is passed or a
        ValueError if an out of range value is supplied.

        Args:
            thresh: Threshold value (must be float between -1.0 and 1.0)

        Raises:
            TypeError: If wrong variable type is provided.
            ValueError: If wrong value is provided.
        """
        if not isinstance(thresh, float):
            raise TypeError("Threshold must be a float between -1.0 and 1.0")
        if thresh < -1.0 or thresh > 1.0:
            raise ValueError("Threshold must be a float between -1.0 and 1.0")

