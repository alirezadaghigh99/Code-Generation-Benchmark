def find_duplicates(
        self,
        image_dir: PurePath = None,
        encoding_map: Dict[str, str] = None,
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
        recursive: Optional[bool] = False,
        num_enc_workers: int = cpu_count(),
        num_dist_workers: int = cpu_count()
    ) -> Dict:
        """
        Find duplicates for each file. Takes in path of the directory or encoding dictionary in which duplicates are to
        be detected. All images with hamming distance less than or equal to the max_distance_threshold are regarded as
        duplicates. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the below the given hamming distance could be returned instead of just duplicate filenames for each
        query file.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            encoding_map: Optional,  used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding hashes.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            scores: Optional, boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            duplicates dictionary: if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
                        score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}. if scores is False, then a
                        dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'],
                        'image2.jpg':['image1_duplicate1.jpg',..], ..}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True,
        outfile='results.json')

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, scores=True, outfile='results.json')
        ```
        """
        self._check_hamming_distance_bounds(thresh=max_distance_threshold)
        if image_dir:
            result = self._find_duplicates_dir(
                image_dir=image_dir,
                max_distance_threshold=max_distance_threshold,
                scores=scores,
                outfile=outfile,
                search_method=search_method,
                recursive=recursive,
                num_enc_workers=num_enc_workers,
                num_dist_workers=num_dist_workers
            )
        elif encoding_map:
            if recursive:
                warnings.warn('recursive parameter is irrelevant when using encodings.', SyntaxWarning)
            
            warnings.warn('Parameter num_enc_workers has no effect since encodings are already provided', RuntimeWarning)
            
            result = self._find_duplicates_dict(
                encoding_map=encoding_map,
                max_distance_threshold=max_distance_threshold,
                scores=scores,
                outfile=outfile,
                search_method=search_method,
                num_dist_workers=num_dist_workers
            )
        else:
            raise ValueError('Provide either an image directory or encodings!')
        return result

def encode_images(self, image_dir=None, recursive: bool = False, num_enc_workers: int = cpu_count()):
        """
        Generate hashes for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            dictionary: A dictionary that contains a mapping of filenames and corresponding 64 character hash string
                        such as {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        mapping = myencoder.encode_images('path/to/directory')
        ```
        """
        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        files = generate_files(image_dir, recursive)

        logger.info(f'Start: Calculating hashes...')

        hashes = parallelise(function=self.encode_image, data=files, verbose=self.verbose, num_workers=num_enc_workers)
        hash_initial_dict = dict(zip(generate_relative_names(image_dir, files), hashes))
        hash_dict = {
            k: v for k, v in hash_initial_dict.items() if v
        }  # To ignore None (returned if some probelm with image file)

        logger.info(f'End: Calculating hashes!')
        return hash_dict

def encode_image(
        self, image_file=None, image_array: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate hash for a single image.

        Args:
            image_file: Path to the image file.
            image_array: Optional, used instead of image_file. Image typecast to numpy array.

        Returns:
            hash: A 16 character hexadecimal string hash for the image.

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        myhash = myencoder.encode_image(image_file='path/to/image.jpg')
        OR
        myhash = myencoder.encode_image(image_array=<numpy array of image>)
        ```
        """
        try:
            if image_file and os.path.exists(image_file):
                image_file = Path(image_file)
                image_pp = load_image(
                    image_file=image_file, target_size=self.target_size, grayscale=True
                )

            elif isinstance(image_array, np.ndarray):
                check_image_array_hash(image_array)  # Do sanity checks on array
                image_pp = preprocess_image(
                    image=image_array, target_size=self.target_size, grayscale=True
                )
            else:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError('Please provide either image file path or image array!')

        return self._hash_func(image_pp) if isinstance(image_pp, np.ndarray) else None

def hamming_distance(hash1: str, hash2: str) -> float:
        """
        Calculate the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
        to be 64 for each hash and then calculates the hamming distance.

        Args:
            hash1: hash string
            hash2: hash string

        Returns:
            hamming_distance: Hamming distance between the two hashes.
        """
        hash1_bin = bin(int(hash1, 16))[2:].zfill(
            64
        )  # zfill ensures that len of hash is 64 and pads MSB if it is < A
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])

def find_duplicates_to_remove(
        self,
        image_dir: PurePath = None,
        encoding_map: Dict[str, str] = None,
        max_distance_threshold: int = 10,
        outfile: Optional[str] = None,
        recursive: Optional[bool] = False,
        num_enc_workers: int = cpu_count(),
        num_dist_workers: int = cpu_count()
    ) -> List:
        """
        Give out a list of image file names to remove based on the hamming distance threshold threshold. Does not
        remove the mentioned files.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            encoding_map: Optional, used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding hashes.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            duplicates: List of image file names that are found to be duplicate of me other file in the directory.

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory'),
        max_distance_threshold=15)

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, outfile='results.json')
        ```
        """
        result = self.find_duplicates(
            image_dir=image_dir,
            encoding_map=encoding_map,
            max_distance_threshold=max_distance_threshold,
            scores=False,
            recursive=recursive,
            num_enc_workers=num_enc_workers,
            num_dist_workers=num_dist_workers
        )
        files_to_remove = get_files_to_remove(result)
        if outfile:
            save_json(files_to_remove, outfile)
        return files_to_remove

def encode_images(self, image_dir=None, recursive: bool = False, num_enc_workers: int = cpu_count()):
        """
        Generate hashes for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            dictionary: A dictionary that contains a mapping of filenames and corresponding 64 character hash string
                        such as {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        mapping = myencoder.encode_images('path/to/directory')
        ```
        """
        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        files = generate_files(image_dir, recursive)

        logger.info(f'Start: Calculating hashes...')

        hashes = parallelise(function=self.encode_image, data=files, verbose=self.verbose, num_workers=num_enc_workers)
        hash_initial_dict = dict(zip(generate_relative_names(image_dir, files), hashes))
        hash_dict = {
            k: v for k, v in hash_initial_dict.items() if v
        }  # To ignore None (returned if some probelm with image file)

        logger.info(f'End: Calculating hashes!')
        return hash_dict

def _find_duplicates_dict(
        self,
        encoding_map: Dict[str, str],
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
        num_dist_workers: int = cpu_count()
    ) -> Dict:
        """
        Take in dictionary {filename: encoded image}, detects duplicates below the given hamming distance threshold
        and returns a dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
        the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            encoding_map: Dictionary with keys as file names and values as encoded images (hashes).
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether hamming distance scores are to be returned along with retrieved
            duplicates.
            outfile: Optional, name of the file to save the results. Default is None.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        logger.info('Start: Evaluating hamming distances for getting duplicates')

        result_set = HashEval(
            test=encoding_map,
            queries=encoding_map,
            distance_function=self.hamming_distance,
            verbose=self.verbose,
            threshold=max_distance_threshold,
            search_method=search_method,
            num_dist_workers=num_dist_workers
        )

        logger.info('End: Evaluating hamming distances for getting duplicates')

        self.results = result_set.retrieve_results(scores=scores)
        if outfile:
            save_json(self.results, outfile)
        return self.results

def find_duplicates(
        self,
        image_dir: PurePath = None,
        encoding_map: Dict[str, str] = None,
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
        recursive: Optional[bool] = False,
        num_enc_workers: int = cpu_count(),
        num_dist_workers: int = cpu_count()
    ) -> Dict:
        """
        Find duplicates for each file. Takes in path of the directory or encoding dictionary in which duplicates are to
        be detected. All images with hamming distance less than or equal to the max_distance_threshold are regarded as
        duplicates. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the below the given hamming distance could be returned instead of just duplicate filenames for each
        query file.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            encoding_map: Optional,  used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding hashes.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            scores: Optional, boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            duplicates dictionary: if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
                        score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}. if scores is False, then a
                        dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'],
                        'image2.jpg':['image1_duplicate1.jpg',..], ..}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True,
        outfile='results.json')

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, scores=True, outfile='results.json')
        ```
        """
        self._check_hamming_distance_bounds(thresh=max_distance_threshold)
        if image_dir:
            result = self._find_duplicates_dir(
                image_dir=image_dir,
                max_distance_threshold=max_distance_threshold,
                scores=scores,
                outfile=outfile,
                search_method=search_method,
                recursive=recursive,
                num_enc_workers=num_enc_workers,
                num_dist_workers=num_dist_workers
            )
        elif encoding_map:
            if recursive:
                warnings.warn('recursive parameter is irrelevant when using encodings.', SyntaxWarning)
            
            warnings.warn('Parameter num_enc_workers has no effect since encodings are already provided', RuntimeWarning)
            
            result = self._find_duplicates_dict(
                encoding_map=encoding_map,
                max_distance_threshold=max_distance_threshold,
                scores=scores,
                outfile=outfile,
                search_method=search_method,
                num_dist_workers=num_dist_workers
            )
        else:
            raise ValueError('Provide either an image directory or encodings!')
        return result

def find_duplicates_to_remove(
        self,
        image_dir: PurePath = None,
        encoding_map: Dict[str, str] = None,
        max_distance_threshold: int = 10,
        outfile: Optional[str] = None,
        recursive: Optional[bool] = False,
        num_enc_workers: int = cpu_count(),
        num_dist_workers: int = cpu_count()
    ) -> List:
        """
        Give out a list of image file names to remove based on the hamming distance threshold threshold. Does not
        remove the mentioned files.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            encoding_map: Optional, used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding hashes.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            duplicates: List of image file names that are found to be duplicate of me other file in the directory.

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory'),
        max_distance_threshold=15)

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, outfile='results.json')
        ```
        """
        result = self.find_duplicates(
            image_dir=image_dir,
            encoding_map=encoding_map,
            max_distance_threshold=max_distance_threshold,
            scores=False,
            recursive=recursive,
            num_enc_workers=num_enc_workers,
            num_dist_workers=num_dist_workers
        )
        files_to_remove = get_files_to_remove(result)
        if outfile:
            save_json(files_to_remove, outfile)
        return files_to_remove

def _find_duplicates_dict(
        self,
        encoding_map: Dict[str, str],
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
        num_dist_workers: int = cpu_count()
    ) -> Dict:
        """
        Take in dictionary {filename: encoded image}, detects duplicates below the given hamming distance threshold
        and returns a dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
        the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            encoding_map: Dictionary with keys as file names and values as encoded images (hashes).
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether hamming distance scores are to be returned along with retrieved
            duplicates.
            outfile: Optional, name of the file to save the results. Default is None.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        logger.info('Start: Evaluating hamming distances for getting duplicates')

        result_set = HashEval(
            test=encoding_map,
            queries=encoding_map,
            distance_function=self.hamming_distance,
            verbose=self.verbose,
            threshold=max_distance_threshold,
            search_method=search_method,
            num_dist_workers=num_dist_workers
        )

        logger.info('End: Evaluating hamming distances for getting duplicates')

        self.results = result_set.retrieve_results(scores=scores)
        if outfile:
            save_json(self.results, outfile)
        return self.results

def _find_duplicates_dir(
        self,
        image_dir: PurePath,
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
        recursive: Optional[bool] = False,
        num_enc_workers: int = cpu_count(),
        num_dist_workers: int = cpu_count()
    ) -> Dict:
        """
        Take in path of the directory in which duplicates are to be detected below the given hamming distance
        threshold. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            image_dir: Path to the directory containing all the images.
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            outfile: Name of the file the results should be written to.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        encoding_map = self.encode_images(image_dir, recursive=recursive, num_enc_workers=num_enc_workers)
        results = self._find_duplicates_dict(
            encoding_map=encoding_map,
            max_distance_threshold=max_distance_threshold,
            scores=scores,
            outfile=outfile,
            search_method=search_method,
            num_dist_workers=num_dist_workers
        )
        return results

