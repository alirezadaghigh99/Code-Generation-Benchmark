def fetch_data(name, path=None, overwrite=False):
    """
    Download example datasets from OpenNeuro and other sources.
    Datasets are pulled using `datalad` so the raw data will 
    not actually be dowloaded until it is needed. This makes it
    really fast.
    
    Arguments
    ---------
    name : string
        the dataset to download
        Options:
            - ds004711 [OpenNeuroDatasets/ds004711]
            - example/t1-age
            - example/t1-t1_mask
            
    Example
    -------
    import nitrain as nt
    ds = nt.fetch_data('openneuro/ds004711')
    """
    
    if path is None:
        path = get_nitrain_dir()
    else:
        path = os.path.expanduser(path)
    
    if name.startswith('openneuro'):
        import datalad.api as dl
        
        save_dir = os.path.join(path, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        # load openneuro dataset using datalad
        res = dl.clone(source=f'///{name}', path=save_dir)
    elif name.startswith('example'):
        if name == 'example-01':
            # folder with nifti images and csv file
            # this example is good for testing and sanity checks
            # set up directory
            save_dir = os.path.join(path, name)
            
            if overwrite:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            else:
                if os.path.exists(save_dir):
                    return save_dir
            
            os.makedirs(save_dir, exist_ok=True)

            img2d = ants.from_numpy(np.ones((30,40)))
            img3d = ants.from_numpy(np.ones((30,40,50)))
            img3d_seg = ants.from_numpy(np.zeros(img3d.shape).astype('uint8'))
            img3d_seg[10:20,10:30,10:40] = 1
            
            img3d_multiseg = ants.from_numpy(np.zeros(img3d.shape).astype('uint8'))
            img3d_multiseg[:20,:20,:20] = 1
            img3d_multiseg[20:30,20:30,20:30] = 2
            img3d_multiseg[30:,30:,30:]=0
            
            img3d_large = ants.from_numpy(np.ones((60,80,100)))
            for i in range(10):
                sub_dir = os.path.join(save_dir, f'sub_{i}')
                os.mkdir(sub_dir)
                ants.image_write(img2d + i, os.path.join(sub_dir, 'img2d.nii.gz'))
                ants.image_write(img3d + i, os.path.join(sub_dir, 'img3d.nii.gz'))
                ants.image_write(img3d_large + i, os.path.join(sub_dir, 'img3d_large.nii.gz'))
                ants.image_write(img3d + i + 100, os.path.join(sub_dir, 'img3d_100.nii.gz'))
                ants.image_write(img3d_seg, os.path.join(sub_dir, 'img3d_seg.nii.gz'))
                ants.image_write(img3d_multiseg, os.path.join(sub_dir, 'img3d_multiseg.nii.gz'))
                ants.image_write(img3d + i + 1000, os.path.join(sub_dir, 'img3d_1000.nii.gz'))
            
            # write csv file
            ids = [f'sub_{i}' for i in range(10)]
            age = [i + 50 for i in range(10)]
            weight = [i + 200 for i in range(10)]
            img2d = [f'sub_{i}/img2d.nii.gz' for i in range(10)]
            img3d = [f'sub_{i}/img3d.nii.gz' for i in range(10)]
            img3d_large = [f'sub_{i}/img3d_large.nii.gz' for i in range(10)]
            img3d_100 = [f'sub_{i}/img3d_100.nii.gz' for i in range(10)]
            img3d_1000 = [f'sub_{i}/img3d_1000.nii.gz' for i in range(10)]
            img3d_seg = [f'sub_{i}/img3d_seg.nii.gz' for i in range(10)]
            df = pd.DataFrame({'sub_id': ids, 'age': age, 'weight': weight,
                               'img2d': img2d, 'img3d': img3d, 'img3d_large': img3d_large,
                               'img3d_100':img3d_100, 'img3d_1000': img3d_1000,
                               'img3d_seg': img3d_seg})
            df.to_csv(os.path.join(save_dir, 'participants.csv'), index=False)
            
    else:
        raise ValueError('Dataset name not recognized.')

    return save_dir

