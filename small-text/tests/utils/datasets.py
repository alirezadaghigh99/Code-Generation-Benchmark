def random_sklearn_dataset(num_samples, vocab_size=60, num_classes=2, multi_label=False):

    x = sparse.random(num_samples, vocab_size, density=0.15, format='csr')

    if multi_label:
        y = sparse.random(num_samples, num_classes, density=0.5, format='csr')
        y.data[np.s_[:]] = 1
        y = y.astype(int)
    else:
        y = np.random.randint(0, high=num_classes, size=x.shape[0])

    return SklearnDataset(x, y)

def random_text_classification_dataset(num_samples=10, max_length=60, num_classes=2,
                                       multi_label=False, vocab_size=10,
                                       device='cpu', target_labels='inferred', dtype=torch.long):

    if target_labels not in ['explicit', 'inferred']:
        raise ValueError(f'Invalid test parameter value for target_labels: {str(target_labels)}')
    if num_classes > num_samples:
        raise ValueError('When num_classes is greater than num_samples the occurrence of each '
                         'class cannot be guaranteed')

    vocab = Vocab(Counter([f'word_{i}' for i in range(vocab_size)]))

    if multi_label:
        data = [(
                    torch.randint(vocab_size, (max_length,), dtype=dtype, device=device),
                    np.sort(random_labeling(num_classes, multi_label)).tolist()
                 )
                for _ in range(num_samples)]
    else:
        data = [
            (torch.randint(10, (max_length,), dtype=dtype, device=device),
             random_labeling(num_classes, multi_label))
            for _ in range(num_samples)]

    data = assure_all_labels_occur(data, num_classes, multi_label=multi_label)

    target_labels = None if target_labels == 'inferred' else np.arange(num_classes)
    return PytorchTextClassificationDataset(data, vocab, multi_label=multi_label,
                                            target_labels=target_labels)

def random_matrix_data(matrix_type, label_type, num_samples=100, num_dimension=40, num_labels=2):
    if matrix_type == 'dense':
        x = np.random.rand(num_samples, num_dimension)
    elif matrix_type == 'sparse':
        x = sparse.random(num_samples, num_dimension, density=0.15, format='csr')
    else:
        raise ValueError(f'Invalid matrix_type: {matrix_type}')

    if label_type == 'dense':
        y = np.random.randint(0, high=num_labels, size=x.shape[0])
    elif label_type == 'sparse':
        y = sparse.random(num_samples, num_labels, density=0.5, format='csr')
        y.data[np.s_[:]] = 1
        y = y.astype(int)
    else:
        raise ValueError(f'Invalid label_type: {label_type}')

    return x, y

