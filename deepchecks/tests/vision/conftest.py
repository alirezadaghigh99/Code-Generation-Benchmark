def run_update_loop(dataset: VisionData):
    context: Context = Context(dataset, random_state=0)
    dataset.init_cache()
    for batch in context.train:
        batch = BatchWrapper(batch, dataset.task_type, dataset.number_of_images_cached)
        dataset.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)

