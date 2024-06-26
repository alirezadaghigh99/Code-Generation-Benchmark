class BaseFrameSample(BaseTransform):
    """Directly get the key frame, no reference frames.

    Args:
        collect_video_keys (list[str]): The keys of video info to be
            collected.
    """

    def __init__(self,
                 collect_video_keys: List[str] = ['video_id', 'video_length']):
        self.collect_video_keys = collect_video_keys

    def prepare_data(self, video_infos: dict,
                     sampled_inds: List[int]) -> Dict[str, List]:
        """Prepare data for the subsequent pipeline.

        Args:
            video_infos (dict): The whole video information.
            sampled_inds (list[int]): The sampled frame indices.

        Returns:
            dict: The processed data information.
        """
        frames_anns = video_infos['images']
        final_data_info = defaultdict(list)
        # for data in frames_anns:
        for index in sampled_inds:
            data = frames_anns[index]
            # copy the info in video-level into img-level
            for key in self.collect_video_keys:
                if key == 'video_length':
                    data['ori_video_length'] = video_infos[key]
                    data['video_length'] = len(sampled_inds)
                else:
                    data[key] = video_infos[key]
            # Collate data_list (list of dict to dict of list)
            for key, value in data.items():
                final_data_info[key].append(value)

        return final_data_info

    def transform(self, video_infos: dict) -> Optional[Dict[str, List]]:
        """Transform the video information.

        Args:
            video_infos (dict): The whole video information.

        Returns:
            dict: The data information of the key frames.
        """
        if 'key_frame_id' in video_infos:
            key_frame_id = video_infos['key_frame_id']
            assert isinstance(video_infos['key_frame_id'], int)
        else:
            key_frame_id = random.sample(
                list(range(video_infos['video_length'])), 1)[0]
        results = self.prepare_data(video_infos, [key_frame_id])

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(collect_video_keys={self.collect_video_keys})'
        return repr_str