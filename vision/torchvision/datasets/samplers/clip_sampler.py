class UniformClipSampler(Sampler):
    """
    Sample `num_video_clips_per_video` clips for each video, equally spaced.
    When number of unique clips in the video is fewer than num_video_clips_per_video,
    repeat the clips until `num_video_clips_per_video` clips are collected

    Args:
        video_clips (VideoClips): video clips to sample from
        num_clips_per_video (int): number of clips to be sampled per video
    """

    def __init__(self, video_clips: VideoClips, num_clips_per_video: int) -> None:
        if not isinstance(video_clips, VideoClips):
            raise TypeError(f"Expected video_clips to be an instance of VideoClips, got {type(video_clips)}")
        self.video_clips = video_clips
        self.num_clips_per_video = num_clips_per_video

    def __iter__(self) -> Iterator[int]:
        idxs = []
        s = 0
        # select num_clips_per_video for each video, uniformly spaced
        for c in self.video_clips.clips:
            length = len(c)
            if length == 0:
                # corner case where video decoding fails
                continue

            sampled = torch.linspace(s, s + length - 1, steps=self.num_clips_per_video).floor().to(torch.int64)
            s += length
            idxs.append(sampled)
        return iter(cast(List[int], torch.cat(idxs).tolist()))

    def __len__(self) -> int:
        return sum(self.num_clips_per_video for c in self.video_clips.clips if len(c) > 0)

class RandomClipSampler(Sampler):
    """
    Samples at most `max_video_clips_per_video` clips for each video randomly

    Args:
        video_clips (VideoClips): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    """

    def __init__(self, video_clips: VideoClips, max_clips_per_video: int) -> None:
        if not isinstance(video_clips, VideoClips):
            raise TypeError(f"Expected video_clips to be an instance of VideoClips, got {type(video_clips)}")
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self) -> Iterator[int]:
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            size = min(length, self.max_clips_per_video)
            sampled = torch.randperm(length)[:size] + s
            s += length
            idxs.append(sampled)
        idxs_ = torch.cat(idxs)
        # shuffle all clips randomly
        perm = torch.randperm(len(idxs_))
        return iter(idxs_[perm].tolist())

    def __len__(self) -> int:
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.clips)

