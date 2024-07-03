    def get(self,
            item: str,
            ids: Optional[list] = None,
            num_samples: Optional[int] = None,
            behavior: Optional[str] = None) -> torch.Tensor:
        """Get the buffer of a specific item.

        Args:
            item (str): The demanded item.
            ids (list[int], optional): The demanded ids. Defaults to None.
            num_samples (int, optional): Number of samples to calculate the
                results. Defaults to None.
            behavior (str, optional): Behavior to calculate the results.
                Options are `mean` | None. Defaults to None.

        Returns:
            Tensor: The results of the demanded item.
        """
        if ids is None:
            ids = self.ids

        outs = []
        for id in ids:
            out = self.tracks[id][item]
            if isinstance(out, list):
                if num_samples is not None:
                    out = out[-num_samples:]
                    out = torch.cat(out, dim=0)
                    if behavior == 'mean':
                        out = out.mean(dim=0, keepdim=True)
                    elif behavior is None:
                        out = out[None]
                    else:
                        raise NotImplementedError()
                else:
                    out = out[-1]
            outs.append(out)
        return torch.cat(outs, dim=0)