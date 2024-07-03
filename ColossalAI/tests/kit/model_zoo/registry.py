    def get_sub_registry(
        self, keyword: Union[str, List[str]], exclude: Union[str, List[str]] = None, allow_empty: bool = False
    ):
        """
        Get a sub registry with models that contain the keyword.

        Args:
            keyword (str): Keyword to filter models.
        """
        new_dict = dict()

        if isinstance(keyword, str):
            keyword_list = [keyword]
        else:
            keyword_list = keyword
        assert isinstance(keyword_list, (list, tuple))

        if exclude is None:
            exclude_keywords = []
        elif isinstance(exclude, str):
            exclude_keywords = [exclude]
        else:
            exclude_keywords = exclude
        assert isinstance(exclude_keywords, (list, tuple))

        for k, v in self.items():
            for kw in keyword_list:
                if kw in k:
                    should_exclude = False
                    for ex_kw in exclude_keywords:
                        if ex_kw in k:
                            should_exclude = True

                    if not should_exclude:
                        new_dict[k] = v

        if not allow_empty:
            assert len(new_dict) > 0, f"No model found with keyword {keyword}"
        return new_dict