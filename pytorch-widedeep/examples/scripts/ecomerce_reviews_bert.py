    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)