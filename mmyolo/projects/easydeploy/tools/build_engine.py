    def build(self,
              scale: Optional[List[List]] = None,
              fp16: bool = True,
              with_profiling=True):
        self.__build_engine(scale, fp16, with_profiling)