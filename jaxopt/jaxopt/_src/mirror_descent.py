  def run(self,
          init_params: Any,
          hyperparams_proj: Optional[Any] = None,
          *args,
          **kwargs) -> base.OptStep:
    return super().run(init_params, hyperparams_proj, *args, **kwargs)