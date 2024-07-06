def broadcast(
    data_or_fn: Union[T, Callable[[], T]],
    *,
    success: bool = True,
    stage_name: Optional[str] = None,
    rank: int = 0,
    pg: Optional[dist.ProcessGroup] = None,
) -> T:
    """
    Broadcasts the data payload from rank 0 to all other ranks.
    Or if a function is passed, execute it in rank 0 and broadcast result to all other ranks.

    Can be used to broadcast a failure signal to stop all ranks.

    If the function raises an exception, all ranks will raise.

    Args:
        data_or_fn: the data to broadcast or function to execute and broadcast result.
        success: False to stop all ranks.
        stage_name: the name of the logical stage for synchronization and debugging
        rank: rank to broadcast data or execute function and broadcast resutls.
        pg: the process group for sync
    Throws:
        RuntimeError from original exception trace
    Returns:
        the value after synchronization

    Example usage:
    >> id = broadcast(data_or_fn=allocate_id, rank=0, pg=ext_pg.my_pg)
    """

    if not success and data_or_fn is not None:
        raise AssertionError(
            "Data or Function is expected to be None if not successful"
        )

    payload: Optional[T] = None
    exception: Optional[Exception] = None
    # if no pg is passed then execute if rank is 0
    if (pg is None and rank == 0) or (pg is not None and pg.rank() == rank):
        # determine if it is an executable function or data payload only
        if callable(data_or_fn):
            try:
                payload = data_or_fn()
            except Exception as e:
                success = False
                exception = e
        else:
            payload = data_or_fn

    # broadcast the exception type if any to all ranks for failure categorization
    sync_obj = SyncPayload(
        stage_name=stage_name,
        success=success,
        payload=payload,
        exception=exception,
    )

    if pg is not None:
        broadcast_list = [sync_obj]
        dist.broadcast_object_list(broadcast_list, src=rank, group=pg)
        assert len(broadcast_list) == 1
        sync_obj = broadcast_list[0]

    # failure in any rank will trigger a throw in every rank.
    if not sync_obj.success:
        error_msg = f"Rank {rank} failed"
        if stage_name is not None:
            error_msg += f": stage {sync_obj.stage_name}"
        if sync_obj.exception is not None:
            error_msg += f": exception {sync_obj.exception}"
        raise RuntimeError(error_msg) from sync_obj.exception

    return cast(T, sync_obj.payload)

def all_gather(
    data_or_fn: Union[T, Callable[[], T]],
    stage_name: Optional[str] = None,
    pg: Optional[dist.ProcessGroup] = None,
) -> List[T]:
    """
    A simple all_gather primitive with basic synchronization guard logic,
    by checking payload from all ranks has the same stage name.

    Args:
        data_or_fn: the data to be all gathered across ranks or function to be executed
        stage_name: the sync stage name for out-of-sync protection
        pg: the process group for sync
    Throws:
        RuntimeError from original exception trace
    Returns:
        a list of synced data from all ranks

    Example usage:
    >> all_ids = all_gather(data_or_fn=allocate_id, pg=ext_pg.my_pg)
    """
    payload: Optional[T] = None
    exception: Optional[Exception] = None
    success = True
    # determine if it is an executable function or data payload only
    if callable(data_or_fn):
        try:
            payload = data_or_fn()
        except Exception as e:
            success = False
            exception = e
    else:
        payload = data_or_fn

    sync_obj = SyncPayload(
        stage_name=stage_name,
        success=success,
        payload=payload,
        exception=exception,
    )

    if pg is not None:
        # List of success/failure across all ranks.
        total_list = [None] * dist.get_world_size(pg)
        all_gather_object_enforce_type(pg, total_list, sync_obj)
        # Each rank will throw RuntimeError in case of failure on any rank.
        stage_name = cast(SyncPayload[T], total_list[0]).stage_name
        exception_list: List[Tuple[int, Exception]] = []
        ret_list: List[T] = []
        error_msg: str = ""

        for i, sp in enumerate(cast(List[SyncPayload[T]], total_list)):
            if sp.stage_name != stage_name:
                error_msg += (
                    f"Unexpected stage name received from rank {i}: {sp.stage_name} "
                )
                continue
            if not sp.success and sp.exception is not None:
                exception_list.append((i, sp.exception))
                continue
            ret_list.append(sp.payload)

        if len(exception_list) > 0:
            raise RuntimeError(  # type: ignore[misc]
                error_msg, exception_list
            ) from exception_list[0]
        return ret_list
    else:
        if not sync_obj.success:
            raise RuntimeError(
                f"all_gather failed with exception {sync_obj.exception}",
            ) from sync_obj.exception
        return [sync_obj.payload]  # type: ignore[list-item]

