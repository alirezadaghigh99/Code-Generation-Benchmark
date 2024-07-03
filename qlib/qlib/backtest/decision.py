class Order:
    """
    stock_id : str
    amount : float
    start_time : pd.Timestamp
        closed start time for order trading
    end_time : pd.Timestamp
        closed end time for order trading
    direction : int
        Order.SELL for sell; Order.BUY for buy
    factor : float
            presents the weight factor assigned in Exchange()
    """

    # 1) time invariant values
    # - they are set by users and is time-invariant.
    stock_id: str
    amount: float  # `amount` is a non-negative and adjusted value
    direction: OrderDir

    # 2) time variant values:
    # - Users may want to set these values when using lower level APIs
    # - If users don't, TradeDecisionWO will help users to set them
    # The interval of the order which belongs to (NOTE: this is not the expected order dealing range time)
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    # 3) results
    # - users should not care about these values
    # - they are set by the backtest system after finishing the results.
    # What the value should be about in all kinds of cases
    # - not tradable: the deal_amount == 0 , factor is None
    #    - the stock is suspended and the entire order fails. No cost for this order
    # - dealt or partially dealt: deal_amount >= 0 and factor is not None
    deal_amount: float = 0.0  # `deal_amount` is a non-negative value
    factor: Optional[float] = None

    # TODO:
    # a status field to indicate the dealing result of the order

    # FIXME:
    # for compatible now.
    # Please remove them in the future
    SELL: ClassVar[OrderDir] = OrderDir.SELL
    BUY: ClassVar[OrderDir] = OrderDir.BUY

    def __post_init__(self) -> None:
        if self.direction not in {Order.SELL, Order.BUY}:
            raise NotImplementedError("direction not supported, `Order.SELL` for sell, `Order.BUY` for buy")
        self.deal_amount = 0.0
        self.factor = None

    @property
    def amount_delta(self) -> float:
        """
        return the delta of amount.
        - Positive value indicates buying `amount` of share
        - Negative value indicates selling `amount` of share
        """
        return self.amount * self.sign

    @property
    def deal_amount_delta(self) -> float:
        """
        return the delta of deal_amount.
        - Positive value indicates buying `deal_amount` of share
        - Negative value indicates selling `deal_amount` of share
        """
        return self.deal_amount * self.sign

    @property
    def sign(self) -> int:
        """
        return the sign of trading
        - `+1` indicates buying
        - `-1` value indicates selling
        """
        return self.direction * 2 - 1

    @staticmethod
    def parse_dir(direction: Union[str, int, np.integer, OrderDir, np.ndarray]) -> Union[OrderDir, np.ndarray]:
        if isinstance(direction, OrderDir):
            return direction
        elif isinstance(direction, (int, float, np.integer, np.floating)):
            return Order.BUY if direction > 0 else Order.SELL
        elif isinstance(direction, str):
            dl = direction.lower().strip()
            if dl == "sell":
                return OrderDir.SELL
            elif dl == "buy":
                return OrderDir.BUY
            else:
                raise NotImplementedError(f"This type of input is not supported")
        elif isinstance(direction, np.ndarray):
            direction_array = direction.copy()
            direction_array[direction_array > 0] = Order.BUY
            direction_array[direction_array <= 0] = Order.SELL
            return direction_array
        else:
            raise NotImplementedError(f"This type of input is not supported")

    @property
    def key_by_day(self) -> tuple:
        """A hashable & unique key to identify this order, under the granularity in day."""
        return self.stock_id, self.date, self.direction

    @property
    def key(self) -> tuple:
        """A hashable & unique key to identify this order."""
        return self.stock_id, self.start_time, self.end_time, self.direction

    @property
    def date(self) -> pd.Timestamp:
        """Date of the order."""
        return pd.Timestamp(self.start_time.replace(hour=0, minute=0, second=0))