class Conn:
    starttime: str
    uid: str
    saddr: str
    daddr: str

    dur: float

    proto: str
    appproto: str

    sport: str
    dport: str

    spkts: int
    dpkts: int

    sbytes: int
    dbytes: int

    smac: str
    dmac: str

    state: str
    history: str
    type_: str = "conn"
    dir_: str = "->"

    def __post_init__(self) -> None:
        endtime = str(self.starttime) + str(timedelta(seconds=self.dur))
        self.endtime: str = endtime
        self.pkts: int = self.spkts + self.dpkts
        self.bytes: int = self.sbytes + self.dbytes
        self.state_hist: str = self.history or self.state
        # AIDs are for conn.log flows only
        self.aid = utils.get_aid(self)

