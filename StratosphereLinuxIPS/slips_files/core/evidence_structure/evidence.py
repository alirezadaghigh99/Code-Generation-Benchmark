class Attacker:
    direction: Direction
    attacker_type: IoCType
    value: str  # like the actual ip/domain/url check if value is reserved
    profile: ProfileID = ""

    def __post_init__(self):
        if self.attacker_type == IoCType.IP:
            validate_ip(self.value)
        # each attacker should have a profile if it's an IP
        if self.attacker_type == IoCType.IP:
            self.profile = ProfileID(ip=self.value)

class Victim:
    direction: Direction
    victim_type: IoCType
    value: str  # like the actual ip/domain/url check if value is reserved

    def __post_init__(self):
        if self.victim_type == IoCType.IP:
            validate_ip(self.value)

