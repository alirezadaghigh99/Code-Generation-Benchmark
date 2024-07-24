class FlowHandler:
    def __init__(self, db, symbol_handler, flow):
        self.db = db
        self.publisher = Publisher(self.db)
        self.flow = flow
        self.symbol = symbol_handler
        self.running_non_stop: bool = self.is_running_non_stop()

    def is_running_non_stop(self) -> bool:
        """
        Slips runs non-stop in case of an interface or a growing zeek dir,
        it only stops on ctrl+c
        """
        if (
            self.db.get_input_type() == "interface"
            or self.db.is_growing_zeek_dir()
        ):
            return True

    def is_supported_flow(self):
        supported_types = (
            "ssh",
            "ssl",
            "http",
            "dns",
            "conn",
            "flow",
            "argus",
            "nfdump",
            "notice",
            "dhcp",
            "files",
            "arp",
            "ftp",
            "smtp",
            "software",
            "weird",
            "tunnel",
        )
        return bool(
            self.flow.starttime is not None
            and self.flow.type_ in supported_types
        )

    def make_sure_theres_a_uid(self):
        """
        Generates a uid and adds it to the flow if none is found
        """
        # dhcp flows have uids field instead of uid
        if (type(self.flow) == DHCP and not self.flow.uids) or (
            type(self.flow) != DHCP and not self.flow.uid
        ):
            # In the case of other tools that are not Zeek, there is no UID. So we generate a new one here
            # Zeeks uses human-readable strings in Base62 format, from 112 bits usually.
            # We do base64 with some bits just because we need a fast unique way
            self.flow.uid = base64.b64encode(
                binascii.b2a_hex(os.urandom(9))
            ).decode("utf-8")

    def handle_conn(self):
        role = "Client"
        daddr_as_obj = ipaddress.ip_address(self.flow.daddr)
        # this identified the tuple, it's a combination
        # of daddr, dport and proto
        # this is legacy code and refactoring it will
        # break many things, so i wont:D
        tupleid = f"{daddr_as_obj}-{self.flow.dport}-{self.flow.proto}"

        # Compute the symbol for this flow, for this TW, for this profile.
        # The symbol is based on the 'letters' of the original Startosphere IPS tool
        symbol: Tuple = self.symbol.compute(self.flow, self.twid, "OutTuples")

        # Change symbol for its internal data. Symbol is a tuple and is
        # confusing if we ever change the API
        # Add the out tuple
        self.db.add_tuple(
            self.profileid, self.twid, tupleid, symbol, role, self.flow
        )

        # Add the dstip
        self.db.add_ips(self.profileid, self.twid, self.flow, role)
        # Add the dstport
        port_type = "Dst"
        self.db.add_port(self.profileid, self.twid, self.flow, role, port_type)

        # Add the srcport
        port_type = "Src"
        self.db.add_port(self.profileid, self.twid, self.flow, role, port_type)
        # store the original flow as benign in sqlite
        self.db.add_flow(self.flow, self.profileid, self.twid, "benign")

        self.db.add_mac_addr_to_profile(self.profileid, self.flow.smac)

        if self.running_non_stop:
            # to avoid publishing duplicate MACs, when running on
            # an interface, we should have an arp.log, so we'll publish
            # MACs from there only
            return

        self.publisher.new_MAC(self.flow.smac, self.flow.saddr)
        self.publisher.new_MAC(self.flow.dmac, self.flow.daddr)

    def handle_dns(self):
        self.db.add_out_dns(self.profileid, self.twid, self.flow)
        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_http(self):
        self.db.add_out_http(
            self.profileid,
            self.twid,
            self.flow,
        )

        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_ssl(self):
        self.db.add_out_ssl(self.profileid, self.twid, self.flow)
        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_ssh(self):
        self.db.add_out_ssh(self.profileid, self.twid, self.flow)
        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_notice(self):
        self.db.add_out_notice(self.profileid, self.twid, self.flow)

        if "Gateway_addr_identified" in self.flow.note:
            # get the gw addr form the msg
            gw_addr = self.flow.msg.split(": ")[-1].strip()
            self.db.set_default_gateway("IP", gw_addr)

        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_ftp(self):
        if used_port := self.flow.used_port:
            self.db.set_ftp_port(used_port)

        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_smtp(self):
        to_send = {
            "flow": asdict(self.flow),
            "profileid": self.profileid,
            "twid": self.twid,
        }
        to_send = json.dumps(to_send)
        self.db.publish("new_smtp", to_send)

        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_software(self):
        self.db.add_software_to_profile(self.profileid, self.flow)
        epoch_time = utils.convert_format(self.flow.starttime, "unixtimestamp")
        self.flow.starttime = epoch_time
        self.publisher.new_software(self.profileid, self.flow)

        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_dhcp(self):
        # send this to ip_info module to get vendor info about this MAC
        self.publisher.new_MAC(
            self.flow.smac or False,
            self.flow.saddr,
        )

        self.db.add_mac_addr_to_profile(self.profileid, self.flow.smac)

        if self.flow.server_addr:
            self.db.store_dhcp_server(self.flow.server_addr)
            self.db.mark_profile_as_dhcp(self.profileid)

        epoch_time = utils.convert_format(self.flow.starttime, "unixtimestamp")
        self.flow.starttime = epoch_time

        self.publisher.new_dhcp(self.profileid, self.flow)
        for uid in self.flow.uids:
            # we're modifying the copy of self.flow
            # the goal is to store a copy of this flow for each uid in self.flow.uids
            flow = self.flow
            flow.uid = uid
            self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_files(self):
        """
        Send files.log data to new_downloaded_file channel in the TI module to see if it's malicious
        """

        # files slips sees can be of 2 types: suricata or zeek
        to_send = {
            "flow": asdict(self.flow),
            "type": "suricata" if type(self.flow) == SuricataFile else "zeek",
            "profileid": self.profileid,
            "twid": self.twid,
        }

        to_send = json.dumps(to_send)
        self.db.publish("new_downloaded_file", to_send)
        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_arp(self):
        to_send = {
            "flow": asdict(self.flow),
            "profileid": self.profileid,
            "twid": self.twid,
        }
        # send to arp module
        to_send = json.dumps(to_send)
        self.db.publish("new_arp", to_send)
        self.db.add_mac_addr_to_profile(self.profileid, self.flow.smac)
        self.publisher.new_MAC(self.flow.dmac, self.flow.daddr)
        self.publisher.new_MAC(self.flow.smac, self.flow.saddr)
        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_weird(self):
        """
        handles weird.log zeek flows
        """
        to_send = {
            "profileid": self.profileid,
            "twid": self.twid,
            "flow": asdict(self.flow),
        }
        to_send = json.dumps(to_send)
        self.db.publish("new_weird", to_send)
        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

    def handle_tunnel(self):
        to_send = {
            "profileid": self.profileid,
            "twid": self.twid,
            "flow": asdict(self.flow),
        }
        to_send = json.dumps(to_send)
        self.db.publish("new_tunnel", to_send)

        self.db.add_altflow(self.flow, self.profileid, self.twid, "benign")

