class ModuleFactory:
    def __init__(self):
        # same db as in conftest
        self.profiler_queue = Queue()
        self.input_queue = Queue()
        self.dummy_termination_event = Event()
        self.logger = Mock()  # Output()

    def get_default_db(self):
        """default is o port 6379, this is the one we're using in conftest"""
        return self.create_db_manager_obj(6379)

    def create_db_manager_obj(
        self, port, output_dir="output/", flush_db=False
    ):
        db = DBManager(self.logger, output_dir, port, flush_db=flush_db)
        db.r = db.rdb.r
        db.print = do_nothing
        assert db.get_used_redis_port() == port
        return db

    def create_main_obj(self, input_information):
        """returns an instance of Main() class in slips.py"""
        main = Main(testing=True)
        main.input_information = input_information
        main.input_type = "pcap"
        main.line_type = False
        return main

    def create_http_analyzer_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            http_analyzer = HTTPAnalyzer(
                self.logger,
                "dummy_output_dir",
                6379,
                self.dummy_termination_event,
            )
            http_analyzer.db.rdb = mock_db

        # override the self.print function to avoid broken pipes
        http_analyzer.print = do_nothing
        return http_analyzer

    def create_virustotal_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            virustotal = VT(
                self.logger,
                "dummy_output_dir",
                6379,
                self.dummy_termination_event,
            )
            virustotal.db.rdb = mock_db

        # override the self.print function to avoid broken pipes
        virustotal.print = do_nothing
        virustotal.__read_configuration = read_configuration
        virustotal.key_file = (
            "/media/alya/W/SLIPPS/modules/virustotal/api_key_secret"
        )
        return virustotal

    def create_arp_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            arp = ARP(
                self.logger,
                "dummy_output_dir",
                6379,
                self.dummy_termination_event,
            )
            arp.db.rdb = mock_db
        # override the self.print function to avoid broken pipes
        arp.print = do_nothing
        return arp

    def create_blocking_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            blocking = Blocking(
                self.logger,
                "dummy_output_dir",
                6379,
                self.dummy_termination_event,
            )
            blocking.db.rdb = mock_db

        # override the print function to avoid broken pipes
        blocking.print = do_nothing
        return blocking

    def create_flowalerts_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            flowalerts = FlowAlerts(
                self.logger,
                "dummy_output_dir",
                6379,
                self.dummy_termination_event,
            )
            flowalerts.db.rdb = mock_db

        # override the self.print function to avoid broken pipes
        flowalerts.print = do_nothing
        return flowalerts

    def create_dns_analyzer_obj(self, mock_db):
        flowalerts = self.create_flowalerts_obj(mock_db)
        return DNS(flowalerts.db, flowalerts=flowalerts)

    def create_notice_analyzer_obj(self, mock_db):
        flowalerts = self.create_flowalerts_obj(mock_db)
        return Notice(flowalerts.db, flowalerts=flowalerts)

    def create_smtp_analyzer_obj(self, mock_db):
        flowalerts = self.create_flowalerts_obj(mock_db)
        return SMTP(flowalerts.db, flowalerts=flowalerts)

    def create_ssl_analyzer_obj(self, mock_db):
        flowalerts = self.create_flowalerts_obj(mock_db)
        return SSL(flowalerts.db, flowalerts=flowalerts)

    def create_ssh_analyzer_obj(self, mock_db):
        flowalerts = self.create_flowalerts_obj(mock_db)
        return SSH(flowalerts.db, flowalerts=flowalerts)

    def create_downloaded_file_analyzer_obj(self, mock_db):
        flowalerts = self.create_flowalerts_obj(mock_db)
        return DownloadedFile(flowalerts.db, flowalerts=flowalerts)

    def create_tunnel_analyzer_obj(self, mock_db):
        flowalerts = self.create_flowalerts_obj(mock_db)
        return Tunnel(flowalerts.db, flowalerts=flowalerts)

    def create_conn_analyzer_obj(self, mock_db):
        flowalerts = self.create_flowalerts_obj(mock_db)
        return Conn(flowalerts.db, flowalerts=flowalerts)

    def create_software_analyzer_obj(self, mock_db):
        flowalerts = self.create_flowalerts_obj(mock_db)
        return Software(flowalerts.db, flowalerts=flowalerts)

    def create_input_obj(
        self, input_information, input_type, mock_db, line_type=False
    ):
        zeek_tmp_dir = os.path.join(os.getcwd(), "zeek_dir_for_testing")
        dummy_semaphore = Semaphore(0)
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            input = Input(
                Output(),
                "dummy_output_dir",
                6379,
                self.dummy_termination_event,
                is_input_done=dummy_semaphore,
                profiler_queue=self.profiler_queue,
                input_type=input_type,
                input_information=input_information,
                cli_packet_filter=None,
                zeek_or_bro=check_zeek_or_bro(),
                zeek_dir=zeek_tmp_dir,
                line_type=line_type,
                is_profiler_done_event=self.dummy_termination_event,
            )
        input.db.rdb = mock_db
        input.is_done_processing = do_nothing
        input.bro_timeout = 1
        # override the print function to avoid broken pipes
        input.print = do_nothing
        input.stop_queues = do_nothing
        input.testing = True

        return input

    def create_ip_info_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            ip_info = IPInfo(
                self.logger,
                "dummy_output_dir",
                6379,
                self.dummy_termination_event,
            )
            ip_info.db.rdb = mock_db
        # override the self.print function to avoid broken pipes
        ip_info.print = do_nothing
        return ip_info

    def create_asn_obj(self, mock_db):
        return ASN(mock_db)

    def create_leak_detector_obj(self, mock_db):
        # this file will be used for storing the module output
        # and deleted when the tests are done
        test_pcap = "dataset/test7-malicious.pcap"
        yara_rules_path = "tests/yara_rules_for_testing/rules/"
        compiled_yara_rules_path = "tests/yara_rules_for_testing/compiled/"
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            leak_detector = LeakDetector(
                self.logger,
                "dummy_output_dir",
                6379,
                self.dummy_termination_event,
            )
            leak_detector.db.rdb = mock_db
        # override the self.print function to avoid broken pipes
        leak_detector.print = do_nothing
        # this is the path containing 1 yara rule for testing, it matches every pcap
        leak_detector.yara_rules_path = yara_rules_path
        leak_detector.compiled_yara_rules_path = compiled_yara_rules_path
        leak_detector.pcap = test_pcap
        return leak_detector

    def create_profiler_obj(self, mock_db):
        dummy_semaphore = Semaphore(0)
        profiler = Profiler(
            self.logger,
            "output/",
            6379,
            self.dummy_termination_event,
            is_profiler_done=dummy_semaphore,
            profiler_queue=self.input_queue,
            is_profiler_done_event=self.dummy_termination_event,
        )

        # override the self.print function to avoid broken pipes
        profiler.print = do_nothing
        profiler.whitelist_path = "tests/test_whitelist.conf"
        profiler.db = mock_db
        return profiler

    def create_redis_manager_obj(self, main):
        return RedisManager(main)

    def create_process_manager_obj(self):
        return ProcessManager(self.create_main_obj(""))

    def create_utils_obj(self):
        return utils

    def create_threatintel_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            threatintel = ThreatIntel(
                self.logger,
                "dummy_output_dir",
                6379,
                self.dummy_termination_event,
            )
            threatintel.db = mock_db

        # override the self.print function to avoid broken pipes
        threatintel.print = do_nothing
        return threatintel

    def create_update_manager_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            update_manager = UpdateManager(
                self.logger,
                "dummy_output_dir",
                6379,
                self.dummy_termination_event,
            )

            update_manager.db.rdb = mock_db

        # override the self.print function to avoid broken pipes
        update_manager.print = do_nothing
        return update_manager

    def create_whitelist_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            whitelist = Whitelist(self.logger, mock_db)
            whitelist.db.rdb = mock_db

        # override the self.print function to avoid broken pipes
        whitelist.print = do_nothing
        whitelist.whitelist_path = "tests/test_whitelist.conf"
        return whitelist

    def create_flow_handler_obj(self, flow, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            symbol = SymbolHandler(self.logger, mock_db)
            flow_handler = FlowHandler(mock_db, symbol, flow)
            return flow_handler

    def create_horizontal_portscan_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            horizontal_ps = HorizontalPortscan(mock_db)
            return horizontal_ps

    def create_vertical_portscan_obj(self, mock_db):
        with patch.object(DBManager, "create_sqlite_db", return_value=Mock()):
            vertical_ps = VerticalPortscan(mock_db)
            return vertical_ps

