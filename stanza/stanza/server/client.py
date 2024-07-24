class CoreNLPClient(RobustService):
    """ A client to the Stanford CoreNLP server. """

    DEFAULT_ENDPOINT = "http://localhost:9000"
    DEFAULT_TIMEOUT = 60000
    DEFAULT_THREADS = 5
    DEFAULT_OUTPUT_FORMAT = "serialized"
    DEFAULT_MEMORY = "5G"
    DEFAULT_MAX_CHAR_LENGTH = 100000

    def __init__(self, start_server=StartServer.FORCE_START,
                 endpoint=DEFAULT_ENDPOINT,
                 timeout=DEFAULT_TIMEOUT,
                 threads=DEFAULT_THREADS,
                 annotators=None,
                 pretokenized=False,
                 output_format=None,
                 properties=None,
                 stdout=None,
                 stderr=None,
                 memory=DEFAULT_MEMORY,
                 be_quiet=False,
                 max_char_length=DEFAULT_MAX_CHAR_LENGTH,
                 preload=True,
                 classpath=None,
                 **kwargs):

        # whether or not server should be started by client
        self.start_server = start_server
        self.server_props_path = None
        self.server_start_time = None
        self.server_host = None
        self.server_port = None
        self.server_classpath = None
        # validate properties
        validate_corenlp_props(properties=properties, annotators=annotators, output_format=output_format)
        # set up client defaults
        self.properties = properties
        self.annotators = annotators
        self.pretokenized = pretokenized
        self.output_format = output_format
        self._setup_client_defaults()
        # start the server
        if isinstance(start_server, bool):
            warning_msg = f"Setting 'start_server' to a boolean value when constructing {self.__class__.__name__} is deprecated and will stop" + \
                " to function in a future version of stanza. Please consider switching to using a value from stanza.server.StartServer."
            logger.warning(warning_msg)
            start_server = StartServer.FORCE_START if start_server is True else StartServer.DONT_START

        # start the server
        if start_server is StartServer.FORCE_START or start_server is StartServer.TRY_START:
            # record info for server start
            self.server_start_time = datetime.now()
            # set up default properties for server
            self._setup_server_defaults()
            host, port = urlparse(endpoint).netloc.split(":")
            port = int(port)
            assert host == "localhost", "If starting a server, endpoint must be localhost"
            classpath = resolve_classpath(classpath)
            start_cmd = f"java -Xmx{memory} -cp '{classpath}'  edu.stanford.nlp.pipeline.StanfordCoreNLPServer " \
                        f"-port {port} -timeout {timeout} -threads {threads} -maxCharLength {max_char_length} " \
                        f"-quiet {be_quiet} "

            self.server_classpath = classpath
            self.server_host = host
            self.server_port = port

            # set up server defaults
            if self.server_props_path is not None:
                start_cmd += f" -serverProperties {self.server_props_path}"

            # possibly set pretokenized
            if self.pretokenized:
                start_cmd += f" -preTokenized"

            # set annotators for server default
            if self.annotators is not None:
                annotators_str = self.annotators if type(annotators) == str else ",".join(annotators)
                start_cmd += f" -annotators {annotators_str}"

            # specify what to preload, if anything
            if preload:
                if type(preload) == bool:
                    # -preload flag means to preload all default annotators
                    start_cmd += " -preload"
                elif type(preload) == list:
                    # turn list into comma separated list string, only preload these annotators
                    start_cmd += f" -preload {','.join(preload)}"
                elif type(preload) == str:
                    # comma separated list of annotators
                    start_cmd += f" -preload {preload}"

            # set outputFormat for server default
            # if no output format requested by user, set to serialized
            start_cmd += f" -outputFormat {self.output_format}"

            # additional options for server:
            # - server_id
            # - ssl
            # - status_port
            # - uriContext
            # - strict
            # - key
            # - username
            # - password
            # - blockList
            for kw in ['ssl', 'strict']:
                if kwargs.get(kw) is not None:
                    start_cmd += f" -{kw}"
            for kw in ['status_port', 'uriContext', 'key', 'username', 'password', 'blockList', 'server_id']:
                if kwargs.get(kw) is not None:
                    start_cmd += f" -{kw} {kwargs.get(kw)}"
            stop_cmd = None
        else:
            start_cmd = stop_cmd = None
            host = port = None

        super(CoreNLPClient, self).__init__(start_cmd, stop_cmd, endpoint,
                                            stdout, stderr, be_quiet, host=host, port=port, ignore_binding_error=(start_server == StartServer.TRY_START))

        self.timeout = timeout

    def _setup_client_defaults(self):
        """
        Do some processing of annotators and output_format specified for the client.
        If interacting with an externally started server, these will be defaults for annotate() calls.
        :return: None
        """
        # normalize annotators to str
        if self.annotators is not None:
            self.annotators = self.annotators if type(self.annotators) == str else ",".join(self.annotators)

        # handle case where no output format is specified
        if self.output_format is None:
            if type(self.properties) == dict and 'outputFormat' in self.properties:
                self.output_format = self.properties['outputFormat']
            else:
                self.output_format = CoreNLPClient.DEFAULT_OUTPUT_FORMAT

    def _setup_server_defaults(self):
        """
        Set up the default properties for the server.

        The properties argument can take on one of 3 value types

        1. File path on system or in CLASSPATH (e.g. /path/to/server.props or StanfordCoreNLP-french.properties
        2. Name of a Stanford CoreNLP supported language (e.g. french or fr)
        3. Python dictionary (properties written to tmp file for Java server, erased at end)

        In addition, an annotators list and output_format can be specified directly with arguments. These
        will overwrite any settings in the specified properties.

        If no properties are specified, the standard Stanford CoreNLP English server will be launched. The outputFormat
        will be set to 'serialized' and use the ProtobufAnnotationSerializer.
        """

        # ensure properties is str or dict
        if self.properties is None or (not isinstance(self.properties, str) and not isinstance(self.properties, dict)):
            if self.properties is not None:
                logger.warning('properties passed invalid value (not a str or dict), setting properties = {}')
            self.properties = {}
        # check if properties is a string, pass on to server which can handle
        if isinstance(self.properties, str):
            # try to translate to Stanford CoreNLP language name, or assume properties is a path
            if is_corenlp_lang(self.properties):
                if self.properties.lower() in LANGUAGE_SHORTHANDS_TO_FULL:
                    self.properties = LANGUAGE_SHORTHANDS_TO_FULL[self.properties]
                logger.info(
                    f"Using CoreNLP default properties for: {self.properties}.  Make sure to have "
                    f"{self.properties} models jar (available for download here: "
                    f"https://stanfordnlp.github.io/CoreNLP/) in CLASSPATH")
            else:
                if not os.path.isfile(self.properties):
                    logger.warning(f"{self.properties} does not correspond to a file path. Make sure this file is in "
                                   f"your CLASSPATH.")
            self.server_props_path = self.properties
        elif isinstance(self.properties, dict):
            # make a copy
            server_start_properties = dict(self.properties)
            if self.annotators is not None:
                server_start_properties['annotators'] = self.annotators
            if self.output_format is not None and isinstance(self.output_format, str):
                server_start_properties['outputFormat'] = self.output_format
            # write desired server start properties to tmp file
            # set up to erase on exit
            tmp_path = write_corenlp_props(server_start_properties)
            logger.info(f"Writing properties to tmp file: {tmp_path}")
            atexit.register(clean_props_file, tmp_path)
            self.server_props_path = tmp_path

    def _request(self, buf, properties, reset_default=False, **kwargs):
        """
        Send a request to the CoreNLP server.

        :param (str | bytes) buf: data to be sent with the request
        :param (dict) properties: properties that the server expects
        :return: request result
        """
        if self.start_server is not StartServer.DONT_START:
            self.ensure_alive()

        try:
            input_format = properties.get("inputFormat", "text")
            if input_format == "text":
                ctype = "text/plain; charset=utf-8"
            elif input_format == "serialized":
                ctype = "application/x-protobuf"
            else:
                raise ValueError("Unrecognized inputFormat " + input_format)
            # handle auth
            if 'username' in kwargs and 'password' in kwargs:
                kwargs['auth'] = requests.auth.HTTPBasicAuth(kwargs['username'], kwargs['password'])
                kwargs.pop('username')
                kwargs.pop('password')
            r = requests.post(self.endpoint,
                              params={'properties': str(properties), 'resetDefault': str(reset_default).lower()},
                              data=buf, headers={'content-type': ctype},
                              timeout=(self.timeout*2)/1000, **kwargs)
            r.raise_for_status()
            return r
        except requests.exceptions.Timeout as e:
            raise TimeoutException("Timeout requesting to CoreNLPServer. Maybe server is unavailable or your document is too long")
        except requests.exceptions.RequestException as e:
            if e.response is not None and e.response.text is not None:
                raise AnnotationException(e.response.text) from e
            elif e.args:
                raise AnnotationException(e.args[0]) from e
            raise AnnotationException() from e

    def annotate(self, text, annotators=None, output_format=None, properties=None, reset_default=None, **kwargs):
        """
        Send a request to the CoreNLP server.

        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (list | string) annotators: list of annotators to use
        :param (str) output_format: output type from server: serialized, json, text, conll, conllu, or xml
        :param (dict) properties: additional request properties (written on top of defaults)
        :param (bool) reset_default: don't use server defaults

        Precedence for settings:

        1. annotators and output_format args
        2. Values from properties dict
        3. Client defaults self.annotators and self.output_format (set during client construction)
        4. Server defaults

        Additional request parameters (apart from CoreNLP pipeline properties) such as 'username' and 'password'
        can be specified with the kwargs.

        :return: request result
        """

        # validate request properties
        validate_corenlp_props(properties=properties, annotators=annotators, output_format=output_format)
        # set request properties
        request_properties = {}

        # start with client defaults
        if self.annotators is not None:
            request_properties['annotators'] = self.annotators
        if self.output_format is not None:
            request_properties['outputFormat'] = self.output_format

        # add values from properties arg
        # handle str case
        if type(properties) == str:
            if is_corenlp_lang(properties):
                properties = {'pipelineLanguage': properties.lower()}
                if reset_default is None:
                    reset_default = True
            else:
                raise ValueError(f"Unrecognized properties keyword {properties}")

        if type(properties) == dict:
            request_properties.update(properties)

        # if annotators list is specified, override with that
        # also can use the annotators field the object was created with
        if annotators is not None and (type(annotators) == str or type(annotators) == list):
            request_properties['annotators'] = annotators if type(annotators) == str else ",".join(annotators)

        # if output format is specified, override with that
        if output_format is not None and type(output_format) == str:
            request_properties['outputFormat'] = output_format

        # make the request
        # if not explicitly set or the case of pipelineLanguage, reset_default should be None
        if reset_default is None:
            reset_default = False
        r = self._request(text.encode('utf-8'), request_properties, reset_default, **kwargs)
        if request_properties["outputFormat"] == "json":
            return r.json()
        elif request_properties["outputFormat"] == "serialized":
            doc = Document()
            parseFromDelimitedString(doc, r.content)
            return doc
        elif request_properties["outputFormat"] in ["text", "conllu", "conll", "xml"]:
            return r.text
        else:
            return r

    def update(self, doc, annotators=None, properties=None):
        if properties is None:
            properties = {}
            properties.update({
                'inputFormat': 'serialized',
                'outputFormat': 'serialized',
                'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
            })
        if annotators:
            properties['annotators'] = annotators if type(annotators) == str else ",".join(annotators)
        with io.BytesIO() as stream:
            writeToDelimitedString(doc, stream)
            msg = stream.getvalue()

        r = self._request(msg, properties)
        doc = Document()
        parseFromDelimitedString(doc, r.content)
        return doc

    def tokensregex(self, text, pattern, filter=False, to_words=False, annotators=None, properties=None):
        # this is required for some reason
        matches = self.__regex('/tokensregex', text, pattern, filter, annotators, properties)
        if to_words:
            matches = regex_matches_to_indexed_words(matches)
        return matches

    def semgrex(self, text, pattern, filter=False, to_words=False, annotators=None, properties=None):
        matches = self.__regex('/semgrex', text, pattern, filter, annotators, properties)
        if to_words:
            matches = regex_matches_to_indexed_words(matches)
        return matches

    def fill_tree_proto(self, tree, proto_tree):
        if tree.label:
            proto_tree.value = tree.label
        for child in tree.children:
            proto_child = proto_tree.child.add()
            self.fill_tree_proto(child, proto_child)

    def tregex(self, text=None, pattern=None, filter=False, annotators=None, properties=None, trees=None):
        # parse is not included by default in some of the pipelines,
        # so we may need to manually override the annotators
        # to include parse in order for tregex to do anything
        if annotators is None and self.annotators is not None:
            assert isinstance(self.annotators, str)
            pieces = self.annotators.split(",")
            if "parse" not in pieces:
                annotators = self.annotators + ",parse"
        else:
            annotators = "tokenize,ssplit,pos,parse"
        if pattern is None:
            raise ValueError("Cannot have None as a pattern for tregex")

        # TODO: we could also allow for passing in a complete document,
        # along with the original text, so that the spans returns are more accurate
        if trees is not None:
            if properties is None:
                properties = {}
            properties['inputFormat'] = 'serialized'
            if 'serializer' not in properties:
                properties['serializer'] = 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
            doc = Document()
            full_text = []
            for tree_idx, tree in enumerate(trees):
                sentence = doc.sentence.add()
                sentence.sentenceIndex = tree_idx
                sentence.tokenOffsetBegin = len(full_text)
                leaves = tree.leaf_labels()
                full_text.extend(leaves)
                sentence.tokenOffsetEnd = len(full_text)
                self.fill_tree_proto(tree, sentence.parseTree)
                for word in leaves:
                    token = sentence.token.add()
                    # the other side uses both value and word, weirdly enough
                    token.value = word
                    token.word = word
                    # without the actual tokenization, at least we can
                    # stop the words from running together
                    token.after = " "
            doc.text = " ".join(full_text)
            with io.BytesIO() as stream:
                writeToDelimitedString(doc, stream)
                text = stream.getvalue()

        return self.__regex('/tregex', text, pattern, filter, annotators, properties)

    def __regex(self, path, text, pattern, filter, annotators=None, properties=None):
        """
        Send a regex-related request to the CoreNLP server.

        :param (str | unicode) path: the path for the regex endpoint
        :param text: raw text for the CoreNLPServer to apply the regex
        :param (str | unicode) pattern: regex pattern
        :param (bool) filter: option to filter sentences that contain matches, if false returns matches
        :param properties: option to filter sentences that contain matches, if false returns matches
        :return: request result
        """
        if self.start_server is not StartServer.DONT_START:
            self.ensure_alive()
        if properties is None:
            properties = {}
            properties.update({
                'inputFormat': 'text',
                'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
            })
        if annotators:
            properties['annotators'] = ",".join(annotators) if isinstance(annotators, list) else annotators

        # force output for regex requests to be json
        properties['outputFormat'] = 'json'
        # if the server is trying to send back character offsets, it
        # should send back codepoints counts as well in case the text
        # has extra wide characters
        properties['tokenize.codepoint'] = 'true'

        try:
            # Error occurs unless put properties in params
            input_format = properties.get("inputFormat", "text")
            if input_format == "text":
                ctype = "text/plain; charset=utf-8"
            elif input_format == "serialized":
                ctype = "application/x-protobuf"
            else:
                raise ValueError("Unrecognized inputFormat " + input_format)
            # change request method from `get` to `post` as required by CoreNLP
            r = requests.post(
                self.endpoint + path, params={
                    'pattern': pattern,
                    'filter': filter,
                    'properties': str(properties)
                },
                data=text.encode('utf-8') if isinstance(text, str) else text,
                headers={'content-type': ctype},
                timeout=(self.timeout*2)/1000,
            )
            r.raise_for_status()
            if r.encoding is None:
                r.encoding = "utf-8"
            return json.loads(r.text)
        except requests.HTTPError as e:
            if r.text.startswith("Timeout"):
                raise TimeoutException(r.text)
            else:
                raise AnnotationException(r.text)
        except json.JSONDecodeError:
            raise AnnotationException(r.text)


    def scenegraph(self, text, properties=None):
        """
        Send a request to the server which processes the text using SceneGraph

        This will require a new CoreNLP release, 4.5.5 or later
        """
        # since we're using requests ourself,
        # check if the server has started or not
        if self.start_server is not StartServer.DONT_START:
            self.ensure_alive()

        if properties is None:
            properties = {}
        # the only thing the scenegraph knows how to use is text
        properties['inputFormat'] = 'text'
        ctype = "text/plain; charset=utf-8"
        # the json output format is much more useful
        properties['outputFormat'] = 'json'
        try:
            r = requests.post(
                self.endpoint + "/scenegraph",
                params={
                    'properties': str(properties)
                },
                data=text.encode('utf-8') if isinstance(text, str) else text,
                headers={'content-type': ctype},
                timeout=(self.timeout*2)/1000,
            )
            r.raise_for_status()
            if r.encoding is None:
                r.encoding = "utf-8"
            return json.loads(r.text)
        except requests.HTTPError as e:
            if r.text.startswith("Timeout"):
                raise TimeoutException(r.text)
            else:
                raise AnnotationException(r.text)
        except json.JSONDecodeError:
            raise AnnotationException(r.text)

