    def remove_decoder_layer(self):
        """Removes a decoder layer from transformer."""
        if len(self.decoder_layers) > 1:
            self.decoder_layers = self.decoder_layers[:-1]
            self.recreate_shrunk_nets()