def forward(self):
        return self.renderer.forward(
            self.vert_pos,
            self.vert_col,
            self.vert_rad,
            torch.cat([self.cam_pos, self.cam_rot, self.cam_sensor]),
            self.gamma,
            45.0,
        )

