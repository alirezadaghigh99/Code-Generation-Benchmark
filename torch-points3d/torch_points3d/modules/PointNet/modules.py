class PointNetSTN3D(BaseLinearTransformSTNkD):
    def __init__(self, local_nn=[3, 64, 128, 1024], global_nn=[1024, 512, 256], batch_size=1):
        super().__init__(MiniPointNet(local_nn, global_nn), global_nn[-1], 3, batch_size)

    def forward(self, x, batch):
        return super().forward(x, x, batch)

