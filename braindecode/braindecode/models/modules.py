class DropPath(nn.Module):
    """Drop paths, also known as Stochastic Depth, per sample.

        When applied in main path of residual blocks.

        Parameters:
        -----------
        drop_prob: float (default=None)
            Drop path probability (should be in range 0-1).

        Notes
        -----
        Code copied and modified from VISSL facebookresearch:
    https://github.com/facebookresearch/vissl/blob/0b5d6a94437bc00baed112ca90c9d78c6ccfbafb/vissl/models/model_helpers.py#L676
        All rights reserved.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    # Utility function to print DropPath module
    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"