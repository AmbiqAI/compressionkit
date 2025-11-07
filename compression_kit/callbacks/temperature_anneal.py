import keras
from compression_kit.layers.gumbel_softmax_bottleneck import GumbelSoftmaxBottleneck

class TemperatureAnneal(keras.callbacks.Callback):
    """
    Cosine/exponential temperature schedule for Gumbel-Softmax.
    Example: start=1.0 -> end=0.3 over T epochs.
    """
    def __init__(self, gs_layer: GumbelSoftmaxBottleneck, start: float=1.0, end: float=0.3, epochs: int=20, mode="cosine"):
        super().__init__()
        self.gs = gs_layer
        self.start, self.end, self.epochs, self.mode = float(start), float(end), int(epochs), mode

    def on_epoch_begin(self, epoch, logs=None):
        t = min(epoch, self.epochs) / max(1, self.epochs)
        if self.mode == "cosine":
            import math
            tau = self.end + 0.5*(self.start - self.end)*(1 + math.cos(math.pi * t))
        else:  # "exp"
            tau = self.start * (self.end / self.start) ** t
        self.gs.set_temperature(tau)
