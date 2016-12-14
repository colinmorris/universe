from universe.vectorized.monitoring import Monitor
import gym.wrappers
from universe import vectorized
from vectorize import WeakUnvectorize

class DummyMonitor(object):
    """A placeholder for a yet-to-be-initialized monitor. Replaced upon calling
    configure() on the env that owns it.
    """
    def start(*args, **kwargs):
        pass

class Monitored(gym.wrappers.Monitored, vectorized.Wrapper):
    """Analogue of gym.wrappers.Monitored for vectorized envs.
    """
    metadata = {
        'runtime.vectorized': True
    }

    def __init__(self, env, *start_args, **start_kwargs):
        self._start_args = start_args
        self._start_kwargs = start_kwargs
        super(Monitored, self).__init__(env)

    def _monitor_factory(self, env):
        # Unfortunately we can't create the monitor immediately on construction
        # because we won't know self.n. We need to just store a placeholder 
        # until configure() is called.
        return DummyMonitor()

    def _configure(self, **kwargs):
        super(Monitored, self)._configure(**kwargs)
        unvectorized = [WeakUnvectorize(self.env, i) for i in range(self.env.n)]
        # Maintain a pointer to unvectorized envs to avoid them being GC'd
        self._unvectorized = unvectorized
        self.monitor = Monitor(unvectorized)
        self.monitor.start(*self._start_args, **self._start_kwargs)
        
