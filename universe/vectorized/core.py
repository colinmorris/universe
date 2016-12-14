import gym
from gym import spaces
from universe import error

class Env(gym.Env):
    metadata = {
        'runtime.vectorized': True
    }

    # User should set this!
    n = None

class Wrapper(Env, gym.Wrapper):
    autovectorize = True
    standalone = True

    def __init__(self, env=None):
        super(Wrapper, self).__init__(env)
        if env is not None and not env.metadata.get('runtime.vectorized'):
            if self.autovectorize:
                # Circular dependency :(
                from universe import wrappers
                env = wrappers.Vectorize(env)
            else:
                raise error.Error('This wrapper can only wrap vectorized envs (i.e. where env.metadata["runtime.vectorized"] = True), not {}. Set "self.autovectorize = True" to automatically add a Vectorize wrapper.'.format(env))

        if env is None and not self.standalone:
            raise error.Error('This env requires a non-None env to be passed. Set "self.standalone = True" to allow env to be omitted or None.')

        self.env = env

    def _configure(self, **kwargs):
        super(Wrapper, self)._configure(**kwargs)
        assert self.env.n is not None, "Did not set self.env.n: self.n={} self.env={} self={}".format(self.env.n, self.env, self)
        self.n = self.env.n

class ObservationWrapper(Wrapper, gym.ObservationWrapper):
    pass

class RewardWrapper(Wrapper, gym.RewardWrapper):
    pass

class ActionWrapper(Wrapper, gym.ActionWrapper):
    pass
