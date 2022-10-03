class ReplayBuffer(object):
    def __init__(self, rand_state, capacity=1e6):
        '''Initialize replay buffer.

        Parameters
        ----------
        rand_state : numpy.random.RandomState
            Control random numbers
        capacity : int
            Size of replay buffer

        '''
        self._capacity = capacity
        self._rand_state = rand_state
        self._next_idx = 0
        self._memory = []

    def append(self, transition) -> None:
        '''Append transition to replay buffer

        Parameters
        ----------
        transition: NamedTuple
            Tuple defined as ("state", "action", "next_state", "reward", "done", "omega")
        '''
        if self._next_idx >= len(self._memory):
            self._memory.append(transition)
        else:
            self._memory[self._next_idx] = transition
        self._next_idx = int((self._next_idx + 1) % self._capacity)

    def sample(self, batch_size):
        '''Sample mini-batch from replay buffer

        Parameters
        ----------
        batch_size: int
            Size of mini-batch to be retrieved from replay buffer

        '''
        if len(self._memory) < batch_size:
            return None
        indexes = self._rand_state.randint(0, len(self._memory) - 1, size=batch_size)
        batch = []
        for ind in indexes:
            batch.append(self._memory[ind])
        return batch

    def reset(self):
        '''Reset replay buffer

        '''
        self._memory.clear()

    def __len__(self):
        '''Size of current replay buffer

        '''
        return len(self._memory)
