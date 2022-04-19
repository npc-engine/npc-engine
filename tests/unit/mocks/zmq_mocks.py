class Socket:
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._connected = False

    def connect(self, endpoint):
        assert not self._connected
        assert endpoint != ""
        self._connected = True

    def bind(self, endpoint):
        assert not self._connected
        assert endpoint != ""
        self._connected = True

    def close(self):
        assert self._connected
        self._connected = False


class Context:
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._destroyed = False

    def socket(self, sock_type):
        assert not self._destroyed
        return Socket()

    def destroy(self, linger):
        assert not self._destroyed
        self._destroyed = True
