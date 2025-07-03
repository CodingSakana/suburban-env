
class DeviceProvider:
    """全局更改设备的device"""

    __devices = ['cpu', 'cuda:0']
    device = __devices[0]
