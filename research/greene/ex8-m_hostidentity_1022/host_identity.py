"""Separate controller alias from independently checked physical hostname."""
def checked_host(actual, alias, hostnames):
    if (type(hostnames) is not dict or set(hostnames) != {'gpu01'}
            or type(hostnames['gpu01']) is not str or not hostnames['gpu01']):
        raise ValueError('Invalid frozen hostname mapping')
    if alias != 'gpu01' or actual != hostnames[alias]:
        raise ValueError('Dispatcher alias/physical hostname mismatch')
    return actual
