import hashlib


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    Example:

    ```
    python
        '>>> from keras.data_utils import _hash_file
        '>>>_hash_file('/path/to/file.zip')
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    Arguments:
        fpath: path to the file being validated
        algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        The file hash
    """
    if (algorithm is 'sha256') or (algorithm is 'auto' and len(hash) is 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()
