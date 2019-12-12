from data_utils._hash_file import _hash_file


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    Arguments:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        Whether the file is valid
    """
    if ((algorithm is 'sha256') or
        (algorithm is 'auto' and len(file_hash) is 64)):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False