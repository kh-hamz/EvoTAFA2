"""Single-case debugging and sequential queue entrypoints."""


def execute(*args, **kwargs):
    from edgefl.experiments.sessions import execute as run
    return run(*args, **kwargs)


def queue(*args, **kwargs):
    from edgefl.experiments.runner import execute_queue
    return execute_queue(*args, **kwargs)
