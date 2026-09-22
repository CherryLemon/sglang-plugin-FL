"""DeepSeek V4.1 integration for the pinned SGLang 0.5.18 backport."""


def install():
    # Keep optional model dependencies out of normal plugin discovery.
    from .backend import install_backend

    return install_backend()
