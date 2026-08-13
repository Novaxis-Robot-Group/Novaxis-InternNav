from importlib import import_module

_AGENT_IMPORTS = {
    'Agent': 'internnav.agent.base',
    'CmaAgent': 'internnav.agent.cma_agent',
    'DialogAgent': 'internnav.agent.dialog_agent',
    'InternVLAN1Agent': 'internnav.agent.internvla_n1_agent',
    'RdpAgent': 'internnav.agent.rdp_agent',
    'Seq2SeqAgent': 'internnav.agent.seq2seq_agent',
}

__all__ = list(_AGENT_IMPORTS)


def __getattr__(name):
    if name not in _AGENT_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_AGENT_IMPORTS[name])
    agent = getattr(module, name)
    globals()[name] = agent
    return agent
