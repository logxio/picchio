"""Five probe_checks probe families retain missing evidence as UNKNOWN."""
from .vendor.probe_checks import thinking_cap, constraint_enabled


def five_probe_states(receipt, engine_request, reply):
    h=receipt['hardware'];c=receipt['context']
    cap=thinking_cap(engine_request)
    return {
        'hardware': {'state':'OBSERVED' if h['offloaded_layers'] is not None else 'UNKNOWN',
                     'offloaded_layers':h['offloaded_layers'],'os_scope':h['sample_scope']},
        'required_property_order': {'state':'UNKNOWN','reason':'This completion task has no tool schema order probe'},
        'constrained_decoding': {'state':'ON' if engine_request.get('json_schema') or constraint_enabled(engine_request) else 'OFF',
                                'scope':'Engine request json_schema or response_format; JSON output alone is separate evidence'},
        'thinking_budget': {'state':'UNKNOWN' if cap is None else 'OBSERVED', 'cap':cap},
        'context_truncation': {'state':'CAUGHT' if not all(c['engine_nonce_presence']) else 'PRESENT',
                               'nonce_presence':c['engine_nonce_presence'],
                               'engine_truncated':reply.get('truncated')}
    }
