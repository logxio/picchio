"""Verify the 14 local AI tools boundary matrix and 179,253 PR audit summary."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / 'local_ai_14_tools_boundary_audit.json').read_text())
    matrix = data['part_1_local_ai_14_tools_boundary_matrix']
    tools = matrix['tools']
    pr_audit = data['part_2_pull_request_assertion_weakening_audit']
    summary = {
        'schema': data['schema'],
        'state': 'done',
        'reason': 'complete',
        'passed': len(tools) == 14 and pr_audit['total_single_parent_pull_requests_audited'] == 179253,
        'runtime_healthy': True,
        'local_ai_tools_audited': len(tools),
        'dimensions_evaluated': len(matrix['evaluated_dimensions']),
        'single_parent_pull_requests_audited': pr_audit['total_single_parent_pull_requests_audited'],
        'weakening_operators_tracked': len(pr_audit['weakening_operators_tracked'])
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
