"""One blind, fixed-probability audit of the first admitted query report.

The audit reads the same physical crop. It adds fallible evidence for the native
final reader, without changing the original DAG or treating agreement as truth.
"""
import math
from copy import deepcopy

from ..common import digest
from . import partition_prompts as prompts
from .partition import PartitionDAG
from .question_context import question_context


class QueryRetest:
    def __init__(self, config, case_key):
        self.probability = config.get('retest_probability', 0.0)
        if (type(self.probability) not in {int, float}
                or not math.isfinite(self.probability) or not 0 <= self.probability <= 1):
            raise ValueError('INVALID_RETEST_PROBABILITY')
        self.draw = int(digest(['query-retest-v1', config.get('seed', 0), case_key])[:13], 16) / 16**13
        self.opportunity = None

    def consider(self, run, selected, added):
        query = len(run.dag.nodes) - 1
        eligible = [o for o in added if o['node_index'] == query and o['observed']]
        if self.opportunity is not None or not eligible:
            return None
        original = eligible[0]
        depths = []
        for node in run.dag.nodes:
            depths.append(1 + max((depths[d] for d in node['requires']), default=0))
        self.opportunity = dict(
            probability=self.probability, draw=self.draw,
            selected=self.draw < self.probability, turn=run.turn, crop_id=selected['id'],
            primary_receipt_id=original['receipt_id'],
            features=dict(nodes=len(run.dag.nodes), query_depth=depths[query],
                          edges=sum(len(n['requires']) for n in run.dag.nodes),
                          repairs=run.repairs, previous_reports=len(run.dag.observations),
                          mapping_present=original['compatible_answers'] is not None,
                          crop_area_fraction=(selected['normalized'][2]-selected['normalized'][0])
                          * (selected['normalized'][3]-selected['normalized'][1]) / 1000000),
        )
        if not self.opportunity['selected']:
            return None
        # A separate one-node validator: previous statuses and reports are not
        # evidence for this independent reading and never enter its request.
        shadow = PartitionDAG(list(run.case['allowed_letters']))
        node = run.dag.nodes[query]
        shadow.accept([dict(task=node['task'], target=node['target'], kind='measure',
                            requires=[], box=node['box'])], run.validate_pixels)
        bound = [{k: deepcopy(n[k]) for k in ['index', 'task', 'target', 'kind', 'requires', 'groups', 'box']}
                 for n in shadow.nodes]
        packet = dict(
            allowed_letters=run.case['allowed_letters'], bound_nodes=bound,
            crop_geometry=dict(original_normalized_box=selected['normalized'],
                               original_pixel_xyxy=selected['original_xyxy'],
                               original_size=run.case['original_size']),
            **question_context(run.question, run.case['allowed_letters']),
        )
        audit = dict(
            crop_id=selected['id'], original_query_index=query, semantic_truth_verified=False,
            warning='Independent rereading of the SAME crop, not new pixels or external verification. '
                    'Both reports are fallible. Agreement, repetition and answer completeness do not '
                    'establish truth. Compare both reports against this image and the exact task. '
                    'Disagreement remains unresolved; neither the newer report nor a majority wins automatically. '
                    'An unresolved report does not establish absence.',
        )
        try:
            p, receipt = run.tool(
                'retest_query', prompts.OBSERVE, packet, prompts.observe_tool(bound, shadow.letters),
                selected['id'], include_original=False,
            )
            if 'observations' not in p or set(p) - {'observations', 'answer'}:
                raise ValueError('RETEST_ROOT_FIELDS')
            rows = prompts.bind_observations(p['observations'], bound, p.get('answer'), shadow.letters)
            report = shadow.observe_batch(rows, [0], selected['id'], receipt['receipt_id'])[0]
            self.opportunity['raw_report'] = deepcopy(report)
            audit['report'] = dict(
                observed=report['observed'],
                fact=report['reported']['observation'] if report['observed'] else None,
                identity=report['reported']['identity_evidence'] if report['observed'] else None,
                answer_hypothesis=report['compatible_answers'],
                limitation=report['reported']['reason'],
                limitation_is_unverified_model_report=True,
            )
            a, b = original['compatible_answers'], report['compatible_answers']
            audit['choice_relation'] = ('same' if a == b else 'different') if a and b else 'unmapped'
        except ValueError as exc:
            # No model-output retry: malformed/length outputs remain paid and
            # supply no audit fact. Transport retries retain existing billing.
            audit.update(format_error=str(exc), report=None, choice_relation='unavailable')
            run.rejections.append(dict(stage='retest_query', error=str(exc)))
        self.opportunity['audit'] = deepcopy(audit)
        return audit
