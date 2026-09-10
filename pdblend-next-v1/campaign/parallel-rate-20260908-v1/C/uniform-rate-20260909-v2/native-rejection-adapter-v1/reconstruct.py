"""Append an independently reclassified view of a frozen native-refusal cell."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p

CLASSIFIER = dict(path=str(HERE / 'cell_audit.py'),
    sha256='6ab69ab0519fbe8c3680523e47d6e4009c8b42643ae375aedf1c1ba7f8fe848d')
METRICS = dict(path=str(HERE / 'metrics.py'),
    sha256='366b80bcfc85bb0f27e5397a6e26560e9fa65530d9859677767318ac7db3e13b')


def audit(checkpoint):
    reference = checkpoint if isinstance(checkpoint, dict) else p.ref(checkpoint)
    classifier = p.load(CLASSIFIER, 'C_native_refusal_exact_classification')
    observation = p.load(classifier.PARENT, 'C_native_refusal_original_full_cell').audit(reference)
    revised = p.load(METRICS, 'C_native_refusal_independent_metrics').audit_checkpoint(reference['path'])
    for name, value in revised.items():
        if name != 'zero_output_diagnosis':
            p.need(observation[name] == value, 'native classification changed an existing metric: ' + name)
    proof = revised.get('zero_output_diagnosis') or {}
    p.need(proof.get('schema') == 'C-Eco-native128-refusal-and-deadline-audit-v1'
        and proof['checkpoint'] == reference, 'exact native refusal proof required')
    observation['zero_output_diagnosis'] = proof
    observation = classifier.classify(observation)
    observation['verification']['native_reclassification'] = dict(
        passed=True, all_metric_values_equal_to_original_full_cell_audit=True,
        original_checkpoint_unchanged=reference, classifier=CLASSIFIER, metrics=METRICS,
        auditor=p.ref(__file__))
    return observation


def audit_checkpoint(path):
    return audit(p.ref(path))


def publish(checkpoint, original_observation, out):
    out = Path(out)
    p.need(not out.exists(), 'fresh derived observation directory required')
    result = audit(checkpoint)
    original = p.checked(original_observation)
    p.need(original['checkpoint'] == checkpoint, 'original observation belongs to another checkpoint')
    compared = []
    for key in ('n_expected', 'completed_work_requests', 'good_requests', 'generated_tokens',
        'expected_generated_tokens', 'slo_attainment', 'producer_slo_attainment', 'request_timeouts',
        'ttft_avg_s', 'tpot_avg_s', 'energy_j', 'gpu_util', 'gpu_util_per_gpu',
        'request_throughput_rps', 'token_throughput_tps', 'token_throughput_is_exact',
        'actual_output_tokens', 'measurement_duration_s', 'measurement_valid', 'work_complete'):
        p.need(result[key] == original[key], 'published observation metric differs: ' + key)
        compared.append(key)
    out.mkdir(parents=True)
    proof = result['zero_output_diagnosis']
    p.save(out / 'proof.json', proof)
    diagnosis = dict(schema='uniform-baseline-capacity-rejection-diagnosis-v1',
        passed=True, independently_recomputed=True, checkpoint=checkpoint,
        classification=result['baseline_service_failure'], verification=result['verification'],
        no_unknown_errors=True, no_PDB_complete_boundary_claim=True,
        original_observation=original_observation, original_metric_fields_preserved=compared,
        native_proof=p.ref(out / 'proof.json'), no_gpu_actions=True)
    p.save(out / 'diagnosis.json', diagnosis)
    result.update(failure_class='independently_diagnosed_capacity_rejection',
        diagnosis_reference=p.ref(out / 'diagnosis.json'),
        original_observation=original_observation, independent_reclassification_only=True)
    p.save(out / 'observation.json', result)
    files = dict(proof['files'])
    files.update({item['path']: item['sha256'] for item in (CLASSIFIER, METRICS,
        checkpoint, original_observation, p.ref(__file__))})
    classifier = p.load(CLASSIFIER, 'C_native_saved_reclassification_parent')
    files[classifier.PARENT['path']] = classifier.PARENT['sha256']
    for name in ('native_queue.py', 'contract.py', 'run_cells.py', 'prepare_dispatch.py'):
        item = p.ref(HERE / name); files[item['path']] = item['sha256']
    p.save(out / 'source-closure.json', dict(schema='C-native-refusal-classification-source-v1',
        files=files, checkpoint=checkpoint, original_observation=original_observation,
        observation=p.ref(out / 'observation.json'), diagnosis=p.ref(out / 'diagnosis.json'),
        proof=p.ref(out / 'proof.json'), auditor=p.ref(__file__),
        declaration_contract=p.ref(HERE / 'contract.py'), no_gpu_actions=True))
    return p.ref(out / 'observation.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--original-observation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(publish(p.ref(args.checkpoint), p.ref(args.original_observation), args.out))
