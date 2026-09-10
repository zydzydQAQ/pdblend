"""Verify one system in a fresh interpreter, preserving source import identity."""
import argparse
import support as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--handoff', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    handoff = p.read(args.handoff)
    validator = p.load(handoff['qualification_validator'], 'uniform_handoff_saved_qualification')
    result = validator.verify(handoff['qualification'])
    p.need(result['passed'] and result['independently_recomputed'], 'stage qualification failed')
    p.save(args.out, dict(handoff=p.ref(args.handoff), qualification=result, binding=result['binding']))


if __name__ == '__main__':
    main()
