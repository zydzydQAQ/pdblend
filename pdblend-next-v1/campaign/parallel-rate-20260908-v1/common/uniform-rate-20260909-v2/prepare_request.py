"""Build a release in its own interpreter to isolate system-specific imports."""
import argparse
import support as p
import prepare_release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    request = p.read(args.request)
    if request.get('prepare_adapter'):
        adapter = p.load(request['prepare_adapter'], 'uniform_custom_dispatch_prepare')
        result = adapter.prepare_dispatch(rows=request['rows'], **request['kwargs'])
    else:
        result = prepare_release.prepare(**request['kwargs'])
    p.save(args.out, result)


if __name__ == '__main__':
    main()
