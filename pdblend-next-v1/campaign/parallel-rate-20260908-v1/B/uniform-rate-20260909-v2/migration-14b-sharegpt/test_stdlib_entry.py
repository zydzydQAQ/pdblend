import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ENTRY=Path(__file__).resolve().parent/'stdlib_entry.py'

class EntryTests(unittest.TestCase):
    def test_thread_executor_ignores_same_directory_queue_script(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            (d/'queue.py').write_text('raise RuntimeError("local declaration queue shadowed standard library")')
            source=d/'probe.py';source.write_text('import asyncio,queue,json\nasync def run(): print(json.dumps(dict(value=await asyncio.to_thread(lambda: 7),queue=queue.__file__)))\nasyncio.run(run())\n')
            failed=subprocess.run([sys.executable,'-B',str(source)],capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            succeeded=subprocess.run([sys.executable,'-B',str(ENTRY),'--source',str(source)],capture_output=True,text=True,check=True)
            result=json.loads(succeeded.stdout)
            self.assertEqual(result['value'],7)
            self.assertNotEqual(Path(result['queue']).parent,d)

if __name__=='__main__':unittest.main()
